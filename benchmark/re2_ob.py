"""RE2-OB case model and evaluation for the event-based RCA prototype.

Ground truth (service, fault, inject_time) is used ONLY here for scoring. The
agent-facing `agent_context()` deliberately excludes all of it. Candidate
component/reason spaces match the RCA-Eval Online Boutique setup.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


FAULT_TO_METRIC = {
    "cpu": "cpu",
    "mem": "mem",
    "delay": "latency",
    "loss": "latency",
    "disk": "diskio",
    "socket": "socket",
}

CANDIDATE_COMPONENTS = [
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "productcatalogservice",
    "recommendationservice",
]

CANDIDATE_REASONS = ["cpu", "mem", "diskio", "latency", "socket"]

EVENT_TYPES = [
    "metric_anomaly",
    "metric_summary",
    "span_slowdown",
    "error_code",
    "log_pattern",
    "call_edge",
]


@dataclass(frozen=True)
class Case:
    case_id: str
    event_dir: str
    service: str  # ground truth component -- scoring only
    fault: str  # ground truth fault -- scoring only
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.case_id

    @property
    def answer_metric(self) -> str:
        return FAULT_TO_METRIC.get(self.fault, self.fault)

    @property
    def answer_rank(self) -> str:
        return f"{self.service}_{self.answer_metric}"

    def time_range(self) -> Dict[str, Any]:
        return {
            "unit": "relative_seconds_from_case_start",
            "start_time": int(self.meta.get("start_time", 0)),
            "end_time": int(self.meta.get("end_time", 1440)),
        }

    def agent_context(self) -> Dict[str, Any]:
        """Case context handed to the Controller and Analyst.

        Never includes inject_time, fault, or the ground-truth service.
        """
        return {
            "case_name": self.name,
            "dataset": "Online Boutique (RCA-Eval RE2-OB)",
            "telemetry_time_range": self.time_range(),
            "window_size_seconds": int(self.meta.get("window_size_seconds", 30)),
            "event_types": list(EVENT_TYPES),
            "services": list(self.meta.get("services", [])),
            "possible_root_cause_components": list(CANDIDATE_COMPONENTS),
            "possible_root_cause_reasons": list(CANDIDATE_REASONS),
            "prediction_format": (
                "Return a ranked list (top-3) of root causes, each with `component` "
                "from possible_root_cause_components and `reason` from possible_root_cause_reasons."
            ),
        }


ANSWERS_FILENAME = "answers.json"


def load_answers(events_root: str) -> Dict[str, Dict[str, Any]]:
    """Answer key: problem_id -> {service, fault, source_name, ...}. Eval only."""
    path = os.path.join(events_root, ANSWERS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Answer key not found: {path}. Run preprocess.py to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    answers: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("cases", []):
        pid = item.get("problem_id")
        if pid:
            answers[pid] = item
    return answers


def discover_event_cases(
    events_root: str,
    cases: Optional[Iterable[str]] = None,
    case_limit: Optional[int] = None,
) -> List[Case]:
    if not os.path.isdir(events_root):
        raise FileNotFoundError(f"Events root does not exist: {events_root}")

    answers = load_answers(events_root)
    # A requested case may be given by problem id or by source name.
    source_to_pid = {a.get("source_name"): pid for pid, a in answers.items()}
    wanted: Optional[set] = None
    if cases:
        wanted = set()
        for name in cases:
            if name in answers:
                wanted.add(name)
            elif name in source_to_pid:
                wanted.add(source_to_pid[name])
            else:
                wanted.add(name)  # keep for a clear not-found error below

    discovered: List[Case] = []
    for name in sorted(os.listdir(events_root)):
        case_dir = os.path.join(events_root, name)
        meta_path = os.path.join(case_dir, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        if wanted is not None and name not in wanted:
            continue
        answer = answers.get(name)
        if answer is None:
            continue  # a preprocessed dir with no answer entry; skip
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        discovered.append(
            Case(
                case_id=name,
                event_dir=case_dir,
                service=str(answer["service"]),
                fault=str(answer["fault"]),
                meta=meta,
            )
        )

    if wanted is not None:
        missing = wanted - {c.case_id for c in discovered}
        if missing:
            raise FileNotFoundError(
                f"Requested cases not found under {events_root}: {sorted(missing)}"
            )
    if case_limit is not None:
        discovered = discovered[:case_limit]
    if not discovered:
        raise FileNotFoundError(f"No preprocessed event cases found under {events_root}")
    return discovered


def normalize_rank_item(item: object) -> Optional[str]:
    if isinstance(item, dict):
        service = item.get("service") or item.get("entity") or item.get("component")
        metric = item.get("metric") or item.get("fault") or item.get("reason")
        if service and metric:
            item = f"{service}_{metric}"
        else:
            return None
    if not isinstance(item, str):
        return None
    item = item.strip().replace(" ", "_")
    if not item or item.lower() in {"unknown", "none", "null"}:
        return None
    item = item.replace("_latency-90", "_latency").replace("_lat_90", "_latency")
    return item


def normalize_ranking(ranking: Iterable[object], limit: int = 5) -> List[str]:
    seen = set()
    result: List[str] = []
    for raw in ranking or []:
        item = normalize_rank_item(raw)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= limit:
            break
    return result


def split_rank(rank: str) -> Tuple[str, str]:
    if "_" not in rank:
        return rank, "unknown"
    service, metric = rank.split("_", 1)
    if metric in {"delay", "loss"}:
        metric = "latency"
    if metric == "disk":
        metric = "diskio"
    return service.replace("-db", ""), metric


def ranking_from_final(final_ranking: Iterable[Dict[str, Any]]) -> List[str]:
    """Convert Analyst final_ranking objects into `component_reason` strings."""
    ranks: List[str] = []
    for item in final_ranking or []:
        if not isinstance(item, dict):
            continue
        component = item.get("component")
        reason = item.get("reason")
        if component and reason:
            ranks.append(f"{component}_{reason}")
    return ranks


def evaluate_cases(records: List[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    summary: Dict[str, Dict[str, float]] = {}

    for record in records:
        case: Case = record["case"]  # type: ignore[assignment]
        prediction = normalize_ranking(record.get("prediction", []))
        components: List[str] = []
        reasons: List[str] = []
        fine: List[Tuple[str, str]] = []
        for rank in prediction:
            component, reason = split_rank(rank)
            if component not in components:
                components.append(component)
            if reason not in reasons:
                reasons.append(reason)
            fine.append((component, reason))

        answer_component = case.service.replace("-db", "")
        answer_reason = case.answer_metric
        answer_fine = (answer_component, answer_reason)
        timing_s = record.get("timing_s", {})
        elapsed_s = timing_s.get("all") if isinstance(timing_s, dict) else None
        row: Dict[str, Any] = {
            "case": case.name,
            "answer": case.answer_rank,
            "prediction": prediction,
            "steps": len(record.get("steps", [])),
            "time_s": elapsed_s,
        }
        for k in range(1, 6):
            row[f"top_{k}_component"] = answer_component in components[:k]
            row[f"top_{k}_reason"] = answer_reason in reasons[:k]
            row[f"top_{k}_both"] = answer_fine in fine[:k]
        rows.append(row)

    group_names = ["overall"] + sorted({r["answer"].split("_", 1)[1] for r in rows})
    for group_name in group_names:
        group_rows = (
            rows
            if group_name == "overall"
            else [r for r in rows if r["answer"].endswith(f"_{group_name}")]
        )
        if not group_rows:
            continue
        summary[group_name] = {"n": len(group_rows)}
        for k in range(1, 6):
            summary[group_name][f"top_{k}_component"] = sum(
                bool(r[f"top_{k}_component"]) for r in group_rows
            ) / len(group_rows)
            summary[group_name][f"top_{k}_reason"] = sum(
                bool(r[f"top_{k}_reason"]) for r in group_rows
            ) / len(group_rows)
            summary[group_name][f"top_{k}_both"] = sum(
                bool(r[f"top_{k}_both"]) for r in group_rows
            ) / len(group_rows)
        time_values = [float(r["time_s"]) for r in group_rows if r.get("time_s") is not None]
        if time_values:
            summary[group_name]["time_s_total"] = sum(time_values)
            summary[group_name]["time_s_mean"] = mean(time_values)
            summary[group_name]["time_s_median"] = median(time_values)

    return {"summary": summary, "rows": rows}


def write_outputs(output_dir: str, records: List[Dict[str, object]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    serializable = []
    for record in records:
        case: Case = record["case"]  # type: ignore[assignment]
        serializable.append(
            {
                "case": case.name,
                "answer": case.answer_rank,
                "prediction": normalize_ranking(record.get("prediction", [])),
                "final_ranking": record.get("final_ranking", []),
                "error": record.get("error"),
                "steps": record.get("steps", []),
                "timing_s": record.get("timing_s"),
            }
        )
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    evaluation = evaluate_cases(records)
    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "predictions.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "answer", "prediction", "error"])
        writer.writeheader()
        for row in serializable:
            writer.writerow(
                {
                    "case": row["case"],
                    "answer": row["answer"],
                    "prediction": json.dumps(row["prediction"]),
                    "error": row["error"],
                }
            )
