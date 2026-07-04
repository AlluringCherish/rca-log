#!/usr/bin/env python3
"""Convert RCA-Eval RE2-OB telemetry into a unified per-case event stream.

Cases are written to OPAQUE `problem_<index>` folders so the directory name never
leaks the ground truth. The answer key (service, fault, inject_time, source path)
lives in a single `answers.json` at the output root and is read only by the
evaluator — never by the agent. Anomaly baselines are unsupervised and never read
inject_time.txt (see events/extract.py).

Usage:
  python preprocess.py --source-root /path/to/RE2-OB --output-root data/events --cases all
  python preprocess.py --cases problem_000001,checkoutservice_cpu_1   # by problem id or source name
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from events import extract, schema

DEFAULT_SOURCE_ROOT = "/mnt/data/logjun/data/log/Benchmarks/RE2-OB"
RUN_IDS = ("1", "2", "3")
REQUIRED_FILES = ("simple_metrics.csv", "traces.csv", "logts.csv", "cluster_info.json")
ANSWERS_FILENAME = "answers.json"

TYPE_ORDER = {
    schema.METRIC_ANOMALY: 0,
    schema.SPAN_SLOWDOWN: 1,
    schema.ERROR_CODE: 2,
    schema.LOG_PATTERN: 3,
    schema.CALL_EDGE: 4,
    schema.METRIC_SUMMARY: 5,
}
ANOMALY_TYPES = (
    schema.METRIC_ANOMALY,
    schema.SPAN_SLOWDOWN,
    schema.ERROR_CODE,
    schema.LOG_PATTERN,
)

FAULT_TO_METRIC = {"cpu": "cpu", "mem": "mem", "delay": "latency",
                   "loss": "latency", "disk": "diskio", "socket": "socket"}


class RawCase:
    __slots__ = ("problem_id", "source_name", "service", "fault", "run", "run_dir")

    def __init__(self, problem_id, source_name, service, fault, run, run_dir):
        self.problem_id = problem_id
        self.source_name = source_name
        self.service = service
        self.fault = fault
        self.run = run
        self.run_dir = run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--output-root", default="data/re2-ob/events")
    p.add_argument("--cases", default="all",
                   help="'all' or comma-separated problem ids or source names (service_fault_run).")
    p.add_argument("--case-limit", type=int, default=None)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--baseline-end", type=int, default=600)
    p.add_argument("--warmup", type=int, default=120,
                   help="Suppress anomaly events before this relative second (system stabilization).")
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--clean", action="store_true",
                   help="Remove the output root before writing (fresh problem_* numbering).")
    return p.parse_args()


def discover_raw_cases(source_root: str) -> List[RawCase]:
    """Return all RE2-OB runs, deterministically ordered, with stable problem ids.

    Indexing is computed over the FULL sorted set so a given problem_<index>
    always maps to the same source run regardless of any --cases filter.
    """
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    raw: List[Tuple[str, str, str, str, Path]] = []
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir() or "_" not in group_dir.name:
            continue
        parts = group_dir.name.split("_")
        if len(parts) != 2:
            continue
        service, fault = parts
        for run in RUN_IDS:
            run_dir = group_dir / run
            if all((run_dir / f).exists() for f in REQUIRED_FILES):
                raw.append((f"{service}_{fault}_{run}", service, fault, run, run_dir))
    raw.sort(key=lambda item: item[0])
    return [
        RawCase(f"problem_{index:06d}", source_name, service, fault, run, run_dir)
        for index, (source_name, service, fault, run, run_dir) in enumerate(raw, start=1)
    ]


def select_cases(all_cases: List[RawCase], wanted: str, case_limit: Optional[int]) -> List[RawCase]:
    if wanted.strip().lower() != "all":
        ids = {c.strip() for c in wanted.split(",") if c.strip()}
        selected = [c for c in all_cases if c.problem_id in ids or c.source_name in ids]
        matched = {c.problem_id for c in selected} | {c.source_name for c in selected}
        missing = ids - matched
        if missing:
            raise SystemExit(f"Requested cases not found: {sorted(missing)}")
    else:
        selected = list(all_cases)
    if case_limit is not None:
        selected = selected[:case_limit]
    return selected


def assign_ids_and_lines(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events.sort(
        key=lambda e: (
            int(e["window"][0]),
            TYPE_ORDER.get(e["type"], 99),
            -schema.event_magnitude(e),
            str(e.get("service", "")),
        )
    )
    counters: Dict[str, int] = defaultdict(int)
    for event in events:
        prefix = schema.TYPE_PREFIX[event["type"]]
        counters[event["type"]] += 1
        event["id"] = f"{prefix}{counters[event['type']]:04d}"
        event["line"] = schema.render_line(event)
    return events


def merge_topology(existing: Dict[str, Any], topologies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Union of the Online Boutique call graph across cases (structure is common)."""
    services = set(existing.get("services", []))
    edges: Dict[Tuple[str, str], int] = {
        (e["caller"], e["callee"]): int(e.get("total_calls", 0)) for e in existing.get("edges", [])
    }
    for topo in topologies:
        services.update(topo.get("services", []))
        for e in topo.get("edges", []):
            key = (e["caller"], e["callee"])
            edges[key] = edges.get(key, 0) + int(e.get("total_calls", 0))
    return {
        "dataset": "Online Boutique (RCA-Eval RE2-OB)",
        "note": "Common service call graph; total_calls is summed over processed cases.",
        "services": sorted(services),
        "edges": [{"caller": c, "callee": cl, "total_calls": t}
                  for (c, cl), t in sorted(edges.items())],
    }


def process_case(
    case: RawCase,
    output_root: Path,
    window_size: int,
    baseline_end: int,
    z_threshold: float,
    warmup: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    run_dir = case.run_dir
    metric_path = run_dir / "simple_metrics.csv"
    trace_path = run_dir / "traces.csv"
    logts_path = run_dir / "logts.csv"
    logs_path = run_dir / "logs.csv"
    cluster_info_path = run_dir / "cluster_info.json"

    metric_times = pd.read_csv(metric_path, usecols=["time"])
    case_start = int(metric_times["time"].min())
    telemetry_end = int(metric_times["time"].max() - case_start)
    telemetry_end -= telemetry_end % window_size

    with open(cluster_info_path, "r", encoding="utf-8") as f:
        cluster_info = json.load(f)
    level_by_template = extract.load_level_by_template(logs_path) if logs_path.exists() else {}

    summaries, anomalies = extract.extract_metric_events(
        metric_path, case_start, telemetry_end, window_size, baseline_end, z_threshold
    )
    span_events, error_events, call_edges, topology = extract.extract_trace_events(
        trace_path, case_start, telemetry_end, window_size, baseline_end, z_threshold
    )
    log_events = extract.extract_log_events(
        logts_path, cluster_info, level_by_template,
        case_start, telemetry_end, window_size, baseline_end, z_threshold,
    )

    def keep(event: Dict[str, Any]) -> bool:
        if event["type"] in ANOMALY_TYPES and event["window"][0] < warmup:
            return False
        return True

    events = [
        e for e in (summaries + anomalies + span_events + error_events + log_events + call_edges)
        if keep(e)
    ]
    events = assign_ids_and_lines(events)

    counts: Dict[str, int] = defaultdict(int)
    for event in events:
        counts[event["type"]] += 1

    contamination: Dict[str, float] = {}
    for etype in ANOMALY_TYPES:
        typed = [e for e in events if e["type"] == etype]
        if typed:
            inside = sum(1 for e in typed if e["window"][0] < baseline_end)
            contamination[etype] = round(inside / len(typed), 3)

    case_dir = output_root / case.problem_id
    case_dir.mkdir(parents=True, exist_ok=True)
    with (case_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    # meta.json is agent-visible and MUST NOT leak the answer (no service/fault,
    # no source path, no inject_time).
    meta = {
        "case_id": case.problem_id,
        "case_start_unix_seconds": case_start,
        "time_unit": "relative_seconds_from_case_start",
        "start_time": 0,
        "end_time": telemetry_end,
        "window_size_seconds": window_size,
        "baseline_window": [0, baseline_end],
        "warmup_seconds": warmup,
        "z_threshold": z_threshold,
        "services": topology["services"],
        "record_count": len(events),
        "record_count_by_type": dict(counts),
        "baseline_contamination": contamination,
    }
    with (case_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta, topology


def read_inject_time(run_dir: Path) -> Optional[int]:
    path = run_dir / "inject_time.txt"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_cases = discover_raw_cases(args.source_root)
    selected = select_cases(all_cases, args.cases, args.case_limit)
    if not selected:
        raise SystemExit("No cases selected.")

    # Load/merge the answer key so partial runs accumulate answers over time.
    answers_path = output_root / ANSWERS_FILENAME
    answers: Dict[str, Dict[str, Any]] = {}
    if answers_path.exists() and not args.clean:
        with open(answers_path, "r", encoding="utf-8") as f:
            for item in json.load(f).get("cases", []):
                answers[item["problem_id"]] = item

    processed: List[str] = []
    topologies: List[Dict[str, Any]] = []
    for case in selected:
        case_dir = output_root / case.problem_id
        if case_dir.exists() and (case_dir / "meta.json").exists() and not args.overwrite:
            print(f"skip {case.problem_id} ({case.source_name}; exists, use --overwrite)")
        else:
            meta, topology = process_case(case, output_root, args.window_size,
                                          args.baseline_end, args.z_threshold, args.warmup)
            topologies.append(topology)
            counts = meta["record_count_by_type"]
            print(
                f"ok   {case.problem_id} [{case.source_name}]: total={meta['record_count']} "
                + " ".join(f"{k}={counts.get(k, 0)}" for k in schema.EVENT_TYPES)
                + f" contamination={meta['baseline_contamination']}"
            )
        answers[case.problem_id] = {
            "problem_id": case.problem_id,
            "source_name": case.source_name,
            "service": case.service,
            "fault": case.fault,
            "run": case.run,
            "answer_rank": f"{case.service}_{FAULT_TO_METRIC.get(case.fault, case.fault)}",
            "inject_time_unix": read_inject_time(case.run_dir),
            "source_run_dir": str(case.run_dir),
        }
        processed.append(case.problem_id)

    ordered_answers = [answers[k] for k in sorted(answers)]
    with answers_path.open("w", encoding="utf-8") as f:
        json.dump({"dataset": "RCA-Eval RE2-OB", "cases": ordered_answers}, f,
                  ensure_ascii=False, indent=2)

    # Shared Online Boutique topology, one level above the events dir (common to
    # all cases). Merge into any existing shared file so partial runs accumulate.
    topo_path = output_root.parent / "topology.json"
    existing_topo: Dict[str, Any] = {}
    if topo_path.exists() and not args.clean:
        with open(topo_path, "r", encoding="utf-8") as f:
            existing_topo = json.load(f)
    if topologies or not topo_path.exists():
        merged_topo = merge_topology(existing_topo, topologies)
        with topo_path.open("w", encoding="utf-8") as f:
            json.dump(merged_topo, f, ensure_ascii=False, indent=2)

    manifest = {
        "source_root": args.source_root,
        "window_size_seconds": args.window_size,
        "baseline_end": args.baseline_end,
        "warmup_seconds": args.warmup,
        "z_threshold": args.z_threshold,
        "case_count": len(ordered_answers),
        "cases": [a["problem_id"] for a in ordered_answers],
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nprocessed {len(processed)} cases; answers.json now holds {len(ordered_answers)} cases in {output_root}")


if __name__ == "__main__":
    main()
