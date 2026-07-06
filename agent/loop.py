"""ReAct-style RCA loop.

Controller plans/fetches; Analyst interprets and owns Stop + the final top-3.
When the Analyst requests data, the Controller satisfies it; a deterministic
translator is the fallback if the Controller returns nothing usable. A forced
final-ranking call guarantees every case yields an answer.
"""

import json
import time
from typing import Any, Dict, List, Optional

from agent.analyst import Analyst, DEFAULT_FINDINGS, merge_findings
from agent.controller import Controller
from benchmark.re2_ob import Case, ranking_from_final, normalize_ranking
from events.schema import event_magnitude
from events.tools import EventToolRuntime, PATTERN_TO_TOOL
from reasoning.units import UnitDB

MAX_TOOL_RETRIES = 2

# Accumulated key-events digest handed to the Analyst each step (cross-step,
# cross-modal). Per-type quota, strongest |z| first.
DIGEST_QUOTA = {"metric_anomaly": 12, "span_slowdown": 8, "error_code": 4, "log_pattern": 4}


def build_digest(fetched_events: List[Dict[str, Any]], event_index: Dict[str, str]) -> List[str]:
    best: Dict[str, Dict[str, Any]] = {}
    for e in fetched_events:
        eid = e.get("id")
        if eid and eid not in best and e.get("type") in DIGEST_QUOTA:
            best[eid] = e
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for e in best.values():
        by_type.setdefault(e["type"], []).append(e)
    lines: List[str] = []
    for etype, quota in DIGEST_QUOTA.items():
        for e in sorted(by_type.get(etype, []), key=lambda ev: -event_magnitude(ev))[:quota]:
            ln = event_index.get(e["id"])
            if ln:
                lines.append(ln)
    return lines

REASON_BY_KPI = {
    "cpu": "cpu", "mem": "mem", "diskio": "diskio", "socket": "socket",
    "latency.p50": "latency", "latency.p90": "latency",
    "latency.p99": "latency", "latency.self_p99": "latency",
}
# Near-zero in baseline => a present anomaly is a strong injected-resource signature.
DISTINCTIVE_REASONS = ("diskio", "socket")


def _candidate_reason_severity(fetched_events: List[Dict[str, Any]]) -> Dict[tuple, float]:
    """Max metric_anomaly magnitude (|z|) per (candidate component, reason) seen so far."""
    from events.schema import CANDIDATE_COMPONENTS, normalize_component, event_magnitude
    best: Dict[tuple, float] = {}
    for event in fetched_events:
        if event.get("type") != "metric_anomaly":
            continue
        comp = normalize_component(event.get("service"))
        if comp not in CANDIDATE_COMPONENTS:
            continue
        reason = REASON_BY_KPI.get(event.get("attrs", {}).get("kpi"))
        if not reason:
            continue
        key = (comp, reason)
        best[key] = max(best.get(key, 0.0), event_magnitude(event))
    return best


def pad_ranking(
    final_ranking: List[Dict[str, Any]], fetched_events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Ensure a top-3 ranking (requirement: output Top-1/2/3).

    Rank 1 (the Analyst's choice) is preserved. Fills up to 3 from real evidence:
    first the distinctive injected-resource signals (diskio/socket) of the
    suspected top component, then other candidate (component, reason) anomalies by
    severity. Never fabricates: only pairs backed by a metric_anomaly are added.
    """
    result = [dict(r) for r in final_ranking][:3]
    have = {(r.get("component"), r.get("reason")) for r in result}
    best = _candidate_reason_severity(fetched_events)

    if result and len(result) < 3:
        top = result[0].get("component")
        for reason in DISTINCTIVE_REASONS:
            key = (top, reason)
            if len(result) >= 3:
                break
            if key in best and key not in have:
                result.append({"component": top, "reason": reason,
                               "justification": f"injected-resource signature on {top} ({reason})"})
                have.add(key)

    for (comp, reason), sev in sorted(best.items(), key=lambda kv: -kv[1]):
        if len(result) >= 3:
            break
        if (comp, reason) in have:
            continue
        result.append({"component": comp, "reason": reason,
                       "justification": f"metric_anomaly evidence (|z|={sev:.0f})"})
        have.add((comp, reason))
    return result


def tool_call_signature(call: Dict[str, Any]) -> str:
    return json.dumps({"name": call.get("name"), "args": call.get("args", {})},
                      ensure_ascii=False, sort_keys=True)


def suppress_repeated_tool_calls(
    tool_calls: List[Dict[str, Any]], action_history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    executed = set()
    for entry in action_history:
        for call in entry.get("tool_calls", []) or []:
            executed.add(tool_call_signature(call))
    result = []
    for call in tool_calls:
        sig = tool_call_signature(call)
        if sig in executed:
            continue
        executed.add(sig)
        result.append(call)
    return result


def translate_data_requests(
    data_requests: List[Dict[str, Any]], case_context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Deterministic pattern -> tool translation (fallback when the LLM output is empty)."""
    time_range = case_context.get("telemetry_time_range", {})
    default_start = int(time_range.get("start_time", 0))
    default_end = int(time_range.get("end_time", 1440))
    tool_calls: List[Dict[str, Any]] = []
    for req in data_requests:
        tool = PATTERN_TO_TOOL.get(req.get("pattern"))
        if not tool:
            continue
        window = req.get("window") or [default_start, default_end]
        start, end = int(window[0]), int(window[1])
        args: Dict[str, Any]
        if tool == "get_anomaly_events":
            args = {"start_time": start, "end_time": end,
                    "services": [req["service"]] if req.get("service") else []}
        elif tool == "get_metric_events":
            args = {"service": req.get("service", ""),
                    "kpis": _as_kpi_list(req.get("kpi")),
                    "start_time": start, "end_time": end}
        elif tool == "get_trace_events":
            kind = "error_code" if req.get("pattern") == "error_code" else "span_slowdown"
            args = {"start_time": start, "end_time": end,
                    "services": [req["service"]] if req.get("service") else [], "kinds": [kind]}
        elif tool == "get_log_events":
            args = {"service": req.get("service", ""), "start_time": start, "end_time": end, "k": 20}
        elif tool == "get_topology":
            args = {"start_time": start, "end_time": end}
        else:
            continue
        tool_calls.append({"name": tool, "args": args, "reasoning": f"data_request: {req.get('reason', '')}"})
    return tool_calls


def _as_kpi_list(kpi: Any) -> List[str]:
    if not kpi:
        return []
    if isinstance(kpi, list):
        return [str(s) for s in kpi]
    return [str(kpi)]


def execute_tool_calls_with_retry(
    controller: Controller,
    tool_runtime: EventToolRuntime,
    case_context: Dict[str, Any],
    analyst_report: Any,
    data_requests: List[Dict[str, Any]],
    action_history: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    step: int,
    max_steps: int,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    current_calls = tool_calls
    for attempt_no in range(1, MAX_TOOL_RETRIES + 2):
        observations = tool_runtime.execute_tool_calls(current_calls)
        status = bool(current_calls) and all(bool(o.get("status")) for o in observations)
        attempts.append({"attempt": attempt_no, "tool_calls": current_calls, "status": status})
        if status or attempt_no > MAX_TOOL_RETRIES:
            return {"tool_calls": current_calls, "tool_observations": observations, "attempts": attempts}
        # Ask the Controller to correct the failed calls.
        feedback = {"tool_calls": current_calls, "status": False,
                    "feedback": "One or more tool calls failed; return corrected tool_calls with valid names/args."}
        try:
            retry = controller.decide(case_context, analyst_report, data_requests,
                                      action_history + [feedback], step, max_steps)
            retry_calls = suppress_repeated_tool_calls(retry.get("tool_calls", []), action_history)
        except Exception:
            retry_calls = []
        if not retry_calls:
            return {"tool_calls": current_calls, "tool_observations": observations, "attempts": attempts}
        current_calls = retry_calls
    return {"tool_calls": current_calls, "tool_observations": [], "attempts": attempts}


def run_case(
    case: Case,
    controller: Controller,
    analyst: Analyst,
    unit_db: UnitDB,
    max_steps: int = 8,
    verbose: bool = False,
) -> Dict[str, Any]:
    case_context = case.agent_context()
    time_range = case_context["telemetry_time_range"]
    end_time = int(time_range.get("end_time", 1440))
    tool_runtime = EventToolRuntime(case.event_dir)

    fetched_events: List[Dict[str, Any]] = []
    event_index: Dict[str, str] = {}  # id -> rendered line, accumulated across steps
    findings: Dict[str, Any] = dict(DEFAULT_FINDINGS)
    analyst_report: Optional[Dict[str, Any]] = None
    data_requests: List[Dict[str, Any]] = []
    action_history: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    final_ranking: List[Dict[str, Any]] = []
    error: Optional[str] = None

    llm = getattr(controller, "llm", None)

    def drain_metrics() -> List[Dict[str, Any]]:
        return llm.pop_call_metrics() if llm is not None else []

    started = time.perf_counter()
    drain_metrics()  # clear any stragglers before this case
    try:
        for step in range(1, max_steps + 1):
            # --- Controller plans ---
            decision = controller.decide(case_context, analyst_report, data_requests,
                                         action_history, step, max_steps)
            tool_calls = decision.get("tool_calls", [])
            if not tool_calls and data_requests:
                tool_calls = translate_data_requests(data_requests, case_context)
            if not tool_calls and step == 1:
                # Step-1 bundle: fetch all modalities at once so the Analyst sees a
                # large, cross-modal observation in its first call (fewer round-trips).
                tool_calls = [
                    {"name": "get_anomaly_events",
                     "args": {"start_time": 0, "end_time": end_time, "services": []},
                     "reasoning": "broad anomaly scan"},
                    {"name": "get_trace_events",
                     "args": {"start_time": 0, "end_time": end_time, "services": [],
                              "kinds": ["span_slowdown", "error_code"]},
                     "reasoning": "broad trace scan"},
                    {"name": "get_topology",
                     "args": {"start_time": 0, "end_time": end_time},
                     "reasoning": "topology"},
                ]
            tool_calls = suppress_repeated_tool_calls(tool_calls, action_history)
            if not tool_calls:
                # Nothing new to fetch: let the Analyst decide with current evidence.
                tool_calls = []

            # --- Tools execute ---
            exec_result = execute_tool_calls_with_retry(
                controller, tool_runtime, case_context, analyst_report, data_requests,
                action_history, tool_calls, step, max_steps)
            observations = exec_result["tool_observations"]
            tool_calls = exec_result["tool_calls"]

            new_lines: List[str] = []
            for obs in observations:
                if not obs.get("status"):
                    continue
                try:
                    payload = json.loads(obs["observation"])
                except (json.JSONDecodeError, TypeError):
                    continue
                evs = payload.get("events", [])
                lns = payload.get("lines", [])
                for ev, ln in zip(evs, lns):
                    if ev.get("id"):
                        event_index[ev["id"]] = ln
                fetched_events.extend(evs)
                new_lines.extend(lns)
            action_history.append({
                "step": step,
                "tool_calls": tool_calls,
                "statuses": [bool(o.get("status")) for o in observations],
            })

            # --- Select reasoning unit + Analyst interprets ---
            controller_calls = drain_metrics()  # controller decide(s) incl. tool retries
            unit = unit_db.select_unit(fetched_events, phase="normal")
            key_events = build_digest(fetched_events, event_index)
            report = analyst.analyze(case_context, findings, unit, new_lines, step, max_steps,
                                     key_events=key_events)
            analyst_calls = drain_metrics()
            findings = merge_findings(findings, report["findings"])
            analyst_report = report
            data_requests = report["data_requests"]

            steps.append({
                "step": step,
                "tool_calls": tool_calls,
                "tool_lines": new_lines,
                "unit_ids": unit["unit_ids"],
                "analysis": report["analysis"],
                "rankings": report["findings"]["rankings"],
                "stop": report["stop"],
                "data_requests": data_requests,
                "llm_calls": {"controller": controller_calls, "analyst": analyst_calls},
            })
            if verbose:
                print(f"  step {step}: tools={[c['name'] for c in tool_calls]} "
                      f"unit={unit['unit_ids']} stop={report['stop']} "
                      f"top={report['findings']['rankings'][:1]}")

            if report["stop"]:
                if report["final_ranking"]:
                    final_ranking = report["final_ranking"]
                    break
                # stop requested without a valid ranking -> force a final call
                final = _final_call(analyst, case_context, findings, unit_db, fetched_events,
                                    build_digest(fetched_events, event_index), step, max_steps)
                final_ranking = final["final_ranking"]
                steps.append(_final_step(step, final, drain_metrics()))
                break
        else:
            # budget exhausted without stop -> forced final ranking
            final = _final_call(analyst, case_context, findings, unit_db, fetched_events,
                                build_digest(fetched_events, event_index), max_steps, max_steps)
            final_ranking = final["final_ranking"]
            steps.append(_final_step(max_steps, final, drain_metrics()))
    except Exception as exc:  # pragma: no cover - runtime/LLM failures
        error = f"{type(exc).__name__}: {exc}"

    # Guarantee a top-3 from real evidence (requirement: output Top-1/2/3).
    final_ranking = pad_ranking(final_ranking, fetched_events)
    prediction = ranking_from_final(final_ranking)
    if not prediction:
        # last-resort fallback: derive from accumulated rankings
        prediction = normalize_ranking(
            [f"{r.get('component')}_{r.get('reason')}" for r in findings.get("rankings", [])])
    timing = {"all": round(time.perf_counter() - started, 3)}
    timing.update(aggregate_llm_metrics(steps))
    return {
        "case": case,
        "prediction": prediction,
        "final_ranking": final_ranking,
        "findings": findings,
        "steps": steps,
        "timing_s": timing,
        "error": error,
    }


def aggregate_llm_metrics(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-role sums of the per-call KV/timing metrics, flattened for the report."""
    roles = ("controller", "analyst")
    agg = {r: {"n_calls": 0, "cache_hits": 0, "prefill_s": 0.0, "decode_s": 0.0,
               "total_s": 0.0, "gen_tokens": 0, "prefix_tokens": 0, "suffix_tokens": 0,
               "cache_build_s": 0.0} for r in roles}
    for s in steps:
        for role, calls in (s.get("llm_calls") or {}).items():
            if role not in agg:
                continue
            for m in calls or []:
                a = agg[role]
                a["n_calls"] += 1
                a["cache_hits"] += int(bool(m.get("cache_hit")))
                a["gen_tokens"] += int(m.get("gen_tokens", 0))
                a["prefix_tokens"] += int(m.get("prefix_tokens", 0))
                a["suffix_tokens"] += int(m.get("suffix_tokens", 0))
                for k in ("prefill_s", "decode_s", "total_s", "cache_build_s"):
                    a[k] += float(m.get(k, 0.0))
    out: Dict[str, Any] = {}
    for role in roles:
        for k, v in agg[role].items():
            out[f"{role}_{k}"] = round(v, 4) if isinstance(v, float) else v
    return out


def _final_call(analyst, case_context, findings, unit_db, fetched_events, key_events, step, max_steps):
    unit = unit_db.select_unit(fetched_events, phase="final")
    report = analyst.analyze(case_context, findings, unit, [], step, max_steps,
                             phase="final", key_events=key_events)
    report["unit_ids"] = unit["unit_ids"]
    return report


def _final_step(step: int, report: Dict[str, Any], final_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "step": step,
        "phase": "final",
        "unit_ids": report.get("unit_ids", []),
        "analysis": report["analysis"],
        "final_ranking": report["final_ranking"],
        "stop": True,
        "llm_calls": {"controller": [], "analyst": final_calls},
    }
