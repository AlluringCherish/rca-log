"""Per-modality extractors that turn raw RE2-OB telemetry into events.

Design (DiagFusion-style multimodal eventization):
- metrics  -> metric_summary (every window x series) + metric_anomaly (robust z)
- traces   -> span_slowdown (edge/op p99) + error_code (non-OK status) + call_edge
- logs     -> log_pattern (per-template count deviations / new templates)

Anomaly detection is unsupervised: a robust median/MAD baseline over the early
window [0, baseline_end). It never reads inject_time.txt. baseline_end defaults
to 600s, strictly shorter than the dataset's 720s injection offset so the
baseline is not contaminated by the fault for standard cases.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from events import schema
from events.schema import (
    CALL_EDGE,
    CONTEXT_KPIS,
    ERROR_CODE,
    LOG_PATTERN,
    METRIC_ANOMALY,
    METRIC_SUMMARY,
    METRIC_UNITS,
    REASON_KPIS,
    SPAN_SLOWDOWN,
    clean_operation,
    make_event,
    normalize_component,
    parse_logts_column,
    parse_metric_column,
    robust_baseline,
    robust_z,
)


# Absolute deviation floors applied ONLY when the baseline scale is degenerate
# (constant or empty baseline). They sit well below real fault magnitudes but
# above sensor noise, so a signal that is flat-then-flat stays quiet while a
# genuine step change (socket 9->22, diskio 0->1e10) still fires. Calibrated
# from the RE2-OB signal ranges.
DEGENERATE_ABS_FLOOR = {
    "cpu": 0.5,
    "mem": 5e6,
    "diskio": 1e7,
    "socket": 3.0,
    "latency.p50": 0.01,
    "latency.p90": 0.02,
}

MIN_SPAN_SAMPLES = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_metric_events(
    metric_path: Path,
    case_start: int,
    telemetry_end: int,
    window_size: int,
    baseline_end: int,
    z_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (summary_events, anomaly_events)."""
    df = pd.read_csv(metric_path, low_memory=False)
    df["_rel"] = pd.to_numeric(df["time"], errors="coerce") - case_start
    df = df[(df["_rel"] >= 0) & (df["_rel"] < telemetry_end)].copy()
    df["_win"] = (df["_rel"] // window_size * window_size).astype(int)

    metric_cols = [c for c in df.columns if parse_metric_column(c) is not None]
    # Collapse aliased columns (e.g. frontend-external_workload -> frontend).
    col_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for col in metric_cols:
        parsed = parse_metric_column(col)
        if parsed is not None:
            col_map[parsed].append(col)

    means = df.groupby("_win")[metric_cols].mean(numeric_only=True)
    windows = sorted(int(w) for w in means.index)

    summaries: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    for (component, signal), cols in sorted(col_map.items()):
        if not component:
            continue
        # Per-window value = mean across the (aliased) member columns.
        per_window: Dict[int, float] = {}
        for win in windows:
            row = means.loc[win, cols]
            value = float(pd.to_numeric(row, errors="coerce").mean())
            if value == value:  # not NaN
                per_window[win] = value

        baseline_vals = [v for w, v in per_window.items() if w < baseline_end]
        med, smad = robust_baseline(baseline_vals)
        unit = METRIC_UNITS.get(signal, "value")

        window_scores: List[Tuple[int, float, float]] = []  # (win, value, z)
        for win in windows:
            if win not in per_window:
                continue
            value = per_window[win]
            z = robust_z(value, med, smad)
            window_scores.append((win, value, z))
            summaries.append(
                make_event(
                    METRIC_SUMMARY,
                    (win, win + window_size),
                    component,
                    {"kpi": signal, "value": value, "unit": unit, "z": round(z, 2)},
                )
            )

        if signal in CONTEXT_KPIS or signal not in REASON_KPIS:
            continue  # workload/error stay as summaries only
        anomalies.extend(
            _anomaly_events_from_scores(
                component, signal, med, smad, window_scores, window_size, z_threshold
            )
        )

    return summaries, anomalies


def _anomaly_events_from_scores(
    component: str,
    signal: str,
    med: float,
    smad: float,
    window_scores: List[Tuple[int, float, float]],
    window_size: int,
    z_threshold: float,
) -> List[Dict[str, Any]]:
    degenerate = smad <= 1e-9
    floor = DEGENERATE_ABS_FLOOR.get(signal, 0.0)

    flagged = []
    for win, value, z in window_scores:
        if abs(z) < z_threshold:
            continue
        if degenerate and abs(value - med) < floor:
            continue
        flagged.append((win, value, z))

    # Merge runs of consecutive anomalous windows into one event.
    # Emission gate: RE2-OB faults are sustained (~12 min = many windows), so we
    # keep persistent runs (>=2 windows) and only admit a single-window anomaly
    # when its deviation is sharp (>=2x threshold). This drops the isolated
    # moderate blips that dominate benign baseline noise without risking the
    # sustained target fault.
    single_window_z = 2.0 * z_threshold
    events: List[Dict[str, Any]] = []
    run: List[Tuple[int, float, float]] = []

    def flush(run_items: List[Tuple[int, float, float]]) -> None:
        if not run_items:
            return
        peak = max(run_items, key=lambda item: abs(item[2]))
        if len(run_items) < 2 and abs(peak[2]) < single_window_z:
            return
        first_start = run_items[0][0]
        last_end = run_items[-1][0] + window_size
        events.append(
            make_event(
                METRIC_ANOMALY,
                (first_start, last_end),
                component,
                {
                    "kpi": signal,
                    "peak_value": peak[1],
                    "baseline": med,
                    "z": round(peak[2], 2),
                    "direction": "up" if peak[2] >= 0 else "down",
                    "n_windows": len(run_items),
                    "persistent": len(run_items) >= 2,
                },
            )
        )

    for item in flagged:
        if run and item[0] != run[-1][0] + window_size:
            flush(run)
            run = []
        run.append(item)
    flush(run)
    return events


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------

def extract_trace_events(
    trace_path: Path,
    case_start: int,
    telemetry_end: int,
    window_size: int,
    baseline_end: int,
    z_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Return (span_slowdown, error_code, call_edge, topology)."""
    df = pd.read_csv(trace_path, low_memory=False)
    required = {"traceID", "spanID", "parentSpanID", "serviceName", "startTimeMillis", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{trace_path} lacks trace columns: {sorted(missing)}")

    seconds = pd.to_numeric(df["startTimeMillis"], errors="coerce") / 1000.0
    df["_rel"] = seconds - case_start
    df = df[(df["_rel"] >= 0) & (df["_rel"] < telemetry_end)].copy()
    df["_win"] = (df["_rel"] // window_size * window_size).astype(int)
    df["_service"] = df["serviceName"].map(normalize_component)
    op_col = "operationName" if "operationName" in df.columns else "methodName"
    df["_op"] = df[op_col].map(clean_operation) if op_col in df.columns else "<unknown>"
    df["_duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    trace_ids = df["traceID"].fillna("").astype(str)
    span_ids = df["spanID"].fillna("").astype(str)
    parent_ids = df["parentSpanID"].fillna("").astype(str)
    span_keys = trace_ids + "|" + span_ids
    parent_keys = trace_ids + "|" + parent_ids
    service_by_key = dict(zip(span_keys, df["_service"]))
    df["_parent_service"] = parent_keys.map(service_by_key).fillna("")

    if "statusCode" in df.columns:
        status = pd.to_numeric(df["statusCode"], errors="coerce").fillna(0.0)
    else:
        status = pd.Series([0.0] * len(df), index=df.index)
    df["_is_error"] = (status != 0.0).astype(int)

    edge_df = df[(df["_parent_service"] != "") & (df["_service"] != "")].copy()
    if edge_df.empty:
        return [], [], [], {"services": sorted(set(df["_service"]) - {""}), "edges": []}

    # Per (window, caller, callee, op) aggregates.
    grouped = edge_df.groupby(["_win", "_parent_service", "_service", "_op"], sort=True)
    rows: List[Dict[str, Any]] = []
    for (win, caller, callee, op), g in grouped:
        rows.append(
            {
                "win": int(win),
                "caller": str(caller),
                "callee": str(callee),
                "op": str(op),
                "p99": float(g["_duration"].quantile(0.99)),
                "errors": int(g["_is_error"].sum()),
                "calls": int(g.shape[0]),
            }
        )

    span_events = _span_slowdown_events(rows, window_size, baseline_end, z_threshold)
    error_events = _error_code_events(rows, window_size, baseline_end)
    call_edge_events, topology = _call_edge_events(rows, window_size, df)
    return span_events, error_events, call_edge_events, topology


def _span_slowdown_events(
    rows: List[Dict[str, Any]],
    window_size: int,
    baseline_end: int,
    z_threshold: float,
) -> List[Dict[str, Any]]:
    baseline: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        if r["win"] < baseline_end and r["p99"] == r["p99"]:
            baseline[(r["caller"], r["callee"], r["op"])].append(r["p99"])

    stats = {key: robust_baseline(vals) for key, vals in baseline.items()}
    events: List[Dict[str, Any]] = []
    for r in rows:
        if r["calls"] < MIN_SPAN_SAMPLES or r["p99"] != r["p99"]:
            continue
        med, smad = stats.get((r["caller"], r["callee"], r["op"]), (0.0, 0.0))
        z = robust_z(r["p99"], med, smad)
        if z < z_threshold:
            continue
        events.append(
            make_event(
                SPAN_SLOWDOWN,
                (r["win"], r["win"] + window_size),
                r["callee"],
                {
                    "kpi": "latency.p99",
                    "caller": r["caller"],
                    "callee": r["callee"],
                    "operation": r["op"],
                    "p99_us": r["p99"],
                    "baseline_us": med,
                    "z": round(z, 2),
                    "n": r["calls"],
                },
            )
        )
    return events


def _error_code_events(
    rows: List[Dict[str, Any]],
    window_size: int,
    baseline_end: int,
) -> List[Dict[str, Any]]:
    baseline_max: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in rows:
        if r["win"] < baseline_end:
            key = (r["caller"], r["callee"], r["op"])
            baseline_max[key] = max(baseline_max[key], r["errors"])

    events: List[Dict[str, Any]] = []
    for r in rows:
        if r["errors"] <= 0:
            continue
        key = (r["caller"], r["callee"], r["op"])
        if r["errors"] <= baseline_max.get(key, 0):
            continue
        rate = r["errors"] / max(r["calls"], 1)
        events.append(
            make_event(
                ERROR_CODE,
                (r["win"], r["win"] + window_size),
                r["callee"],
                {
                    "caller": r["caller"],
                    "callee": r["callee"],
                    "operation": r["op"],
                    "error_count": r["errors"],
                    "call_count": r["calls"],
                    "status_codes": f"nonzero_rate={rate:.2f}",
                },
            )
        )
    return events


def _call_edge_events(
    rows: List[Dict[str, Any]],
    window_size: int,
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_edge_window: Dict[Tuple[int, str, str], Dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "errors": 0}
    )
    edge_totals: Dict[Tuple[str, str], int] = defaultdict(int)
    for r in rows:
        key = (r["win"], r["caller"], r["callee"])
        per_edge_window[key]["calls"] += r["calls"]
        per_edge_window[key]["errors"] += r["errors"]
        edge_totals[(r["caller"], r["callee"])] += r["calls"]

    events: List[Dict[str, Any]] = []
    for (win, caller, callee), agg in sorted(per_edge_window.items()):
        events.append(
            make_event(
                CALL_EDGE,
                (win, win + window_size),
                callee,
                {
                    "caller": caller,
                    "callee": callee,
                    "count": agg["calls"],
                    "error_count": agg["errors"],
                },
            )
        )

    services = sorted({s for s in df["_service"].unique() if s})
    topology = {
        "services": services,
        "edges": [
            {"caller": caller, "callee": callee, "total_calls": total}
            for (caller, callee), total in sorted(edge_totals.items())
        ],
    }
    return events, topology


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

def extract_log_events(
    logts_path: Path,
    cluster_info: Dict[str, Any],
    level_by_template: Dict[str, str],
    case_start: int,
    telemetry_end: int,
    window_size: int,
    baseline_end: int,
    z_threshold: float,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(logts_path, low_memory=False)
    if "time" not in df.columns:
        return []
    df["_rel"] = pd.to_numeric(df["time"], errors="coerce") - case_start
    df["_win"] = (df["_rel"] // window_size * window_size).astype("Int64")
    df = df[(df["_win"] >= 0) & (df["_win"] < telemetry_end)].copy()
    if df.empty:
        return []

    template_cols = [c for c in df.columns if parse_logts_column(c) is not None]
    windowed = df.groupby("_win")[template_cols].sum(numeric_only=True)
    windows = sorted(int(w) for w in windowed.index)

    events: List[Dict[str, Any]] = []
    for col in template_cols:
        parsed = parse_logts_column(col)
        if parsed is None:
            continue
        service, template_id = parsed
        info = cluster_info.get(template_id, {})
        template = info.get("template", "")
        level = level_by_template.get(template_id, "unknown")

        counts = {int(w): float(windowed.loc[w, col]) for w in windows}
        baseline_vals = [c for w, c in counts.items() if w < baseline_end]
        med, smad = robust_baseline(baseline_vals)
        baseline_seen = any(v > 0 for v in baseline_vals)

        for win in windows:
            count = counts.get(win, 0.0)
            if count <= 0:
                continue
            z = robust_z(count, med, smad)
            new_template = (not baseline_seen) and count > 0
            if abs(z) < z_threshold and not new_template:
                continue
            events.append(
                make_event(
                    LOG_PATTERN,
                    (win, win + window_size),
                    service,
                    {
                        "template_id": template_id,
                        "template": template,
                        "level": level,
                        "count": int(count),
                        "baseline": med,
                        "z": round(z, 2),
                        "new_template": new_template,
                    },
                )
            )
    return events


def load_level_by_template(logs_path: Path) -> Dict[str, str]:
    """Dominant log level per cluster_id, from one light pass over logs.csv."""
    try:
        df = pd.read_csv(logs_path, usecols=["cluster_id", "level"], low_memory=False)
    except (ValueError, OSError):
        return {}
    df = df.dropna(subset=["cluster_id"])
    if df.empty:
        return {}
    df["_tid"] = df["cluster_id"].apply(
        lambda v: str(int(v)) if float(v).is_integer() else str(v)
    )
    df["level"] = df["level"].fillna("unknown").astype(str)
    level_by_template: Dict[str, str] = {}
    for tid, group in df.groupby("_tid"):
        mode = group["level"].mode()
        level_by_template[str(tid)] = str(mode.iloc[0]) if not mode.empty else "unknown"
    return level_by_template
