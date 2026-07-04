"""Shared constants and helpers for the unified RCA event stream.

An event is a compact multimodal record derived from RE2-OB telemetry, in the
spirit of DiagFusion (Robust Failure Diagnosis of Microservice System through
Multimodal Data): metrics, traces, and logs are all reduced to timestamped,
service-scoped events with a small typed attribute bag.

Event record shape (one JSON object per line in events.jsonl):
    {
      "id":      "<TYPE_PREFIX><counter>",     # e.g. M0007, S0003
      "type":    "metric_anomaly" | ...,
      "window":  [start_sec, end_sec],          # relative seconds from case start
      "service": "checkoutservice",             # normalized owning component
      "attrs":   {...},                          # type-specific fields (metrics use `kpi`)
      "line":    "[metric_anomaly] id=... ",     # pre-rendered compact string for the Analyst
    }

Sorting/severity is not stored on the event; `event_magnitude()` derives it on the
fly from attrs (|z|, or error_count for error_code).
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Event types and id prefixes
# ---------------------------------------------------------------------------

METRIC_SUMMARY = "metric_summary"
METRIC_ANOMALY = "metric_anomaly"
SPAN_SLOWDOWN = "span_slowdown"
ERROR_CODE = "error_code"
LOG_PATTERN = "log_pattern"
CALL_EDGE = "call_edge"

EVENT_TYPES = [
    METRIC_ANOMALY,
    METRIC_SUMMARY,
    SPAN_SLOWDOWN,
    ERROR_CODE,
    LOG_PATTERN,
    CALL_EDGE,
]

TYPE_PREFIX = {
    METRIC_ANOMALY: "M",
    METRIC_SUMMARY: "U",
    SPAN_SLOWDOWN: "S",
    ERROR_CODE: "E",
    LOG_PATTERN: "L",
    CALL_EDGE: "G",
}


# ---------------------------------------------------------------------------
# Component naming
# ---------------------------------------------------------------------------

SERVICE_ALIASES = {
    "frontendservice": "frontend",
    "frontend-external": "frontend",
    "redis-cart": "redis",
}

KNOWN_COMPONENTS = {
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "productcatalogservice",
    "recommendationservice",
    "frontend",
    "adservice",
    "cartservice",
    "redis",
    "paymentservice",
    "shippingservice",
    "loadgenerator",
}

CANDIDATE_COMPONENTS = {
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "productcatalogservice",
    "recommendationservice",
}

# Metric KPIs that can be a root-cause *reason*. workload/error are context
# only and are stored as summaries but never eventized as metric_anomaly.
REASON_KPIS = {"cpu", "mem", "diskio", "socket", "latency.p50", "latency.p90"}
RESOURCE_KPIS = {"cpu", "mem", "diskio", "socket"}
LATENCY_KPIS = {"latency.p50", "latency.p90"}
CONTEXT_KPIS = {"workload", "error"}

METRIC_UNITS = {
    "cpu": "percent",
    "mem": "bytes",
    "diskio": "bytes/s",
    "socket": "count",
    "workload": "rps",
    "error": "count",
    "latency.p50": "s",
    "latency.p90": "s",
}


def normalize_component(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    raw = str(value).strip().lower()
    if raw in SERVICE_ALIASES:
        return SERVICE_ALIASES[raw]
    if raw in KNOWN_COMPONENTS:
        return raw
    for component in sorted(KNOWN_COMPONENTS, key=len, reverse=True):
        if raw.startswith(component):
            return component
    return raw


def parse_metric_column(column: str) -> Optional[Tuple[str, str]]:
    """`checkoutservice_latency-90` -> ("checkoutservice", "latency.p90")."""
    if column == "time" or column.startswith("_") or "_" not in column:
        return None
    component, signal = column.split("_", 1)
    component = normalize_component(component)
    signal = signal.replace("latency-50", "latency.p50").replace("latency-90", "latency.p90")
    return component, signal


def parse_logts_column(column: str) -> Optional[Tuple[str, str]]:
    """`currencyservice_1` -> ("currencyservice", "1") where 1 is a template id."""
    if column == "time" or "_" not in column:
        return None
    service, template_id = column.rsplit("_", 1)
    return normalize_component(service), template_id


def clean_operation(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<unknown>"
    text = str(value).strip()
    if not text:
        return "<unknown>"
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text or "<unknown>"


# ---------------------------------------------------------------------------
# Numeric formatting and robust statistics
# ---------------------------------------------------------------------------

def format_number(value: Any) -> str:
    if value is None:
        return "null"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "null"
    if abs(number) >= 1000 and number.is_integer():
        return str(int(number))
    if abs(number) >= 100:
        return f"{number:.1f}".rstrip("0").rstrip(".")
    if abs(number) >= 10:
        return f"{number:.2f}".rstrip("0").rstrip(".")
    if abs(number) >= 1:
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return f"{number:.6g}"


def format_z(value: float) -> str:
    if not math.isfinite(value):
        return "0.0"
    return f"{value:+.1f}"


_MAD_SCALE = 1.4826


def robust_baseline(values: List[float]) -> Tuple[float, float]:
    """Return (median, scaled MAD) for a baseline sample.

    Scaled MAD ~ a robust standard-deviation estimate. Returns 0.0 scale when
    the sample is empty, a single point, or has (near) zero dispersion.
    """
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return 0.0, 0.0
    med = _median(clean)
    if len(clean) < 2:
        return med, 0.0
    deviations = [abs(v - med) for v in clean]
    mad = _median(deviations)
    return med, _MAD_SCALE * mad


def robust_z(value: float, median_value: float, scaled_mad: float) -> float:
    """Robust deviation of `value` from a baseline median.

    Uses scaled MAD when available; otherwise falls back to a relative-delta
    denominator so constant-but-nonzero baselines (common for socket counts and
    latency) still produce a bounded, comparable score instead of exploding.
    """
    if not math.isfinite(value) or not math.isfinite(median_value):
        return 0.0
    if math.isfinite(scaled_mad) and scaled_mad > 1e-9:
        z = (value - median_value) / scaled_mad
    else:
        denom = max(0.05 * abs(median_value), 1e-6)
        z = (value - median_value) / denom
    # Bound the score so degenerate (near-zero) baselines don't produce
    # astronomical values that dominate formatting and severity ranking.
    return max(-1000.0, min(1000.0, z))


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# ---------------------------------------------------------------------------
# Event construction + line rendering
# ---------------------------------------------------------------------------

def make_event(
    event_type: str,
    window: Tuple[int, int],
    service: str,
    attrs: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "type": event_type,
        "window": [int(window[0]), int(window[1])],
        "service": service,
        "attrs": attrs,
    }


def event_magnitude(event: Dict[str, Any]) -> float:
    """Derived sort magnitude (replaces the removed `severity` field).

    |z| for z-scored events, error_count for error_code, 0 otherwise.
    """
    attrs = event.get("attrs", {})
    if event.get("type") == ERROR_CODE:
        return float(attrs.get("error_count", 0) or 0)
    z = attrs.get("z")
    if z is None:
        return 0.0
    try:
        return abs(float(z))
    except (TypeError, ValueError):
        return 0.0


def render_line(event: Dict[str, Any]) -> str:
    etype = event["type"]
    win = f"{event['window'][0]}-{event['window'][1]}"
    eid = event["id"]
    service = event.get("service", "")
    a = event.get("attrs", {})

    if etype == METRIC_ANOMALY:
        return (
            f"[metric_anomaly] id={eid} window={win} service={service} "
            f"kpi={a.get('kpi')} peak={format_number(a.get('peak_value'))} "
            f"base={format_number(a.get('baseline'))} z={format_z(a.get('z', 0.0))} "
            f"dir={a.get('direction')} n_windows={a.get('n_windows')} "
            f"persistent={a.get('persistent')}"
        )
    if etype == METRIC_SUMMARY:
        return (
            f"[metric_summary] id={eid} window={win} service={service} "
            f"kpi={a.get('kpi')} value={format_number(a.get('value'))} "
            f"unit={a.get('unit')} z={format_z(a.get('z', 0.0))}"
        )
    if etype == SPAN_SLOWDOWN:
        return (
            f"[span_slowdown] id={eid} window={win} edge={a.get('caller')}>{a.get('callee')} "
            f"op={a.get('operation')} kpi={a.get('kpi')} "
            f"p99_us={format_number(a.get('p99_us'))} base_us={format_number(a.get('baseline_us'))} "
            f"z={format_z(a.get('z', 0.0))} n={a.get('n')}"
        )
    if etype == ERROR_CODE:
        return (
            f"[error_code] id={eid} window={win} edge={a.get('caller')}>{a.get('callee')} "
            f"op={a.get('operation')} errors={a.get('error_count')} calls={a.get('call_count')} "
            f"status={a.get('status_codes')}"
        )
    if etype == LOG_PATTERN:
        template = _short_template(a.get("template"))
        return (
            f"[log_pattern] id={eid} window={win} service={service} "
            f"template_id={a.get('template_id')} level={a.get('level')} "
            f"count={a.get('count')} base={format_number(a.get('baseline'))} "
            f"z={format_z(a.get('z', 0.0))} new_template={a.get('new_template')} "
            f'template="{template}"'
        )
    if etype == CALL_EDGE:
        return (
            f"[call_edge] id={eid} window={win} edge={a.get('caller')}>{a.get('callee')} "
            f"calls={a.get('count')} errors={a.get('error_count')}"
        )
    return f"[{etype}] id={eid} window={win} service={service}"


def _short_template(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").replace('"', "'").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Trimmed view used by the reasoning-unit pattern matcher.

    Carries structure (type/window/service + matchable attrs incl. z/kpi) but not
    the full rendered `line`, so composition stays cheap and text-free.
    """
    a = event.get("attrs", {})
    keep = {
        "kpi": a.get("kpi"),
        "z": a.get("z"),
        "direction": a.get("direction"),
        "persistent": a.get("persistent"),
        "caller": a.get("caller"),
        "callee": a.get("callee"),
        "operation": a.get("operation"),
        "level": a.get("level"),
        "template_id": a.get("template_id"),
        "new_template": a.get("new_template"),
        "error_count": a.get("error_count"),
    }
    return {
        "id": event.get("id"),
        "type": event.get("type"),
        "window": event.get("window"),
        "service": event.get("service"),
        "attrs": {k: v for k, v in keep.items() if v is not None},
    }
