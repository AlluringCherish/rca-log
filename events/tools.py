"""Event DB tool runtime for the Controller.

Maps the event-pattern vocabulary the Controller (and Analyst data requests)
speak to concrete resolvers over the EventStore. Tool calls are plain JSON
objects `{"name": ..., "args": {...}}`; each resolver returns a JSON-string
observation with human-readable `lines` (for the Analyst) and compact `events`
(for reasoning-unit pattern matching).
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from events import schema
from events.store import EventStore

# Event-pattern -> resolver. Single source of truth, shared by the Controller
# prompt and the deterministic data-request translator in the agent loop.
PATTERN_TO_TOOL = {
    "metric_anomaly": "get_anomaly_events",
    "metric_summary": "get_metric_events",
    "span_slowdown": "get_trace_events",
    "error_code": "get_trace_events",
    "log_pattern": "get_log_events",
    "call_edge": "get_topology",
}

ToolObservation = Dict[str, Any]


def _dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class EventToolRuntime:
    TOOL_NAMES = {
        "get_anomaly_events",
        "get_metric_events",
        "get_trace_events",
        "get_log_events",
        "get_topology",
    }

    ANOMALY_CAP = 40
    METRIC_CAP = 120
    TRACE_CAP = 40

    def __init__(self, event_dir: str) -> None:
        self.store = EventStore(event_dir)

    # -- dispatch ----------------------------------------------------------

    def execute_tool_calls(self, tool_calls: Any) -> List[ToolObservation]:
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, list):
            return [
                {
                    "tool_call": tool_calls,
                    "status": False,
                    "observation": "Error: tool_calls must be a list of tool call objects.",
                }
            ]

        observations: List[ToolObservation] = []
        for index, raw_call in enumerate(tool_calls, start=1):
            call = self._normalize_tool_call(raw_call)
            result: ToolObservation = {"tool_call": call, "status": False, "observation": ""}
            try:
                result["observation"] = self._dispatch(call)
                result["status"] = True
            except Exception as exc:
                result["observation"] = f"Tool call {index} failed: {exc}"
            observations.append(result)
        return observations

    def _dispatch(self, call: Dict[str, Any]) -> str:
        name = call.get("name", "")
        args = call.get("args", {})
        if name not in self.TOOL_NAMES:
            raise ValueError(
                f"Unknown tool `{name}`. Available tools: {', '.join(sorted(self.TOOL_NAMES))}."
            )
        if name == "get_anomaly_events":
            self._require_args(name, args, {"start_time", "end_time"}, {"services"})
            return self.get_anomaly_events(
                args.get("start_time"), args.get("end_time"), args.get("services", [])
            )
        if name == "get_metric_events":
            self._require_args(name, args, {"service", "start_time", "end_time"}, {"kpis"})
            return self.get_metric_events(
                args.get("service"), args.get("kpis", []),
                args.get("start_time"), args.get("end_time"),
            )
        if name == "get_trace_events":
            self._require_args(name, args, {"start_time", "end_time"}, {"services", "kinds"})
            return self.get_trace_events(
                args.get("start_time"), args.get("end_time"),
                args.get("services", []), args.get("kinds", ["span_slowdown", "error_code"]),
            )
        if name == "get_log_events":
            self._require_args(name, args, {"service", "start_time", "end_time"}, {"k"})
            return self.get_log_events(
                args.get("service"), args.get("start_time"), args.get("end_time"),
                int(args.get("k", 20)),
            )
        if name == "get_topology":
            self._require_args(name, args, {"start_time", "end_time"}, set())
            return self.get_topology(args.get("start_time"), args.get("end_time"))
        raise ValueError(f"Unhandled tool `{name}`.")

    # -- resolvers ---------------------------------------------------------

    def get_anomaly_events(self, start_time: Any, end_time: Any, services: Any) -> str:
        start, end = self._window(start_time, end_time)
        svc = self._service_list(services)
        matched = self.store.query(schema.METRIC_ANOMALY, start, end, svc, match="service")
        matched.sort(key=lambda e: -schema.event_magnitude(e))
        return self._payload("get_anomaly_events", {"start_time": start, "end_time": end,
                             "services": sorted(svc) if svc else []}, matched, self.ANOMALY_CAP)

    def get_metric_events(self, service: Any, kpis: Any, start_time: Any, end_time: Any) -> str:
        start, end = self._window(start_time, end_time)
        service_norm = schema.normalize_component(service)
        kpi_set = {self._normalize_kpi(s) for s in kpis} if kpis else None
        matched = [
            e for e in self.store.query(schema.METRIC_SUMMARY, start, end, [service_norm])
            if kpi_set is None or e.get("attrs", {}).get("kpi") in kpi_set
        ]
        matched.sort(key=lambda e: (int(e["window"][0]), str(e.get("attrs", {}).get("kpi"))))
        return self._payload("get_metric_events", {"service": service_norm,
                             "kpis": sorted(kpi_set) if kpi_set else [],
                             "start_time": start, "end_time": end}, matched, self.METRIC_CAP)

    def get_trace_events(self, start_time: Any, end_time: Any, services: Any, kinds: Any) -> str:
        start, end = self._window(start_time, end_time)
        svc = self._service_list(services)
        kind_set = [k for k in (kinds or []) if k in {schema.SPAN_SLOWDOWN, schema.ERROR_CODE}]
        if not kind_set:
            kind_set = [schema.SPAN_SLOWDOWN, schema.ERROR_CODE]
        matched: List[Dict[str, Any]] = []
        for kind in kind_set:
            matched.extend(self.store.query(kind, start, end, svc, match="edge"))
        matched.sort(key=lambda e: -schema.event_magnitude(e))
        return self._payload("get_trace_events", {"start_time": start, "end_time": end,
                             "services": sorted(svc) if svc else [], "kinds": kind_set},
                             matched, self.TRACE_CAP)

    def get_log_events(self, service: Any, start_time: Any, end_time: Any, k: int) -> str:
        start, end = self._window(start_time, end_time)
        service_norm = schema.normalize_component(service)
        matched = self.store.query(schema.LOG_PATTERN, start, end, [service_norm])
        matched.sort(key=lambda e: (
            not bool(e.get("attrs", {}).get("new_template")),
            -abs(float(e.get("attrs", {}).get("z", 0.0))),
        ))
        cap = max(1, min(int(k), 100))
        return self._payload("get_log_events", {"service": service_norm,
                             "start_time": start, "end_time": end, "k": cap}, matched, cap)

    def get_topology(self, start_time: Any, end_time: Any) -> str:
        start, end = self._window(start_time, end_time)
        edges = self.store.query(schema.CALL_EDGE, start, end)
        agg: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "errors": 0})
        for e in edges:
            a = e.get("attrs", {})
            key = (a.get("caller"), a.get("callee"))
            agg[key]["calls"] += int(a.get("count", 0))
            agg[key]["errors"] += int(a.get("error_count", 0))
        lines: List[str] = []
        compact: List[Dict[str, Any]] = []
        for (caller, callee), v in sorted(agg.items(), key=lambda kv: -kv[1]["calls"]):
            lines.append(f"[call_edge] edge={caller}>{callee} calls={v['calls']} errors={v['errors']}")
            compact.append({"caller": caller, "callee": callee,
                            "count": v["calls"], "error_count": v["errors"]})
        payload = {
            "telemetry": "events",
            "tool": "get_topology",
            "query": {"start_time": start, "end_time": end},
            "matched_count": len(agg),
            "returned_count": len(lines),
            "truncated": False,
            "services": self.store.topology().get("services", []),
            "lines": lines,
            "events": compact,
        }
        return _dumps(payload)

    # -- helpers -----------------------------------------------------------

    def _payload(self, tool: str, query: Dict[str, Any],
                 matched: List[Dict[str, Any]], cap: int) -> str:
        returned = matched[:cap]
        return _dumps({
            "telemetry": "events",
            "tool": tool,
            "query": query,
            "matched_count": len(matched),
            "returned_count": len(returned),
            "truncated": len(matched) > cap,
            "lines": [str(e.get("line", "")) for e in returned],
            "events": [schema.compact_event(e) for e in returned],
        })

    def _window(self, start_time: Any, end_time: Any) -> tuple:
        meta = self.store.meta()
        lo = int(meta.get("start_time", 0))
        hi = int(meta.get("end_time", 1440))
        try:
            start = int(float(start_time))
        except (TypeError, ValueError):
            start = lo
        try:
            end = int(float(end_time))
        except (TypeError, ValueError):
            end = hi
        start = max(lo, start)
        end = min(hi, end)
        if end <= start:
            end = min(hi, start + int(meta.get("window_size_seconds", 30)))
        return start, end

    @staticmethod
    def _service_list(services: Any) -> List[str]:
        if not services:
            return []
        if isinstance(services, str):
            services = [services]
        return [str(s) for s in services if s]

    @staticmethod
    def _normalize_kpi(kpi: Any) -> str:
        text = str(kpi).strip()
        return text.replace("latency-50", "latency.p50").replace("latency-90", "latency.p90")

    @staticmethod
    def _normalize_tool_call(raw_call: Any) -> Dict[str, Any]:
        if not isinstance(raw_call, dict):
            return {"name": "", "args": {}}
        name = raw_call.get("name") or raw_call.get("tool_name") or raw_call.get("tool") or ""
        args = raw_call.get("args", raw_call.get("parameters", {}))
        if not isinstance(args, dict):
            args = {}
        return {"name": str(name), "args": args}

    @staticmethod
    def _require_args(name: str, args: Dict[str, Any], required: set, optional: set) -> None:
        if not isinstance(args, dict):
            raise ValueError(f"`{name}` args must be an object.")
        missing = required - set(args)
        if missing:
            raise ValueError(f"`{name}` missing required args: {sorted(missing)}. "
                             f"Required: {sorted(required)}, optional: {sorted(optional)}.")
        unknown = set(args) - required - optional - {"reasoning"}
        if unknown:
            raise ValueError(f"`{name}` got unknown args: {sorted(unknown)}. "
                             f"Allowed: {sorted(required | optional)}.")
