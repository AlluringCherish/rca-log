"""Reasoning-unit DB: storage and single-unit selection.

A reasoning unit is a distilled, self-contained diagnostic procedure (SOP): an
event-pattern trigger (IF) plus a step-by-step reasoning text with <placeholder>
variables and a worked example (THEN). Given the events fetched so far, UnitDB
selects the ONE most relevant unit for the current step; the Analyst binds the
placeholders from the actual event lines.

Because the unit set is a small fixed DB (<=10), each unit's rendered prompt
prefix can be precomputed once as a KV-state and reused at inference (see
common/llm.py warm_prefixes).
"""

import json
from typing import Any, Dict, Iterable, List, Optional

from events.schema import CANDIDATE_COMPONENTS, METRIC_ANOMALY, event_magnitude, normalize_component


class ReasoningUnit:
    def __init__(self, spec: Dict[str, Any]) -> None:
        self.id = str(spec["id"])
        self.name = str(spec.get("name", self.id))
        self.priority = int(spec.get("priority", 50))
        self.trigger = spec.get("trigger", {}) or {}
        self.variables = list(spec.get("variables", []))
        self.reasoning = str(spec.get("reasoning", "")).strip()
        self.example = str(spec.get("example", "")).strip()

    @property
    def is_final(self) -> bool:
        return self.trigger.get("phase") == "final"

    @property
    def is_always(self) -> bool:
        return bool(self.trigger.get("always"))


def load_units(path: str) -> List[ReasoningUnit]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [ReasoningUnit(spec) for spec in payload.get("units", [])]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _kpi_matches(want: str, actual: Optional[str]) -> bool:
    if actual is None:
        return False
    want = str(want)
    actual = str(actual)
    # "latency" matches latency.p50/p90 and latency.p99; exact otherwise.
    if want in {"latency", "latency.*"}:
        return actual.startswith("latency")
    return actual == want


def _where_match(where: Dict[str, Any], event: Dict[str, Any]) -> bool:
    attrs = event.get("attrs", {})
    if "kpi" in where:
        wanted = _as_list(where["kpi"])
        if not any(_kpi_matches(w, attrs.get("kpi")) for w in wanted):
            return False
    if "service" in where:
        if event.get("service") not in _as_list(where["service"]):
            return False
    if "direction" in where and attrs.get("direction") != where["direction"]:
        return False
    if "level" in where and attrs.get("level") not in _as_list(where["level"]):
        return False
    if "new_template" in where and bool(attrs.get("new_template")) != bool(where["new_template"]):
        return False
    if "persistent" in where and bool(attrs.get("persistent")) != bool(where["persistent"]):
        return False
    if "min_severity" in where and event_magnitude(event) < float(where["min_severity"]):
        return False
    return True


class UnitDB:
    """Fixed DB of reasoning units + single-unit selector."""

    # A directly-measured resource anomaly at/above this |z| preempts the latency
    # (symptom-prone) unit in selection. Above trigger min_severity (~3) with margin.
    RESOURCE_MIN = 8.0

    def __init__(self, units: List[ReasoningUnit], candidates: Optional[Iterable[str]] = None) -> None:
        self.units = units
        self.candidates = {normalize_component(c) for c in (candidates or CANDIDATE_COMPONENTS)}

    # -- trigger evaluation --------------------------------------------------

    def _is_candidate_event(self, event: Dict[str, Any]) -> bool:
        attrs = event.get("attrs", {})
        for key in ("service", "callee", "caller"):
            value = attrs.get(key) if key != "service" else event.get("service")
            if value is not None and normalize_component(value) in self.candidates:
                return True
        return False

    def _matching_events(self, unit: ReasoningUnit, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trigger = unit.trigger
        event_types = set(_as_list(trigger.get("event_type")))
        where = trigger.get("where", {}) or {}
        candidate_only = bool(trigger.get("candidate_service"))
        out = []
        for event in events:
            if event_types and event.get("type") not in event_types:
                continue
            if not _where_match(where, event):
                continue
            if candidate_only and not self._is_candidate_event(event):
                continue
            out.append(event)
        return out

    def _trigger_fires(self, unit: ReasoningUnit, events: List[Dict[str, Any]], phase: str) -> bool:
        if unit.is_final:
            return phase == "final"
        if unit.is_always:
            return True
        min_count = int(unit.trigger.get("min_count", 1))
        return len(self._matching_events(unit, events)) >= min_count

    _RESOURCE_KPIS = {"cpu", "mem", "diskio", "socket"}

    @staticmethod
    def _unit_kpis(unit: ReasoningUnit) -> set:
        return set(_as_list((unit.trigger.get("where") or {}).get("kpi", [])))

    def _is_resource_unit(self, unit: ReasoningUnit) -> bool:
        kpis = self._unit_kpis(unit)
        return bool(kpis) and kpis <= self._RESOURCE_KPIS

    def _is_latency_unit(self, unit: ReasoningUnit) -> bool:
        return "latency" in self._unit_kpis(unit)

    def _latency_is_symptom(self, events: List[Dict[str, Any]]) -> bool:
        """True iff the candidate with the strongest LOCAL latency metric ALSO has a
        real resource anomaly (cpu/mem/diskio/socket) on the SAME service — i.e. that
        latency is a downstream symptom of resource saturation, not a root cause. This
        is the latency unit's own step L2 applied at the selection layer. Genuine
        latency faults (delay/loss) show latency with NO co-located resource anomaly,
        so this returns False and the latency unit is kept."""
        res_z: Dict[str, float] = {}
        lat_z: Dict[str, float] = {}
        for e in events:
            if e.get("type") != METRIC_ANOMALY:
                continue
            svc = normalize_component(e.get("service"))
            if svc not in self.candidates:
                continue
            kpi = str(e.get("attrs", {}).get("kpi", ""))
            z = event_magnitude(e)
            if kpi.startswith("latency"):
                lat_z[svc] = max(lat_z.get(svc, 0.0), z)
            elif kpi in self._RESOURCE_KPIS:
                res_z[svc] = max(res_z.get(svc, 0.0), z)
        if not lat_z:
            return False
        top_lat_svc = max(lat_z, key=lambda s: lat_z[s])
        return res_z.get(top_lat_svc, 0.0) >= self.RESOURCE_MIN

    def _relevance(self, unit: ReasoningUnit, events: List[Dict[str, Any]]) -> float:
        """Strongest matching *metric_anomaly* magnitude (|z|); always-on fallback = 0.

        Trace events (span_slowdown/error_code) still FIRE a unit's trigger but are
        EXCLUDED from the relevance score: their raw-duration p99 z saturates the
        +/-1000 clip, so scoring them would make unit_latency win every fault the
        moment traces are fetched — contradicting the units' own doctrine (localize by
        the component-LOCAL metric; a raw-duration edge is propagation). Scoring only
        by the local metric anomaly aligns selection with that doctrine."""
        if unit.is_always:
            return 0.0
        metric_matched = [e for e in self._matching_events(unit, events)
                          if e.get("type") == METRIC_ANOMALY]
        return max((event_magnitude(e) for e in metric_matched), default=0.0)

    # -- selection -------------------------------------------------------------

    def select_unit(self, events: Iterable[Dict[str, Any]], phase: str = "normal") -> Dict[str, Any]:
        """Pick the single most relevant unit for the current step's events."""
        events = list(events or [])
        matched = [u for u in self.units if self._trigger_fires(u, events, phase)]

        if phase == "final":
            chosen = [u for u in matched if u.is_final][:1]
        else:
            scored = [
                (self._relevance(u, events), u.priority, u.id, u)
                for u in matched if not u.is_final
            ]
            # highest relevance; tie -> more specific (higher priority number) -> id
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            # Cause-over-symptom: if the top-latency candidate also has a co-located
            # resource anomaly, its latency is a symptom -> demote the latency unit so a
            # resource/injected_io unit is chosen. Genuine delay/loss faults have no such
            # co-located resource anomaly and keep the latency unit.
            if self._latency_is_symptom(events):
                demoted = [t for t in scored if not self._is_latency_unit(t[3])]
                if demoted:
                    scored = demoted
            chosen = [scored[0][3]] if scored else []

        return {
            "unit_ids": [u.id for u in chosen],
            "variables": _dedupe([v for u in chosen for v in u.variables]),
            "unit_text": self.render(chosen[0]) if chosen else "",
        }

    def all_unit_prompts(self) -> List[Dict[str, Any]]:
        """One selection-shaped dict per unit — used to precompute prefix KV states."""
        return [
            {"unit_ids": [u.id], "variables": list(u.variables), "unit_text": self.render(u)}
            for u in self.units
        ]

    # -- rendering ---------------------------------------------------------------

    @staticmethod
    def render(unit: ReasoningUnit) -> str:
        parts = [f"[{unit.name}]", unit.reasoning]
        if unit.example:
            parts.append("")
            parts.append(unit.example)
        return "\n".join(parts)


def _dedupe(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
