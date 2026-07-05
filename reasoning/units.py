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

from events.schema import CANDIDATE_COMPONENTS, event_magnitude, normalize_component


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

    def _relevance(self, unit: ReasoningUnit, events: List[Dict[str, Any]]) -> float:
        """Strongest matching-event magnitude (|z|); the always-on fallback scores 0."""
        if unit.is_always:
            return 0.0
        return max((event_magnitude(e) for e in self._matching_events(unit, events)), default=0.0)

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
