"""Reasoning units and the template composer.

A reasoning unit is a distilled if-then step: an event-pattern trigger (IF) and a
CoT fragment with <placeholder> variables (THEN). The composer matches unit
triggers against the events fetched so far and concatenates the matched units
into an "analysis procedure" (reasoning template) for the current case. The
Analyst binds the placeholders from the actual event lines it receives.
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


class TemplateComposer:
    MAX_NON_FINAL_UNITS = 7

    def __init__(self, units: List[ReasoningUnit], candidates: Optional[Iterable[str]] = None) -> None:
        self.units = units
        self.candidates = {normalize_component(c) for c in (candidates or CANDIDATE_COMPONENTS)}

    def _is_candidate_event(self, event: Dict[str, Any]) -> bool:
        attrs = event.get("attrs", {})
        for key in ("service", "callee", "caller"):
            value = attrs.get(key) if key != "service" else event.get("service")
            if value is not None and normalize_component(value) in self.candidates:
                return True
        return False

    def _trigger_fires(self, unit: ReasoningUnit, events: List[Dict[str, Any]], phase: str) -> bool:
        trigger = unit.trigger
        if unit.is_final:
            return phase == "final"
        if unit.is_always:
            return True
        event_types = set(_as_list(trigger.get("event_type")))
        where = trigger.get("where", {}) or {}
        min_count = int(trigger.get("min_count", 1))
        # A candidate-scoped unit only fires on anomalies of a candidate root-cause
        # service, so noise on supporting services (redis, cart, ...) does not
        # pull in irrelevant localization units.
        candidate_only = bool(trigger.get("candidate_service"))
        count = 0
        for event in events:
            if event_types and event.get("type") not in event_types:
                continue
            if not _where_match(where, event):
                continue
            if candidate_only and not self._is_candidate_event(event):
                continue
            count += 1
            if count >= min_count:
                return True
        return count >= min_count

    def compose(self, events: Iterable[Dict[str, Any]], phase: str = "normal") -> Dict[str, Any]:
        events = list(events or [])
        matched = [u for u in self.units if self._trigger_fires(u, events, phase)]
        matched.sort(key=lambda u: (u.priority, u.id))

        non_final = [u for u in matched if not u.is_final][: self.MAX_NON_FINAL_UNITS]
        chosen: List[ReasoningUnit] = list(non_final)
        if phase == "final":
            chosen += [u for u in matched if u.is_final]

        return {
            "unit_ids": [u.id for u in chosen],
            "variables": _dedupe([v for u in chosen for v in u.variables]),
            "template_text": self._render(chosen),
        }

    @staticmethod
    def _render(units: List[ReasoningUnit]) -> str:
        if not units:
            return ""
        header = (
            f"Analysis procedure (composed from {len(units)} reasoning unit(s) matched to this "
            f"case's events). Follow the steps in order; bind each <placeholder> from the event "
            f"lines you were given.\n"
        )
        blocks = [f"[{i}] {u.name}\n{u.reasoning}" for i, u in enumerate(units, start=1)]
        return header + "\n\n" + "\n\n".join(blocks)


def _dedupe(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
