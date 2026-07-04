"""In-memory event store for one preprocessed case.

Loads events.jsonl (a few thousand records) into plain Python dicts with a
by-type index. No database: filtering is cheap linear scans over pre-sorted
lists. This is the read side that the Controller's tools query.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from events import schema


def window_overlaps(event: Dict[str, Any], start: int, end: int) -> bool:
    ws, we = int(event["window"][0]), int(event["window"][1])
    return ws < end and we > start


class EventStore:
    def __init__(self, event_dir: str) -> None:
        self.event_dir = event_dir
        self._events: Optional[List[Dict[str, Any]]] = None
        self._by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._topology: Optional[Dict[str, Any]] = None
        self._meta: Optional[Dict[str, Any]] = None

    def _load(self) -> None:
        if self._events is not None:
            return
        path = os.path.join(self.event_dir, "events.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Events file does not exist: {path}")
        events: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            by_type.setdefault(event["type"], []).append(event)
        for typed in by_type.values():
            typed.sort(key=lambda e: int(e["window"][0]))
        self._events = events
        self._by_type = by_type

    @property
    def events(self) -> List[Dict[str, Any]]:
        self._load()
        assert self._events is not None
        return self._events

    def by_type(self, event_type: str) -> List[Dict[str, Any]]:
        self._load()
        assert self._by_type is not None
        return self._by_type.get(event_type, [])

    def meta(self) -> Dict[str, Any]:
        if self._meta is None:
            path = os.path.join(self.event_dir, "meta.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
            else:
                self._meta = {"start_time": 0, "end_time": 1440, "window_size_seconds": 30}
        return self._meta

    def topology(self) -> Dict[str, Any]:
        """Shared Online Boutique topology, one level above the events dir.

        Layout: <events_root>/../topology.json (i.e. data/re2-ob/topology.json,
        with events at data/re2-ob/events/problem_XXXXXX). Falls back to a
        per-case file for backward compatibility, then to empty.
        """
        if self._topology is None:
            shared = os.path.join(os.path.dirname(os.path.dirname(self.event_dir)), "topology.json")
            per_case = os.path.join(self.event_dir, "topology.json")
            for path in (shared, per_case):
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        self._topology = json.load(f)
                    break
            else:
                self._topology = {"services": [], "edges": []}
        return self._topology

    def query(
        self,
        event_type: str,
        start: int,
        end: int,
        services: Optional[Iterable[str]] = None,
        match: str = "service",
    ) -> List[Dict[str, Any]]:
        """Filter events of a type by time window and (normalized) service.

        match="service": event.service in services.
        match="edge":    caller or callee in services.
        """
        wanted = {schema.normalize_component(s) for s in services} if services else None
        result: List[Dict[str, Any]] = []
        for event in self.by_type(event_type):
            if not window_overlaps(event, start, end):
                continue
            if wanted:
                if match == "edge":
                    a = event.get("attrs", {})
                    if schema.normalize_component(a.get("caller")) not in wanted and \
                       schema.normalize_component(a.get("callee")) not in wanted:
                        continue
                else:
                    if schema.normalize_component(event.get("service")) not in wanted:
                        continue
            result.append(event)
        return result
