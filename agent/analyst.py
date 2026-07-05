"""Analyst: interprets events against the composed reasoning template.

Cannot call tools. Owns the Stop decision and the final top-3. Requests more data
via `data_requests` in event-pattern vocabulary (not tool names).
"""

import json
from typing import Any, Dict, List, Optional

from common.llm import LLMClient, PREFIX_MARKER
from common.prompts import ANALYST_SYSTEM_PROMPT, FINAL_RANKING_INSTRUCTIONS, CANDIDATE_REASONS

DEFAULT_FINDINGS: Dict[str, Any] = {"metrics": [], "traces": [], "logs": [], "rankings": []}
VALID_PATTERNS = {
    "metric_anomaly", "metric_summary", "span_slowdown", "error_code", "log_pattern", "call_edge",
}
FORBIDDEN_KEYS = ("tool_calls", "tool_call", "completed", "next_action")


class Analyst:
    def __init__(self, llm: LLMClient, system_prompt: str = ANALYST_SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def analyze(
        self,
        case_context: Dict[str, Any],
        findings: Dict[str, Any],
        unit: Dict[str, Any],
        new_event_lines: List[str],
        step: int,
        max_steps: int,
        phase: str = "normal",
        key_events: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        is_final = phase == "final"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._render_user_prompt(
                case_context, findings, unit, new_event_lines, step, max_steps, key_events or [])},
        ]
        if is_final:
            messages.append({"role": "user", "content": FINAL_RANKING_INSTRUCTIONS})

        required = ("analysis", "stop", "final_ranking") if is_final else ("analysis", "stop")
        data = self.llm.json_chat(messages, required_keys=required, forbidden_keys=FORBIDDEN_KEYS)

        report = {
            "analysis": str(data.get("analysis", "")),
            "findings": self._normalize_findings(data.get("findings")),
            "stop": self._to_bool(data.get("stop")),
            "data_requests": self._normalize_data_requests(data.get("data_requests")),
            "final_ranking": self._normalize_final_ranking(data.get("final_ranking")),
        }
        if is_final:
            report["stop"] = True
        return report

    def prefix_messages(self, unit: Dict[str, Any]) -> List[Dict[str, str]]:
        """Messages whose cacheable prefix (system + this unit) should be pre-warmed.

        The dynamic part is a dummy; only the text before PREFIX_MARKER is used."""
        user = self._render_user_prompt(
            {}, DEFAULT_FINDINGS, unit, [], 1, 1, [])
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    # -- rendering ---------------------------------------------------------

    def _render_user_prompt(
        self,
        case_context: Dict[str, Any],
        findings: Dict[str, Any],
        unit: Dict[str, Any],
        new_event_lines: List[str],
        step: int,
        max_steps: int,
        key_events: List[str],
    ) -> str:
        # STATIC PREFIX (cacheable): the selected reasoning unit. Byte-identical for
        # the same unit, so its KV state is precomputed once and reused. PREFIX_MARKER
        # separates it from the dynamic part; the llm backend splits there for prefix
        # KV caching (and strips the marker when caching is off).
        unit_text = str(unit.get("unit_text", "")).strip()
        if unit_text:
            variables = unit.get("variables", [])
            prefix = "\n".join([
                "Reasoning unit (selected for this step's events; bind these "
                f"placeholders from the event lines below: {variables}):",
                unit_text,
            ])
        else:
            prefix = "(no reasoning unit for this step; analyze the events directly)"

        # DYNAMIC SUFFIX: step, case context, accumulated findings, accumulated key
        # events digest (cross-step/cross-modal), and this step's new event lines.
        suffix_parts = [
            f"Step {step} of {max_steps}.",
            "",
            "Case context:",
            self._render_context(case_context),
            "",
            "Accumulated findings (your cumulative state):",
            self._render_findings(findings),
            "",
            "Accumulated key events so far (all steps, strongest first):",
            *([f"- {line}" for line in key_events] if key_events else ["- (none)"]),
            "",
            "New event lines from this step's tool observations:",
            *([f"- {line}" for line in new_event_lines] if new_event_lines else ["- (none)"]),
        ]
        return prefix + PREFIX_MARKER + "\n".join(suffix_parts)

    @staticmethod
    def _render_context(case_context: Dict[str, Any]) -> str:
        keys = [
            "case_name", "telemetry_time_range", "window_size_seconds",
            "possible_root_cause_components", "possible_root_cause_reasons", "services",
        ]
        lines = []
        for key in keys:
            if key in case_context:
                lines.append(f"- {key}: {json.dumps(case_context[key], ensure_ascii=False)}")
        return "\n".join(lines)

    @staticmethod
    def _render_findings(findings: Dict[str, Any]) -> str:
        state = Analyst._normalize_findings(findings)
        lines = []
        for key in ("metrics", "traces", "logs"):
            items = state.get(key, [])
            lines.append(f"{key}: " + ("; ".join(items) if items else "(none)"))
        rankings = state.get("rankings", [])
        if rankings:
            lines.append("rankings: " + "; ".join(
                f"rank{r.get('rank')}={r.get('component')}_{r.get('reason')}" for r in rankings))
        else:
            lines.append("rankings: (none)")
        return "\n".join(lines)

    # -- normalization -----------------------------------------------------

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    @staticmethod
    def _normalize_findings(value: Any) -> Dict[str, Any]:
        state = {"metrics": [], "traces": [], "logs": [], "rankings": []}
        if not isinstance(value, dict):
            return state
        for key in ("metrics", "traces", "logs"):
            raw = value.get(key, [])
            state[key] = [str(item) for item in raw if item is not None][:5] if isinstance(raw, list) else []
        rankings = value.get("rankings", [])
        norm = []
        if isinstance(rankings, list):
            for item in rankings:
                if isinstance(item, dict):
                    norm.append({
                        "rank": item.get("rank"),
                        "component": item.get("component"),
                        "reason": Analyst._canonical_reason(item.get("reason")),
                    })
        state["rankings"] = norm[:3]
        return state

    @staticmethod
    def _normalize_data_requests(value: Any) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        if not isinstance(value, list):
            return requests
        for item in value:
            if not isinstance(item, dict):
                continue
            pattern = str(item.get("pattern", "")).strip()
            if pattern not in VALID_PATTERNS:
                continue
            request: Dict[str, Any] = {"pattern": pattern, "reason": str(item.get("reason", ""))}
            if item.get("service"):
                request["service"] = str(item["service"])
            kpi = item.get("kpi", item.get("signal"))
            if kpi:
                request["kpi"] = kpi
            window = item.get("window")
            if isinstance(window, list) and len(window) == 2:
                request["window"] = [int(window[0]), int(window[1])]
            requests.append(request)
        return requests[:3]

    @staticmethod
    def _normalize_final_ranking(value: Any) -> List[Dict[str, Any]]:
        ranking: List[Dict[str, Any]] = []
        if not isinstance(value, list):
            return ranking
        for item in value:
            if not isinstance(item, dict):
                continue
            component = item.get("component")
            reason = Analyst._canonical_reason(item.get("reason"))
            if not component or not reason:
                continue
            ranking.append({
                "component": str(component),
                "reason": reason,
                "justification": str(item.get("justification", ""))[:200],
            })
            if len(ranking) >= 3:
                break
        return ranking

    @staticmethod
    def _canonical_reason(value: Any) -> Any:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        for reason in CANDIDATE_REASONS:
            if normalized == reason or normalized.startswith(f"{reason}-") or normalized.startswith(f"{reason}_") or normalized.startswith(f"{reason}."):
                return reason
        return normalized


def merge_findings(previous: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    prev = Analyst._normalize_findings(previous)
    nxt = Analyst._normalize_findings(new)
    merged = {"metrics": [], "traces": [], "logs": [], "rankings": []}
    for key in ("metrics", "traces", "logs"):
        seen = set()
        combined = []
        for item in prev.get(key, []) + nxt.get(key, []):
            if item not in seen:
                seen.add(item)
                combined.append(item)
        merged[key] = combined[:6]
    merged["rankings"] = nxt.get("rankings") or prev.get("rankings", [])
    return merged
