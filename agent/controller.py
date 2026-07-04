"""Controller: plans which event data to fetch. It never analyzes or terminates."""

import json
from typing import Any, Dict, List

from common.llm import LLMClient
from common.prompts import CONTROLLER_SYSTEM_PROMPT


class Controller:
    def __init__(self, llm: LLMClient, system_prompt: str = CONTROLLER_SYSTEM_PROMPT) -> None:
        self.llm = llm
        self.system_prompt = system_prompt

    def decide(
        self,
        case_context: Dict[str, Any],
        analyst_report: Any,
        data_requests: List[Dict[str, Any]],
        action_history: List[Dict[str, Any]],
        step: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        payload = {
            "case_context": case_context,
            "analyst_report": self._compact_report(analyst_report),
            "pending_data_requests": data_requests or [],
            "action_history": action_history,
            "step": step,
            "max_steps": max_steps,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
        data = self.llm.json_chat(
            messages,
            required_keys=("tool_calls",),
            forbidden_keys=("completed", "stop", "final_ranking", "analysis"),
        )
        return {"tool_calls": self._normalize_tool_calls(data)}

    @staticmethod
    def _compact_report(report: Any) -> Any:
        if not isinstance(report, dict):
            return None
        return {
            "analysis": report.get("analysis"),
            "rankings": report.get("findings", {}).get("rankings", []),
            "stop": report.get("stop"),
        }

    @staticmethod
    def _normalize_tool_calls(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_calls = data.get("tool_calls")
        if raw_calls is None and isinstance(data.get("tool_call"), dict):
            raw_calls = [data["tool_call"]]
        if isinstance(raw_calls, dict):
            raw_calls = [raw_calls]
        if not isinstance(raw_calls, list):
            return []

        tool_calls: List[Dict[str, Any]] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            name = raw_call.get("name") or raw_call.get("tool_name") or raw_call.get("tool")
            args = raw_call.get("args", raw_call.get("parameters", {}))
            if not name:
                nested = [
                    (key, value)
                    for key, value in raw_call.items()
                    if key not in {"args", "parameters", "reasoning"} and isinstance(value, dict)
                ]
                if len(nested) == 1:
                    name, args = nested[0]
            if not name or not isinstance(args, dict):
                continue
            reasoning = raw_call.get("reasoning") or args.pop("reasoning", "")
            tool_calls.append({"name": str(name), "args": args, "reasoning": str(reasoning)})
        return tool_calls
