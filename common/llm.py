"""LLM client for the RCA prototype.

Simplified from the original project: an OpenAI-compatible backend (OpenAI or
OpenRouter, env-configured) and a local Qwen3 backend (HF transformers). The
KV-graft reasoning-memory feature has been removed. `json_chat` extracts and
repairs a single JSON object with required/forbidden-key retries.

Default backend is `local` (Qwen3 at /data/models/Qwen3-8B).
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Protocol


DEFAULT_LOCAL_MODEL = "/data/models/Qwen3-8B"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMConfigError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


class ChatBackend(Protocol):
    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        ...


def require_llm_env(
    model_override: Optional[str] = None,
    backend: str = "openai",
    local_model: str = DEFAULT_LOCAL_MODEL,
) -> None:
    if backend == "local":
        if not os.path.isdir(local_model):
            raise LLMConfigError(f"Local model directory does not exist: {local_model}")
        return

    missing = []
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        missing.append("OPENAI_API_KEY or OPENROUTER_API_KEY")
    if not (
        model_override
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("OPENROUTER_MODEL")
    ):
        missing.append("OPENAI_MODEL or OPENROUTER_MODEL or --model")
    if missing:
        raise LLMConfigError(
            "Missing LLM configuration: "
            + ", ".join(missing)
            + ". Optional: OPENAI_BASE_URL for OpenAI-compatible endpoints."
        )


def _strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, maxsplit=1, flags=re.IGNORECASE)[-1]
    cleaned = re.sub(r"^\s*<think>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _has_unclosed_thinking(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered and "</think>" not in lowered


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = _strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    balanced = _first_balanced_json_object(cleaned)
    if balanced:
        cleaned = balanced

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        repaired = _repair_truncated_json(cleaned)
        balanced_repaired = _first_balanced_json_object(repaired)
        if balanced_repaired:
            repaired = balanced_repaired
        if repaired != cleaned:
            try:
                value = json.loads(repaired)
            except json.JSONDecodeError:
                raise LLMResponseError(
                    f"LLM did not return valid JSON: {exc}\nRaw response:\n{text}"
                ) from exc
        else:
            raise LLMResponseError(
                f"LLM did not return valid JSON: {exc}\nRaw response:\n{text}"
            ) from exc

    if not isinstance(value, dict):
        raise LLMResponseError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def _first_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return ""


def _repair_truncated_json(text: str) -> str:
    if '"args"' in text:
        text = text.replace('},{"name"', '}},{"name"')
        text = text.replace('},{"tool_name"', '}},{"tool_name"')
        text = text.replace('},{"tool"', '}},{"tool"')
    text = re.sub(r'\]\s*,\s*\[\s*"((?:metrics|traces|logs|rankings)"\s*:)', r'],"\1', text)
    text = re.sub(r'"\]\s*,\s*\["', '","', text)
    stack: List[str] = []
    in_string = False
    escape = False
    output: List[str] = []
    closers = {"{": "}", "[": "]"}
    expected_openers = {"}": "{", "]": "["}
    for char in text:
        output.append(char)
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in expected_openers:
            expected = expected_openers[char]
            if expected in stack:
                inserted = []
                while stack and stack[-1] != expected:
                    inserted.append(closers[stack.pop()])
                if inserted:
                    output[-1:-1] = inserted
                if stack and stack[-1] == expected:
                    stack.pop()
    if in_string:
        output.append('"')
    while stack:
        output.append(closers[stack.pop()])
    return "".join(output)


class OpenAIChatBackend:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 0.9,
    ) -> None:
        require_llm_env(model, backend="openai")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if model:
            self.model = model
        elif openrouter_key and os.environ.get("OPENROUTER_MODEL"):
            self.model = os.environ["OPENROUTER_MODEL"]
        else:
            self.model = os.environ.get("OPENAI_MODEL") or os.environ["OPENROUTER_MODEL"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.enable_thinking = False

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        from openai import OpenAI

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        client_args: Dict[str, Any] = {"api_key": openrouter_key or os.environ["OPENAI_API_KEY"]}
        if openrouter_key:
            client_args["base_url"] = (
                os.environ.get("OPENROUTER_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or DEFAULT_OPENROUTER_BASE_URL
            )
        elif os.environ.get("OPENAI_BASE_URL"):
            client_args["base_url"] = os.environ["OPENAI_BASE_URL"]
        client = OpenAI(**client_args)

        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(**request)
                return response.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - endpoint-specific failures
                last_error = exc
                if "429" in str(exc) or "rate" in str(exc).lower():
                    time.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError(f"LLM request failed after retries: {last_error}")


class LocalQwenChatBackend:
    def __init__(
        self,
        model_path: str = DEFAULT_LOCAL_MODEL,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        cpu: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        require_llm_env(backend="local", local_model=model_path)
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.cpu = cpu
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.cpu and not torch.cuda.is_available():
            raise LLMConfigError(
                "CUDA is not available for local model execution. "
                "Fix the GPU/CUDA environment or pass --cpu explicitly."
            )

        if self.cpu:
            device_map = None
            dtype = torch.float32
        else:
            device_map = {"": f"cuda:{torch.cuda.current_device()}"}
            dtype = torch.bfloat16

        print(f"Loading local model from {self.model_path}", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def prepare(self) -> None:
        self._load()

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        self._load()

        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if not self.cpu:
            inputs = inputs.to(self._model.device)

        temp = self.temperature if temperature is None else temperature
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temp > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temp > 0:
            generation_kwargs["temperature"] = temp
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        backend: str = "local",
        local_model: str = DEFAULT_LOCAL_MODEL,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        cpu: bool = False,
        enable_thinking: bool = False,
    ) -> None:
        self.backend_name = backend
        if backend == "local":
            self.backend: ChatBackend = LocalQwenChatBackend(
                model_path=local_model,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                cpu=cpu,
                enable_thinking=enable_thinking,
            )
        elif backend == "openai":
            self.backend = OpenAIChatBackend(
                model=model,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
            )
        else:
            raise LLMConfigError(f"Unsupported LLM backend: {backend}. Choose local or openai.")

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        return self.backend.chat(messages, temperature=temperature)

    def prepare(self) -> None:
        prepare = getattr(self.backend, "prepare", None)
        if callable(prepare):
            prepare()

    def json_chat(
        self,
        messages: List[Dict[str, str]],
        required_keys: Iterable[str],
        forbidden_keys: Iterable[str] = (),
    ) -> Dict[str, Any]:
        correction_messages = list(messages)
        if getattr(self.backend, "enable_thinking", False):
            correction_messages.append(
                {
                    "role": "user",
                    "content": (
                        "For this JSON response, keep any thinking very brief, close `</think>`, "
                        "and then return exactly one valid JSON object. Do not spend the full token "
                        "budget in thinking."
                    ),
                }
            )
        last_error = ""
        for _ in range(3):
            raw = self.chat(correction_messages)
            try:
                data = extract_json_object(raw)
            except LLMResponseError as exc:
                last_error = str(exc)
                if _has_unclosed_thinking(raw):
                    correction_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous response used the token budget inside `<think>` and "
                                "did not return JSON. In this retry, do not continue that reasoning. "
                                "Return the required JSON object immediately, with no prose."
                            ),
                        }
                    )
                else:
                    correction_messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": "Return exactly one valid JSON object only.",
                            },
                        ]
                    )
                continue

            missing = [key for key in required_keys if key not in data]
            forbidden = [key for key in forbidden_keys if key in data]
            if not missing and not forbidden:
                return data
            last_error = f"Missing keys: {missing}. Forbidden keys present: {forbidden}."
            correction_messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Return one valid JSON object only. "
                            f"Missing keys: {missing}. Forbidden keys present: {forbidden}."
                        ),
                    },
                ]
            )
        raise LLMResponseError(
            f"LLM response did not satisfy the required JSON schema. {last_error}"
        )
