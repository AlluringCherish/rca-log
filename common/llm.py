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

# Splits an Analyst prompt into a static cacheable PREFIX (system + composed
# reasoning template) and a dynamic SUFFIX (case context, findings, events).
# The two sides are tokenized separately and concatenated at the TOKEN level so
# the prefix KV cache aligns exactly (validated: exact-match vs full recompute).
PREFIX_MARKER = "\n<<<KV_PREFIX_END>>>\n"


class _TimingStreamer:
    """Duck-typed generation streamer that records TTFT and decode span.

    `generate()` calls put() once with the prompt ids, then once per new token.
    We skip the first (prompt) put; the next put marks the first generated token,
    so prefill_s (TTFT) = first_token_time - start, decode_s = last - first.
    """

    def __init__(self) -> None:
        self.start: Optional[float] = None
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self._prompt_seen = False

    def put(self, value: Any) -> None:
        now = time.perf_counter()
        if not self._prompt_seen:
            self._prompt_seen = True  # first put = the prompt token ids
            return
        if self.first is None:
            self.first = now
        self.last = now

    def end(self) -> None:
        pass


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
        kv_prefix_cache: bool = False,
        kv_offload_cpu: bool = False,
    ) -> None:
        require_llm_env(backend="local", local_model=model_path)
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.cpu = cpu
        self.enable_thinking = enable_thinking
        self.kv_prefix_cache = kv_prefix_cache
        self.kv_offload_cpu = kv_offload_cpu
        # Precomputed prefix KV registry: prefix_str -> {"kv": [(keys,values),...], "n": int}.
        # The unit set is a small fixed DB (<=10), so all prefixes are built once via
        # warm_prefixes() at startup; no runtime eviction. Raw (keys,values) tensors are
        # stored and cloned into a DynamicCache per call (DynamicCache is only the
        # generate() wrapper, not the cache strategy).
        self._prefix_kv: Dict[str, Dict[str, Any]] = {}
        self._call_metrics: List[Dict[str, Any]] = []  # drained by LLMClient.pop_call_metrics
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

    def _sync(self) -> None:
        import torch

        if not self.cpu and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _build_prefix_kv(self, prefix_str: str) -> Dict[str, Any]:
        """Forward `prefix_str` once and store its raw (keys, values) per layer."""
        import torch

        tok = self._tokenizer
        prefix_ids = tok(prefix_str, return_tensors="pt").input_ids.to(self._model.device)
        self._sync()
        t = time.perf_counter()
        with torch.inference_mode():
            # Use the BASE model (no lm_head) so we don't materialize logits over all
            # prefix positions (vocab*len*2 bytes = several GB for a 20k prefix -> OOM).
            base = getattr(self._model, "model", self._model)
            cache = base(
                prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                use_cache=True,
            ).past_key_values
        self._sync()
        # Store raw tensors (Qwen3 standard GQA; new transformers .layers API).
        kv = [(layer.keys, layer.values) for layer in cache.layers]
        if self.kv_offload_cpu:
            # Move to host RAM so many large (20k-token) prefixes fit; copied back
            # to GPU per call in _prefix_cache_wrap (that copy is timed into TTFT).
            kv = [(k.cpu().contiguous(), v.cpu().contiguous()) for k, v in kv]
            self._sync()
        entry = {"kv": kv, "n": int(prefix_ids.shape[1]), "build_s": time.perf_counter() - t}
        self._prefix_kv[prefix_str] = entry
        return entry

    def prefix_str_of(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """The cacheable prefix (text before PREFIX_MARKER) of a rendered chat, or None."""
        self._load()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        if PREFIX_MARKER in prompt:
            return prompt.split(PREFIX_MARKER, 1)[0]
        return None

    def warm_prefixes(self, prefix_strings: Iterable[str]) -> Dict[str, Any]:
        """Precompute and store the KV state for each distinct prefix (startup)."""
        self._load()
        built, total_s = 0, 0.0
        for prefix_str in prefix_strings:
            if prefix_str in self._prefix_kv:
                continue
            entry = self._build_prefix_kv(prefix_str)
            built += 1
            total_s += entry["build_s"]
        return {"built": built, "total_s": total_s, "registry_size": len(self._prefix_kv)}

    def _prefix_cache_wrap(self, prefix_str: str, metrics: Dict[str, Any]) -> Any:
        """Return a DynamicCache wrapping the stored prefix KV (cloned so generate
        does not mutate the registry). Builds on the fly if not pre-warmed (fallback)."""
        from transformers import DynamicCache

        entry = self._prefix_kv.get(prefix_str)
        if entry is None:  # safety net; normally all prefixes are pre-warmed
            print("[kv] prefix not pre-warmed; building on the fly", file=sys.stderr)
            entry = self._build_prefix_kv(prefix_str)
            metrics["cache_hit"] = False
            metrics["cache_build_s"] = entry["build_s"]
        else:
            metrics["cache_hit"] = True
        dev = self._model.device
        wrapped = DynamicCache()
        for i, (keys, values) in enumerate(entry["kv"]):
            # copy=True makes an independent GPU tensor whether the store is on CPU
            # (offload: real H2D copy) or GPU (in-place clone) — generate mutates it.
            wrapped.update(keys.to(dev, copy=True), values.to(dev, copy=True), i)
        return wrapped

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        self._load()

        import torch

        tok = self._tokenizer
        model = self._model
        assert tok is not None and model is not None

        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        temp = self.temperature if temperature is None else temperature
        metrics: Dict[str, Any] = {
            "cache_hit": False, "cache_build_s": 0.0, "prefix_tokens": 0,
            "suffix_tokens": 0, "gen_tokens": 0, "prefill_s": 0.0,
            "decode_s": 0.0, "total_s": 0.0,
        }

        # When the marker is present we ALWAYS tokenize as prefix_ids ++ suffix_ids
        # (token-level concat), whether or not the cache is on. This guarantees the
        # cache-off baseline and the cache-on run process byte-identical token
        # sequences, so the only difference is timing (accuracy provably unchanged).
        if PREFIX_MARKER in prompt:
            prefix_str, suffix_str = prompt.split(PREFIX_MARKER, 1)
            prefix_ids = tok(prefix_str, return_tensors="pt").input_ids.to(model.device)
            suffix_ids = tok(suffix_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
            metrics["prefix_tokens"] = int(prefix_ids.shape[1])
            metrics["suffix_tokens"] = int(suffix_ids.shape[1])
        else:
            full_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            metrics["suffix_tokens"] = int(full_ids.shape[1])
        cache_prefix_str = prefix_str if (PREFIX_MARKER in prompt and self.kv_prefix_cache) else None

        attn = torch.ones_like(full_ids)
        gen_kwargs: Dict[str, Any] = {
            "attention_mask": attn,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": temp > 0,
            "pad_token_id": tok.eos_token_id,
        }
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = self.top_p

        streamer = _TimingStreamer()
        gen_kwargs["streamer"] = streamer
        self._sync()
        streamer.start = time.perf_counter()
        # Wrap the cached prefix KV (incl. any CPU->GPU copy) AFTER the timing origin
        # so the copy cost is honestly counted inside prefill_s (TTFT).
        if cache_prefix_str is not None:
            gen_kwargs["past_key_values"] = self._prefix_cache_wrap(cache_prefix_str, metrics)
        with torch.inference_mode():
            output_ids = model.generate(full_ids, **gen_kwargs)
        self._sync()
        end = time.perf_counter()

        generated = output_ids[0][full_ids.shape[1]:]
        metrics["gen_tokens"] = int(generated.shape[0])
        metrics["total_s"] = end - streamer.start
        if streamer.first is not None:
            metrics["prefill_s"] = streamer.first - streamer.start  # TTFT
            metrics["decode_s"] = (streamer.last - streamer.first) if streamer.last else 0.0
        else:
            metrics["prefill_s"] = metrics["total_s"]
        self._call_metrics.append(metrics)

        return tok.decode(generated, skip_special_tokens=True).strip()


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
        kv_prefix_cache: bool = False,
        kv_offload_cpu: bool = False,
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
                kv_prefix_cache=kv_prefix_cache,
                kv_offload_cpu=kv_offload_cpu,
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

    def warm_prefixes(self, messages_list: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Precompute prefix KV for each message list (local backend + cache on)."""
        backend = self.backend
        if not getattr(backend, "kv_prefix_cache", False):
            return {"built": 0}
        prefix_of = getattr(backend, "prefix_str_of", None)
        warm = getattr(backend, "warm_prefixes", None)
        if not (callable(prefix_of) and callable(warm)):
            return {"built": 0}
        prefixes = []
        for messages in messages_list:
            p = prefix_of(messages)
            if p is not None:
                prefixes.append(p)
        return warm(prefixes)

    def pop_call_metrics(self) -> List[Dict[str, Any]]:
        """Drain per-`chat()` timing metrics recorded since the last pop.

        One entry per LLM call (including json_chat correction retries). Empty
        for the OpenAI backend."""
        buf = getattr(self.backend, "_call_metrics", None)
        if buf is None:
            return []
        drained = list(buf)
        buf.clear()
        return drained

    def json_chat(
        self,
        messages: List[Dict[str, str]],
        required_keys: Iterable[str],
        forbidden_keys: Iterable[str] = (),
    ) -> Dict[str, Any]:
        required_keys = tuple(required_keys)
        forbidden_keys = tuple(forbidden_keys)
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
                                "content": (
                                    "That was not valid JSON. Return exactly ONE FLAT JSON object. "
                                    f"It MUST include top-level keys: {', '.join(required_keys)}"
                                    + (f"; and MUST NOT include: {', '.join(forbidden_keys)}."
                                       if forbidden_keys else ".")
                                    + " Keep it one flat object — do NOT open a new '{' before "
                                    "`stop`/`data_requests`/`final_ranking`; they sit at the top "
                                    "level, not nested inside `findings`."
                                ),
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
