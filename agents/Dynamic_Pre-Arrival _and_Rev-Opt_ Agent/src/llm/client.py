"""OpenAI-compatible client for the local Qwen3-VL (LM Studio / vLLM / Ollama) with deterministic mock mode.

The LLM is only used for: drafting copy with placeholders, intent/sentiment classification, tone judging.
It never supplies prices/SKUs - those come from the RMS engine and are substituted deterministically.
"""
from __future__ import annotations
import json
import re
from typing import Optional

from src.config import get_settings
from src.utils.logger import get_logger

log = get_logger("llm")
_THINK = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str) -> dict:
    text = _THINK.sub("", text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
        return json.loads(m.group(0))


class LLMClient:
    def __init__(self, mode: Optional[str] = None):
        self.s = get_settings()
        self.mode = mode or self.s.llm_mode
        self._client = None

    @property
    def mock(self) -> bool:
        return self.mode == "mock"

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.s.llm_base_url, api_key=self.s.llm_api_key, timeout=self.s.llm_timeout)
        return self._client

    def is_up(self) -> bool:
        if self.mock:
            return True
        try:
            return self.s.llm_model in [m.id for m in self.client.models.list().data]
        except Exception:
            return False

    def chat(self, system: str, user: str, temperature: float = 0.4, max_tokens: Optional[int] = None) -> str:
        r = self.client.chat.completions.create(
            model=self.s.llm_model, temperature=temperature, max_tokens=max_tokens or self.s.llm_max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return _THINK.sub("", r.choices[0].message.content or "").strip()

    def json(self, system: str, user: str, temperature: float = 0.0, retries: int = 2,
             schema: Optional[dict] = None) -> dict:
        last = None
        for attempt in range(retries + 1):
            try:
                kw = {}
                if schema and attempt == 0:
                    kw["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "out", "schema": schema, "strict": True}}
                r = self.client.chat.completions.create(
                    model=self.s.llm_model, temperature=temperature, max_tokens=self.s.llm_max_tokens,
                    messages=[{"role": "system", "content": system + "\nReturn ONLY one valid JSON object."},
                              {"role": "user", "content": user}], **kw)
                return extract_json(r.choices[0].message.content or "")
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("llm_json_retry", attempt=attempt, error=f"{type(e).__name__}: {e}")
        raise RuntimeError(f"LLM JSON call failed: {last}")


_singleton: dict[str, LLMClient] = {}


def get_llm(mode: Optional[str] = None) -> LLMClient:
    key = mode or get_settings().llm_mode
    if key not in _singleton:
        _singleton[key] = LLMClient(key)
    return _singleton[key]
