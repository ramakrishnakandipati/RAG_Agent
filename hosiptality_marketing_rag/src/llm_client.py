"""OpenAI-compatible client for the local Qwen3-Coder (LM Studio / vLLM / Ollama) with mock mode."""
from __future__ import annotations
import json
import re
from typing import Optional

from loguru import logger

from src.config import load_config


class LLMClient:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or load_config()
        self.cfg = cfg["llm"]
        self.mock = self.cfg["mode"] == "mock"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.cfg["base_url"], api_key=self.cfg["api_key"],
                                  timeout=self.cfg["timeout"])
        return self._client

    def is_up(self) -> bool:
        if self.mock:
            return True
        try:
            return self.cfg["model"] in [m.id for m in self.client.models.list().data]
        except Exception:
            return False

    def chat(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        r = self.client.chat.completions.create(
            model=self.cfg["model"], temperature=self.cfg["temperature"],
            max_tokens=max_tokens or self.cfg["max_tokens"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()

    def json(self, system: str, user: str, schema: dict | None = None, retries: int = 2) -> dict:
        """Structured JSON output: try json_schema response_format, fall back to prompt-only + extraction."""
        last = None
        for attempt in range(retries + 1):
            try:
                kw = {}
                if schema and attempt == 0:
                    kw["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "out", "schema": schema, "strict": True}}
                r = self.client.chat.completions.create(
                    model=self.cfg["model"], temperature=0, max_tokens=self.cfg["max_tokens"],
                    messages=[{"role": "system", "content": system + "\nReturn ONLY valid JSON."},
                              {"role": "user", "content": user}], **kw)
                return extract_json(r.choices[0].message.content or "")
            except Exception as e:  # noqa
                last = e
                logger.warning(f"LLM json attempt {attempt} failed: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM JSON call failed: {last}")


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0) if m else text)
