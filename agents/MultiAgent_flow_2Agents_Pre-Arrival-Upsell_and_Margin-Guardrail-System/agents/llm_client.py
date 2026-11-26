"""OpenAI-compatible client for the local qwen3-vl-30b-a3b-instruct server (vLLM / Ollama / SGLang).

* ChatOpenAI(base_url=.../v1, model='qwen3-vl-30b-a3b-instruct', temperature=0.15, max_retries=3)
* `structured()` enforces a Pydantic JSON schema via `with_structured_output` (json_schema method, falling back to function calling).
* `vision()` is the multimodal hook (image_url content part) used for visual-offer-alignment checks.
* If the server is unreachable (or llm.mode == 'mock') callers receive None and use deterministic fallbacks; the active mode is
  reported as 'qwen' or 'fallback' on every draft/verdict so traces are honest about what ran.
"""
from __future__ import annotations
import base64
import re
import sys
import time
from pathlib import Path
from typing import Optional, Type, TypeVar

import requests
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.utils import load_config  # noqa: E402

T = TypeVar("T", bound=BaseModel)
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM:
    def __init__(self, force_mode: Optional[str] = None):
        self.cfg = load_config()["llm"]; self.mode_cfg = force_mode or self.cfg["mode"]
        self._chat = None; self._up: tuple[float, bool] = (0.0, False)

    def available(self) -> bool:
        if self.mode_cfg == "mock":
            return False
        t, ok = self._up
        if time.time() - t < 20:
            return ok
        try:
            r = requests.get(self.cfg["base_url"].rstrip("/") + "/models", timeout=1.5, headers={"Authorization": f"Bearer {self.cfg['api_key']}"})
            ok = r.status_code == 200
        except Exception:
            ok = False
        self._up = (time.time(), ok)
        return ok

    @property
    def mode(self) -> str:
        return "qwen" if self.available() else "fallback"

    @property
    def chat(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI
            self._chat = ChatOpenAI(base_url=self.cfg["base_url"], api_key=self.cfg["api_key"], model=self.cfg["model"], temperature=self.cfg["temperature"],
                                    max_retries=self.cfg["max_retries"], timeout=self.cfg["timeout_s"])
        return self._chat

    def structured(self, schema: Type[T], system: str, user: str) -> Optional[T]:
        if not self.available():
            return None
        msgs = [("system", system + "\nRespond ONLY with JSON matching the schema."), ("human", user)]
        for method in ("json_schema", "function_calling"):
            try:
                out = self.chat.with_structured_output(schema, method=method).invoke(msgs)
                return out if isinstance(out, schema) else schema.model_validate(out)
            except Exception:  # noqa: BLE001 - try next method then fall back
                continue
        return None

    def vision(self, schema: Type[T], prompt: str, image: str) -> Optional[T]:
        """image: http(s) URL, data URL or local file path."""
        if not self.available():
            return None
        url = image
        if not image.startswith(("http", "data:")) and Path(image).exists():
            url = "data:image/png;base64," + base64.b64encode(Path(image).read_bytes()).decode()
        from langchain_core.messages import HumanMessage
        msg = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}])
        try:
            return self.chat.with_structured_output(schema, method="json_schema").invoke([msg])
        except Exception:  # noqa: BLE001
            return None


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
