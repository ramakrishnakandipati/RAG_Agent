"""Local OpenAI-compatible client for qwen3-vl-30b-a3b-instruct with strict Pydantic output + offline fallback."""
from __future__ import annotations
import base64, json, re
from pathlib import Path
from typing import Callable, Type, TypeVar
from pydantic import BaseModel, ValidationError
from config.settings import discover_llm, get_settings

T = TypeVar("T", bound=BaseModel)
_cache: dict = {}


def endpoint(refresh: bool = False) -> dict:
    if refresh or "ep" not in _cache or not _cache["ep"]["available"]:
        _cache["ep"] = discover_llm()
    return _cache["ep"]


def _client(ep):
    from openai import OpenAI
    return OpenAI(base_url=ep["base_url"], api_key=get_settings().LLM_API_KEY, timeout=get_settings().LLM_TIMEOUT_S)


def _extract_json(txt: str) -> str:
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    return m.group(0) if m else txt


def structured(system: str, user: str, schema: Type[T], fallback: Callable[[], T], max_retries: int = 2) -> tuple[T, dict]:
    """Returns (validated object, meta{mode, attempts, errors}). Retries malformed output up to max_retries; then falls back."""
    ep = endpoint(); meta = {"mode": "offline_fallback", "attempts": 0, "errors": [], "model": ep["model"]}
    if not ep["available"]:
        return fallback(), meta
    cl = _client(ep); msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(max_retries + 1):
        meta["attempts"] = attempt + 1
        try:
            r = cl.chat.completions.create(model=ep["model"], messages=msgs, temperature=0.2, max_tokens=700,
                                           response_format={"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}})
            raw = r.choices[0].message.content or ""
            obj = schema.model_validate_json(_extract_json(raw)); meta["mode"] = "live"; return obj, meta
        except (ValidationError, json.JSONDecodeError) as e:
            meta["errors"].append(f"schema: {str(e)[:160]}")
            msgs = msgs + [{"role": "user", "content": f"Your previous reply was invalid ({str(e)[:160]}). Reply with ONLY valid JSON for schema {schema.__name__}."}]
        except Exception as e:  # network / server error -> offline fallback
            meta["errors"].append(f"transport: {str(e)[:160]}"); break
    return fallback(), meta | {"mode": "offline_fallback"}


def vision(prompt: str, image_path: Path, schema: Type[T], fallback: Callable[[], T], example: dict | None = None) -> tuple[T, dict]:
    ep = endpoint(); meta = {"mode": "offline_fallback", "model": ep["model"], "errors": []}
    if not ep["available"]:
        return fallback(), meta
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    ex = f"\nExample of the required JSON shape (values are illustrative): {json.dumps(example)}" if example else ""
    try:
        r = _client(ep).chat.completions.create(model=ep["model"], temperature=0.1, max_tokens=700,
            response_format={"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}},
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt + ex + "\nReply with ONE JSON object containing your actual findings for THIS image."},
                                                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        obj = schema.model_validate_json(_extract_json(r.choices[0].message.content or "")); meta["mode"] = "live"; return obj, meta
    except Exception as e:
        meta["errors"].append(str(e)[:200]); return fallback(), meta
