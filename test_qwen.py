"""Smoke test for local Qwen models served by LM Studio (OpenAI-compatible API)."""
import sys, time
from openai import OpenAI

BASE_URL = "http://localhost:1234/v1"
MODEL = "qwen3-coder-30b-a3b-instruct"

client = OpenAI(base_url=BASE_URL, api_key="lm-studio", timeout=300)


def check(name, fn):
    t = time.perf_counter()
    try:
        detail = fn()
        print(f"[PASS] {name} ({time.perf_counter()-t:.1f}s): {detail}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def list_models():
    ids = [m.id for m in client.models.list().data]
    assert MODEL in ids, f"{MODEL} not loaded; available: {ids}"
    return ids


def chat():
    r = client.chat.completions.create(
        model=MODEL, max_tokens=64, temperature=0,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}])
    text = r.choices[0].message.content.strip()
    assert "pong" in text.lower(), text
    return f"{text!r} | tokens={r.usage.completion_tokens}"


def code_gen():
    r = client.chat.completions.create(
        model=MODEL, max_tokens=300, temperature=0,
        messages=[{"role": "user", "content": "Write a Python function is_prime(n). Code only."}])
    text = r.choices[0].message.content
    assert "def is_prime" in text, text
    return "got is_prime definition"


def stream():
    t0 = time.perf_counter(); n = 0
    for ch in client.chat.completions.create(
            model=MODEL, max_tokens=100, stream=True,
            messages=[{"role": "user", "content": "Count from 1 to 10."}]):
        if ch.choices and ch.choices[0].delta.content:
            n += 1
    return f"{n} chunks, ~{n/(time.perf_counter()-t0):.1f} chunks/s"


if __name__ == "__main__":
    results = [check("list models", list_models), check("chat", chat),
               check("code generation", code_gen), check("streaming", stream)]
    sys.exit(0 if all(results) else 1)
