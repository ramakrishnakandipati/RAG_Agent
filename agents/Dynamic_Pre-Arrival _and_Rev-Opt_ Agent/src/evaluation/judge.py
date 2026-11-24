"""Deterministic + LLM-as-a-Judge verifier: 0.0% price/SKU/discount hallucination check and brand-tone score."""
from __future__ import annotations
import re

from src.llm.client import get_llm
from src.tools.compliance_tools import BANNED, check_text_tokens

RUBRIC = ("Score the hotel message 1-5 for brand tone: warm, relaxed, coastal, concise, one clear call to action, no pressure language, "
          "no all caps, no exaggerated claims. Return JSON {\"tone_score\": number, \"reason\": str}.")


def deterministic_check(text: str, offer: dict) -> dict:
    viol = check_text_tokens(text, offer)
    low = text.lower()
    viol += [f"banned_phrase:{b}" for b in BANNED if b in low]
    return {"hallucination_free": not [v for v in viol if "hallucination" in v], "violations": viol}


def heuristic_tone(text: str) -> float:
    score = 5.0
    low = text.lower()
    score -= 1.0 * sum(b in low for b in BANNED)
    score -= 0.5 * len(re.findall(r"!{2,}", text))
    score -= 0.5 * len(re.findall(r"\b[A-Z]{5,}\b", text))
    if "Hi " not in text:
        score -= 0.5
    if len(text) > 900:
        score -= 0.5
    return max(1.0, min(5.0, score))


def judge_message(text: str, offer: dict, use_llm: bool = True) -> dict:
    det = deterministic_check(text, offer)
    tone, reason, src = heuristic_tone(text), "heuristic", "heuristic"
    llm = get_llm()
    if use_llm and not llm.mock:
        try:
            r = llm.json(RUBRIC, f"Message:\n{text}", temperature=0.0)
            tone, reason, src = float(r["tone_score"]), str(r.get("reason", "")), "llm"
        except Exception:  # noqa: BLE001
            pass
    return {**det, "tone_score": max(1.0, min(5.0, tone)), "tone_reason": reason, "tone_source": src}
