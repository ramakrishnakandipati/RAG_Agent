"""Grounded executive answer synthesis via Qwen3-Coder (extractive fallback in mock mode)."""
from __future__ import annotations
import re

from src.llm_client import LLMClient
from src.schemas import RetrievedChunk

REFUSAL = ("Insufficient information: the available hospitality marketing documents do not contain "
           "the data needed to answer this question.")

SYSTEM = """You are a hospitality marketing analyst writing for executives.
Answer ONLY from the numbered CONTEXT chunks. Rules:
1. Every factual statement and every number must come from CONTEXT; never use outside knowledge or estimate.
2. After each sentence containing facts, add an inline citation exactly in the form [source | page | period]
   copied from the chunk header (e.g. [case_study_miami_beach_resort.md | 1 | Q3 2025]).
3. If CONTEXT does not contain the answer (different property, period or metric), reply exactly:
   "Insufficient information: the available hospitality marketing documents do not contain the data needed to answer this question."
4. Be concise: a 1-2 sentence headline, then up to 4 bullet points.
5. Quote values exactly as written in CONTEXT. NEVER compute differences, sums, averages, growth rates or percentages
   yourself; describe comparisons in words (higher/lower) and cite the figures that are in CONTEXT."""

_STOP = {"what", "was", "were", "the", "for", "and", "how", "did", "does", "is", "are", "in", "of", "at", "a",
         "an", "to", "only", "show", "me", "our", "my", "us", "by", "with", "about", "which", "per"}
_PLACE = re.compile(r"\b(?:in|at|for|from|of)\s+((?:[A-Z][a-zA-Z.\-]+)(?:\s+[A-Z][a-zA-Z.\-]+)*)")


def cite_tag(c: RetrievedChunk) -> str:
    m = c.metadata
    return f"[{m.source} | {m.page} | {m.period}]"


def out_of_scope(question: str, vocab: dict[str, list[str]]) -> str | None:
    """Rule-based scope guard: a capitalised place name that matches no known property -> out of scope."""
    known = " ".join(vocab.get("property", [])).lower()
    safe = {"q1", "q2", "q3", "q4", "adr", "revpar", "ota", "crm", "ltv", "roi", "direct", "expedia", "booking.com",
            "corporate", "wholesale", "leisure", "business", "group", "loyalty", "email", "summer"}
    for m in _PLACE.finditer(question):
        words = [w for w in re.split(r"\s+", m.group(1)) if w.lower().strip(".") not in safe]
        if words and all(w.lower() not in known for w in words) and not re.match(r"Q[1-4]", m.group(1)):
            return " ".join(words)
    return None


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"CONTEXT {i} {cite_tag(c)}\n{c.text}" for i, c in enumerate(chunks, 1))


class Generator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return REFUSAL
        if self.llm.mock:
            return self._extractive(question, chunks)
        return self.llm.chat(SYSTEM, f"{build_context(chunks)}\n\nQUESTION: {question}")

    @staticmethod
    def _extractive(question: str, chunks: list[RetrievedChunk]) -> str:
        """Deterministic stand-in for the LLM: quote the row summaries that best match the question."""
        terms = {w for w in re.findall(r"[a-z0-9.]+", question.lower()) if w not in _STOP}
        rows = [(c, ln.strip()) for c in chunks for ln in c.text.splitlines() if " channel, " in ln]
        pool = rows or [(c, ln.strip()) for c in chunks for ln in c.text.splitlines() if ln.strip()]
        c, line = max(pool, key=lambda x: sum(t in x[1].lower() for t in terms))
        return f"{line} {cite_tag(c)}"
