"""Citation validation + numerical hallucination detection (regex matcher + LLM-as-judge)."""
from __future__ import annotations
import re

from loguru import logger

from src.llm_client import LLMClient
from src.schemas import Claim, RetrievedChunk, VerificationResult

CITE_RE = re.compile(r"\[([^\[\]|]+?)\s*\|\s*(\d+)\s*\|\s*([^\[\]]+?)\]")
# $ amounts, percentages, multipliers (ROI 3.2x)
NUM_RE = re.compile(r"(?P<cur>\$)?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<suf>%|x\b|k\b|M\b)?", re.I)
REFUSAL_MARKERS = ("insufficient information", "do not contain", "does not contain", "cannot answer",
                   "no information", "not available in the")

JUDGE_SYSTEM = """You are a strict faithfulness judge. Given CONTEXT and an ANSWER, list every sentence of
the ANSWER that is NOT directly supported by CONTEXT (wrong numbers, wrong property/period, invented facts).
A sentence that faithfully paraphrases or summarises CONTEXT IS supported - flag only contradictions, wrong
entities/periods, or facts that appear nowhere in CONTEXT. Copy flagged sentences verbatim.
Return JSON: {"unsupported": ["<sentence>", ...]}. If everything is supported return {"unsupported": []}."""

LEXICAL_OK = 0.6
_WORD = re.compile(r"[a-z]{4,}")


def _content_words(text: str) -> set[str]:
    return set(_WORD.findall(CITE_RE.sub(" ", text).lower()))


def lexical_support(sentence: str, contexts: list[RetrievedChunk]) -> float:
    """Max fraction of the sentence's content words found in any single retrieved chunk."""
    w = _content_words(sentence)
    if not w:
        return 1.0
    return max((len(w & _content_words(c.text)) / len(w) for c in contexts), default=0.0)


def _norm(num: str) -> float:
    return float(num.replace(",", ""))


def extract_numbers(text: str) -> list[tuple[str, str, float]]:
    """-> [(raw, unit, value)] for currency/percent/multiplier numbers only (years, quarters, page ids skipped)."""
    text = CITE_RE.sub(" ", text)
    out = []
    for m in NUM_RE.finditer(text):
        unit = "$" if m.group("cur") else (m.group("suf") or "").lower().replace("m", "m")
        if m.group("cur") and (m.group("suf") or "").lower() == "x":
            unit = "$"
        if unit not in ("$", "%", "x"):
            continue
        out.append((m.group(0).strip(), unit, _norm(m.group("num"))))
    return out


def _sentence_of(text: str, raw: str) -> str:
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if raw in s:
            return s.strip()
    return raw


def count_text_sentences(answer: str) -> int:
    """Non-numeric content sentences (headings and citations ignored); each is one faithfulness check."""
    n = 0
    for line in CITE_RE.sub(" ", answer).splitlines():
        line = re.sub(r"^\s*(•|[-*])\s+", "", line).strip()
        if not line or (line.startswith("**") and line.endswith("**")):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if len(sent.strip(" .")) > 3 and not extract_numbers(sent):
                n += 1
    return n


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


class Verifier:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def verify(self, answer: str, contexts: list[RetrievedChunk], use_judge: bool = True) -> VerificationResult:
        ctx_nums: dict[str, list[tuple[str, float]]] = {}
        for c in contexts:
            for _, unit, val in extract_numbers(c.text):
                ctx_nums.setdefault(unit, []).append((c.chunk_id, val))

        claims: list[Claim] = []
        for raw, unit, val in extract_numbers(answer):
            hit = next((cid for cid, v in ctx_nums.get(unit, []) if abs(v - val) < 0.005), None)
            claims.append(Claim(text=_sentence_of(answer, raw), value=raw, supported=hit is not None,
                                evidence_chunk=hit))
        unsupported = [c for c in claims if not c.supported]

        # citations must point to a retrieved chunk
        valid = {(c.metadata.source, str(c.metadata.page), c.metadata.period) for c in contexts}
        cites = [(a.strip(), p.strip(), per.strip()) for a, p, per in CITE_RE.findall(answer)]
        cit_err = [f"[{a} | {p} | {per}] does not match any retrieved chunk" for a, p, per in cites
                   if (a, p, per) not in valid]
        refusal = is_refusal(answer)
        if claims and not cites and not refusal:
            cit_err.append("answer contains numerical claims but no [source | page | period] citation")

        judge_bad: list[str] = []
        if use_judge and self.llm and not self.llm.mock and not refusal:
            try:
                ctx = "\n\n".join(f"[{c.metadata.source} | {c.metadata.page} | {c.metadata.period}]\n{c.text}"
                                  for c in contexts)
                out = self.llm.json(JUDGE_SYSTEM, f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}")
                already = {c.text for c in unsupported}
                flagged = [s for s in out.get("unsupported", []) if s and s not in already]
                # cross-check the judge: a flagged sentence that is lexically grounded in one chunk is a
                # paraphrase, not a hallucination (numbers are checked strictly by the regex matcher above)
                judge_bad = [s for s in flagged if lexical_support(s, contexts) < LEXICAL_OK]
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}")

        # Score units: every numeric claim, every non-numeric answer sentence, every citation error, plus one
        # sentence-level unit per judge flag that lands on a sentence already counted through its numbers.
        flagged_num = sum(1 for j in judge_bad if extract_numbers(j))
        total = len(claims) + count_text_sentences(answer) + len(cit_err) + flagged_num
        bad = len(unsupported) + len(judge_bad) + len(cit_err)
        total = max(total, bad)  # defensive: a ratio can never exceed 1
        score = 1.0 if total == 0 else max(0.0, round(1 - bad / total, 4))
        return VerificationResult(claims=claims, unsupported_claims=unsupported, citation_errors=cit_err,
                                  judge_unsupported=judge_bad, faithfulness_score=score, is_refusal=refusal)
