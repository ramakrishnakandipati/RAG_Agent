"""Query rewrite + filter extraction (Qwen JSON output, deterministic guardrails)."""
from __future__ import annotations
import re

from loguru import logger

from src.llm_client import LLMClient
from src.schemas import RewrittenQuery

ACRONYMS = {
    "ADR": "average daily rate",
    "RevPAR": "revenue per available room",
    "OTAs": "online travel agencies",
    "OTA": "online travel agency",
    "CRM": "customer relationship management",
    "LTV": "lifetime value",
    "ROI": "return on investment",
    "GOP": "gross operating profit",
}
_LOOKUP = {k.lower(): (k, v) for k, v in ACRONYMS.items()}
_ACR_RE = re.compile(r"\b(" + "|".join(ACRONYMS) + r")\b", re.I)
_PERIOD = re.compile(r"\bQ([1-4])\s*(?:of\s*)?(20\d{2})\b", re.I)
_RESTRICT = re.compile(r"\b(only|just|exclusively|solely|via|through|channel)\b", re.I)
_CHANNELS = [
    (re.compile(r"\bbooking\.com\b", re.I), "Booking.com OTA"),
    (re.compile(r"\bexpedia\b", re.I), "Expedia OTA"),
    (re.compile(r"\bdirect\b", re.I), "Direct"),
    (re.compile(r"\bcorporate\b", re.I), "Corporate"),
    (re.compile(r"\bwholesale\b", re.I), "Wholesale"),
]
_SEGMENTS = {"leisure": "Leisure", "business": "Business", "group": "Group"}
_PROPERTY_ALIASES = {"miami": "Miami Beach Resort", "orlando": "Orlando Grand Hotel",
                     "fort lauderdale": "Fort Lauderdale Marina Inn"}

SYSTEM = """You rewrite user questions for a hospitality-marketing search index.
Return JSON: {"search_queries": [<=3 standalone queries], "filters": {optional keys among
"period","property","channel","segment","doc_type"}}.
Rules: (1) At most 3 search_queries, each self-contained. (2) Expand every acronym:
ADR=average daily rate, RevPAR=revenue per available room, OTA=online travel agency,
CRM=customer relationship management, LTV=lifetime value. Keep the original acronym in parentheses.
(3) Only add a filter if the user explicitly restricts to it (e.g. "only Q3 2025" -> {"period":"Q3 2025"}).
Period format is like "Q3 2025"."""


def expand_acronyms(q: str) -> tuple[str, dict[str, str]]:
    """Replace each acronym with 'full form (ACR)' unless it is already expanded in the text."""
    used: dict[str, str] = {}

    def sub(m: re.Match) -> str:
        acr, full = _LOOKUP[m.group(1).lower()]
        used[acr] = full
        if re.search(re.escape(full) + r"\s*\(?\s*" + re.escape(acr), q, re.I):
            return m.group(0)
        return f"{full} ({acr})"

    return _ACR_RE.sub(sub, q), used


def extract_filters(q: str) -> dict[str, str]:
    f: dict[str, str] = {}
    m = _PERIOD.search(q)
    if m:
        f["period"] = f"Q{m.group(1)} {m.group(2)}"
    if _RESTRICT.search(q):
        for pat, val in _CHANNELS:
            if pat.search(q):
                f["channel"] = val
                break
    low = q.lower()
    for alias, name in _PROPERTY_ALIASES.items():
        if alias in low:
            f["property"] = name
    for k, v in _SEGMENTS.items():
        if re.search(rf"\b{k}\s+(?:segment|guests|travell?ers)\b", low):
            f["segment"] = v
    return f


class QueryRewriter:
    def __init__(self, llm: LLMClient, max_queries: int = 3):
        self.llm, self.max_queries = llm, max_queries

    def rewrite(self, question: str, vocab: dict[str, list[str]] | None = None) -> RewrittenQuery:
        queries: list[str] = []
        filters: dict[str, str] = {}
        if not self.llm.mock:
            try:
                out = self.llm.json(SYSTEM, question)
                queries = [str(x) for x in out.get("search_queries", []) if str(x).strip()]
                filters = {k: str(v) for k, v in (out.get("filters") or {}).items() if v}
            except Exception as e:
                logger.warning(f"query rewrite via LLM failed, using rules: {e}")
        if not queries:
            queries = [question]
        filters = {**filters, **extract_filters(question)}   # deterministic filters win
        if vocab:  # keep only filter values that exist in the index vocabulary
            filters = {k: v for k, v in filters.items() if k not in vocab or v in vocab[k]
                       or k == "period"}
        expansions: dict[str, str] = {}
        final: list[str] = []
        for q in queries[: self.max_queries]:
            q2, used = expand_acronyms(q)
            expansions.update(used)
            final.append(q2)
        _, used = expand_acronyms(question)
        expansions.update(used)
        return RewrittenQuery(original=question, search_queries=final, filters=filters, expansions=expansions)
