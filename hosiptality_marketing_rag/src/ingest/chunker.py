"""Table-preserving + row-linearizing chunker with strict metadata tagging."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_config, resolve
from src.ingest.parser import ParsedElement, parse_dir
from src.schemas import Chunk, ChunkMetadata

KNOWN_PROPERTIES = ["Miami Beach Resort", "Orlando Grand Hotel", "Fort Lauderdale Marina Inn"]
KNOWN_CHANNELS = ["Direct", "Booking.com OTA", "Expedia OTA", "Corporate", "Wholesale"]
KNOWN_SEGMENTS = ["Leisure", "Business", "Group"]
PERIOD_RE = re.compile(r"\bQ[1-4]\s*20\d{2}\b")

GLOSSARY = {  # column name -> (label, value formatter)
    "ADR": "ADR (Average Daily Rate)",
    "RevPAR": "RevPAR (Revenue Per Available Room)",
    "Occupancy": "Occupancy",
    "ROI": "ROI (Return On Investment)",
    "LTV": "LTV (Lifetime Value)",
}

_tok = None


def _tokenizer():
    global _tok
    if _tok is None:
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(load_config()["models"]["embedder"])
        except Exception:  # offline fallback
            _tok = False
    return _tok


def count_tokens(text: str) -> int:
    t = _tokenizer()
    if t:
        return len(t.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


def doc_type_for(source: str) -> str:
    s = source.lower()
    if s.startswith("campaign_report"):
        return "campaign_report"
    if s.startswith("brand_guidelines"):
        return "brand_guidelines"
    return "case_study"  # case studies and KPI sheets


def find_vocab(text: str, vocab: list[str]) -> Optional[str]:
    low = text.lower().replace("_", " ")
    for v in vocab:
        if v.lower() in low:
            return v
    return None


def norm_period(s: str) -> str:
    m = PERIOD_RE.search(s)
    return re.sub(r"\s+", " ", m.group(0)).upper() if m else ""


def linearize_row(row: pd.Series, prop: str = "") -> str:
    """One natural-language sentence per KPI row, e.g.
    'Miami Beach Resort: Q3 2025, Direct channel, ADR (Average Daily Rate) $182, Occupancy 74%, ...'"""
    parts: list[str] = []
    cols = {str(c): c for c in row.index}
    if "Period" in cols:
        parts.append(str(row[cols["Period"]]))
    if "Channel" in cols:
        parts.append(f"{row[cols['Channel']]} channel")
    if "Segment" in cols:
        parts.append(f"{row[cols['Segment']]} segment")
    for name, c in cols.items():
        if name in ("Period", "Channel", "Segment", "Property"):
            continue
        label = GLOSSARY.get(name, name)
        parts.append(f"{label} {row[c]}")
    s = ", ".join(parts)
    return f"{prop}: {s}." if prop else f"{s}."


def _unique(df: pd.DataFrame, col: str, vocab: list[str]) -> str:
    if col in df.columns:
        vals = {str(v) for v in df[col]}
        if len(vals) == 1:
            return vals.pop()
    return "all"


def table_chunk(el: ParsedElement, idx: int) -> Chunk:
    df = el.table
    prop = (_unique(df, "Property", KNOWN_PROPERTIES))
    if prop == "all":
        prop = find_vocab(el.doc_title + " " + el.source, KNOWN_PROPERTIES) or "all"
    period = _unique(df, "Period", [])
    if period == "all":
        period = norm_period(el.section) or "all"
    channel = _unique(df, "Channel", KNOWN_CHANNELS)
    segment = _unique(df, "Segment", KNOWN_SEGMENTS)
    sentences = [linearize_row(r, "" if prop == "all" else prop) for _, r in df.iterrows()]
    text = f"{el.section}\n\n{el.text}\n\nRow summaries:\n" + "\n".join(sentences)
    md = ChunkMetadata(source=el.source, section=el.section, page=el.page, property=prop,
                       period=period, channel=channel, segment=segment, doc_type=doc_type_for(el.source))
    return Chunk(chunk_id=f"{el.source}::p{el.page}::t{idx}", text=text, kind="table",
                 table_md=el.text, metadata=md)


def embedding_text(c: Chunk) -> str:
    """Text fed to the dense embedder. Tables: section + linearized sentences only (the raw
    markdown table stays whole in `text`/`table_md` but is not truncated mid-table for retrieval)."""
    if c.kind == "table":
        sents = c.text.split("Row summaries:\n", 1)[-1]
        return f"{c.metadata.section}\n{sents}"
    return c.text


def split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur, cur_tok = [], [], 0
    for s in sents:
        n = count_tokens(s)
        if cur and cur_tok + n > max_tokens:
            chunks.append(" ".join(cur))
            keep, kt = [], 0
            for prev in reversed(cur):  # sentence-level overlap
                pt = count_tokens(prev)
                if kt + pt > overlap:
                    break
                keep.insert(0, prev); kt += pt
            cur, cur_tok = keep, kt
        cur.append(s); cur_tok += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def text_chunks(el: ParsedElement, start: int, max_tokens: int, overlap: int) -> list[Chunk]:
    out = []
    prop = find_vocab(el.doc_title + " " + el.source + " " + el.section + " " + el.text, KNOWN_PROPERTIES) or "all"
    period = norm_period(el.section)
    if not period:
        ps = {norm_period(m) for m in PERIOD_RE.findall(el.text)}
        period = ps.pop() if len(ps) == 1 else "all"
    channel = find_vocab(el.text, KNOWN_CHANNELS) or "all"
    segment = find_vocab(el.text, KNOWN_SEGMENTS) or "all"
    budget = max_tokens - count_tokens(el.section) - 4
    for i, piece in enumerate(split_text(el.text, budget, overlap)):
        md = ChunkMetadata(source=el.source, section=el.section, page=el.page, property=prop,
                           period=period, channel=channel, segment=segment, doc_type=doc_type_for(el.source))
        out.append(Chunk(chunk_id=f"{el.source}::p{el.page}::x{start + i}",
                         text=f"{el.section}\n{piece}", kind="text", metadata=md))
    return out


def chunk_elements(elements: list[ParsedElement], cfg: dict | None = None) -> list[Chunk]:
    cfg = cfg or load_config()
    mt, ov = cfg["chunking"]["chunk_tokens"], cfg["chunking"]["overlap_tokens"]
    chunks: list[Chunk] = []
    for i, el in enumerate(elements):
        if el.kind == "table":
            chunks.append(table_chunk(el, i))
        else:
            chunks += text_chunks(el, i * 100, mt, ov)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    return chunks


def build_chunks(cfg: dict | None = None, raw_dir: Path | None = None, write: bool = True) -> list[Chunk]:
    cfg = cfg or load_config()
    raw = Path(raw_dir) if raw_dir else resolve(cfg, "raw")
    chunks = chunk_elements(parse_dir(raw), cfg)
    if write:
        p = resolve(cfg, "chunks")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
    return chunks


def load_chunks(cfg: dict | None = None) -> list[Chunk]:
    cfg = cfg or load_config()
    with open(resolve(cfg, "chunks"), encoding="utf-8") as f:
        return [Chunk.model_validate_json(l) for l in f if l.strip()]
