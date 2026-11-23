import pytest
from pydantic import ValidationError

from src.ingest.chunker import count_tokens, linearize_row
from src.ingest.parser import parse_dir
from src.config import resolve
from src.schemas import ChunkMetadata

ANCHOR = "Q3 2025, Direct channel, ADR (Average Daily Rate) $182, Occupancy 74%"
FIELDS = {"source", "section", "page", "property", "period", "channel", "segment", "doc_type"}


def test_tables_never_split(chunks, cfg):
    tables = [e for e in parse_dir(resolve(cfg, "raw")) if e.kind == "table"]
    assert len(tables) >= 20
    for el in tables:
        holders = [c for c in chunks if el.text in c.text]
        assert len(holders) == 1, f"table in {el.source} p{el.page} appears in {len(holders)} chunks"
        assert holders[0].kind == "table" and holders[0].table_md == el.text
        # no row of the table leaks into any other chunk
        row = el.text.splitlines()[2]
        assert sum(row in c.text.splitlines() for c in chunks) == 1


def test_row_linearization_anchor(chunks):
    hits = [c for c in chunks if c.kind == "table" and ANCHOR in c.text]
    assert hits, "anchor sentence 'Q3 2025, Direct channel, ADR (Average Daily Rate) $182, Occupancy 74%' missing"
    assert "RevPAR (Revenue Per Available Room) $134.68" in hits[0].text
    # every table chunk carries one linearized sentence per markdown data row
    for c in (x for x in chunks if x.kind == "table"):
        n_rows = len(c.table_md.splitlines()) - 2
        assert c.text.split("Row summaries:\n")[1].count("channel,") >= n_rows or n_rows == 0


def test_linearize_row_unit():
    import pandas as pd
    s = linearize_row(pd.Series({"Period": "Q3 2025", "Channel": "Direct", "ADR": "$182", "Occupancy": "74%"}))
    assert s == "Q3 2025, Direct channel, ADR (Average Daily Rate) $182, Occupancy 74%."


def test_metadata_schema_all_eight_fields(chunks):
    assert set(ChunkMetadata.model_fields) == FIELDS
    for c in chunks:
        ChunkMetadata.model_validate(c.metadata.model_dump())
        assert c.metadata.doc_type in ("case_study", "campaign_report", "brand_guidelines")
    assert {c.metadata.doc_type for c in chunks} == {"case_study", "campaign_report", "brand_guidelines"}


def test_metadata_rejects_bad_doc_type():
    base = dict(source="a.md", section="s", page=1, property="p", period="Q3 2025", channel="Direct",
                segment="all", doc_type="case_study")
    ChunkMetadata(**base)
    with pytest.raises(ValidationError):
        ChunkMetadata(**{**base, "doc_type": "press_release"})
    with pytest.raises(ValidationError):
        ChunkMetadata(**{**base, "period": ""})


def test_text_chunks_respect_token_budget(chunks):
    for c in chunks:
        if c.kind == "text":
            assert count_tokens(c.text) <= 512 + 8
