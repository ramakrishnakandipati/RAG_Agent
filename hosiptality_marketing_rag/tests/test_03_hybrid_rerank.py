import pytest

from src.evaluation.metrics import compare_systems
from src.retrieval.hybrid_retriever import rrf_fuse
from src.retrieval.query_rewriter import QueryRewriter
from src.llm_client import LLMClient

OTHER_QUARTERS = {"Q1 2025", "Q2 2025", "Q4 2025"}
QUESTIONS = ["What was the ADR only Q3 2025?", "Show RevPAR by channel, only Q3 2025",
             "Occupancy for OTA bookings only Q3 2025", "How did CRM and LTV campaigns perform only Q3 2025?"]


def test_rrf_fusion_math():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63) and set(fused) == {"a", "b", "c", "d"}
    assert [k for k, _ in rrf_fuse([["x", "y"], ["x", "z"]])][0] == "x"  # consensus wins


@pytest.mark.gpu
@pytest.mark.parametrize("q", QUESTIONS)
def test_only_q3_2025_returns_zero_other_quarters(pipeline, q):
    rq = QueryRewriter(LLMClient(pipeline.cfg)).rewrite(q, pipeline.index.vocab)
    assert rq.filters["period"] == "Q3 2025"
    for mode in ("dense", "sparse", "hybrid"):
        got = pipeline.retriever.retrieve(rq, 25, mode=mode)
        assert got, f"{mode}: no results"
        assert not [c for c in got if c.metadata.period in OTHER_QUARTERS], f"{mode} leaked other quarters"
        assert all(c.metadata.period == "Q3 2025" for c in got)


@pytest.mark.gpu
def test_chroma_where_and_bm25_filter_directly(pipeline):
    f = {"period": "Q3 2025", "property": "Orlando Grand Hotel"}
    for hits in (pipeline.index.dense_search("average daily rate", 25, f),
                 pipeline.index.sparse_search("average daily rate", 25, f)):
        assert hits and all(h.metadata.period == "Q3 2025" and h.metadata.property == f["property"] for h in hits)


@pytest.mark.gpu
def test_reranker_improves_mrr_over_bm25_and_dense(pipeline, golden):
    res = compare_systems(pipeline, golden, k=5)
    print({k: round(v["mrr@5"], 3) for k, v in res.items()})
    rr = res["rrf+rerank"]["mrr@5"]
    assert rr > res["bm25"]["mrr@5"]
    assert rr >= res["dense"]["mrr@5"]
    assert rr >= res["rrf"]["mrr@5"] - 1e-9
    assert res["rrf+rerank"]["hit@5"] >= 0.9
