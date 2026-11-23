import pytest

from src.llm_client import LLMClient
from src.retrieval.query_rewriter import QueryRewriter, expand_acronyms
from tests.conftest import FakeLLM

ACR = {"ADR": "average daily rate", "RevPAR": "revenue per available room", "OTA": "online travel agency",
       "CRM": "customer relationship management", "LTV": "lifetime value"}


@pytest.fixture
def rw(cfg):
    c = dict(cfg); c["llm"] = {**cfg["llm"], "mode": "mock"}
    return QueryRewriter(LLMClient(c))


@pytest.mark.parametrize("acr,full", ACR.items())
def test_each_acronym_is_expanded(rw, acr, full):
    rq = rw.rewrite(f"How did {acr} perform across our hotels?")
    assert all(full in q.lower() for q in rq.search_queries)
    assert rq.expansions[acr] == full


def test_all_acronyms_in_one_query(rw):
    rq = rw.rewrite("Compare ADR, RevPAR, OTA mix, CRM and LTV")
    joined = " ".join(rq.search_queries).lower()
    for full in ACR.values():
        assert full in joined


def test_no_double_expansion():
    out, _ = expand_acronyms("average daily rate (ADR) in Q3")
    assert out.count("average daily rate") == 1


def test_query_cap_enforced_even_if_llm_returns_many(cfg):
    sloppy = FakeLLM({"search_queries": [f"q{i} ADR" for i in range(7)], "filters": {}})
    rq = QueryRewriter(sloppy, 3).rewrite("anything")
    assert len(rq.search_queries) <= 3
    assert all("average daily rate" in q for q in rq.search_queries)


def test_only_q3_2025_maps_to_period_filter(rw):
    rq = rw.rewrite("Show ADR only Q3 2025")
    assert rq.filters == {"period": "Q3 2025"}


def test_channel_and_property_filters(rw):
    rq = rw.rewrite("What was RevPAR only via the Direct channel at Miami in Q2 2025?")
    assert rq.filters["channel"] == "Direct"
    assert rq.filters["period"] == "Q2 2025"
    assert rq.filters["property"] == "Miami Beach Resort"


def test_llm_filters_are_merged_but_deterministic_wins(cfg):
    llm = FakeLLM({"search_queries": ["x"], "filters": {"period": "Q1 2025", "channel": "Wholesale"}})
    rq = QueryRewriter(llm).rewrite("only Q3 2025 ADR")
    assert rq.filters["period"] == "Q3 2025" and rq.filters["channel"] == "Wholesale"


@pytest.mark.live
def test_live_qwen_rewrite(cfg):
    rq = QueryRewriter(LLMClient(cfg)).rewrite("What was ADR and RevPAR for OTA bookings, only Q3 2025?")
    assert 1 <= len(rq.search_queries) <= 3 and rq.filters.get("period") == "Q3 2025"
