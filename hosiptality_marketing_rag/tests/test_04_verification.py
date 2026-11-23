import re

import pytest

from src.generation.generator import REFUSAL
from src.generation.verifier import Verifier, count_text_sentences, extract_numbers
from tests.conftest import FakeLLM

CITE = "[case_study_miami_beach_resort.md | 1 | Q3 2025]"
GOOD = f"In Q3 2025 the Direct channel at Miami Beach Resort achieved an ADR of $182 and 74% occupancy {CITE}."
BAD = f"In Q3 2025 the Direct channel at Miami Beach Resort achieved an ADR of $299 and 74% occupancy {CITE}."


@pytest.fixture(scope="module")
def ctx(pipeline):
    r = pipeline.query("What was the ADR for the Direct channel at Miami in Q3 2025?")
    return r.contexts


def test_context_contains_cited_chunk(ctx):
    assert any(c.metadata.source == "case_study_miami_beach_resort.md" and c.metadata.period == "Q3 2025" for c in ctx)


def test_number_extraction_ignores_years_and_citations():
    nums = extract_numbers(f"Q3 2025 ADR $1,182.50 up 5% with 3.2x ROI {CITE}")
    assert [(u, v) for _, u, v in nums] == [("$", 1182.5), ("%", 5.0), ("x", 3.2)]


def test_grounded_answer_passes(ctx):
    v = Verifier(None).verify(GOOD, ctx)
    assert v.faithfulness_score >= 0.95
    assert not v.unsupported_claims and not v.citation_errors
    assert {c.value for c in v.claims} >= {"$182", "74%"}


def test_injected_fabricated_metric_is_flagged(ctx):
    v = Verifier(None).verify(BAD, ctx)
    assert [c.value for c in v.unsupported_claims] == ["$299"]
    assert v.faithfulness_score < 0.95
    assert v.faithfulness_score == pytest.approx(0.5)


def test_citation_to_unretrieved_source_flagged(ctx):
    v = Verifier(None).verify("ADR was $182 [made_up_report.pdf | 9 | Q1 2025].", ctx)
    assert v.citation_errors and v.faithfulness_score < 1.0


def test_missing_citation_flagged(ctx):
    v = Verifier(None).verify("ADR was $182 and occupancy 74%.", ctx)
    assert any("no [source" in e for e in v.citation_errors)


def test_unanswerable_tokyo_refuses(pipeline):
    r = pipeline.query("What was Q3 2025 ADR in Tokyo?")
    assert r.answer == REFUSAL
    assert r.verification.is_refusal and not r.verification.unsupported_claims
    assert not re.search(r"\$\s?\d", r.answer)


def test_unanswerable_missing_period_refuses(pipeline):
    r = pipeline.query("What was the ADR in Miami only in Q3 2031?")
    assert not re.search(r"\$\s?\d", r.answer) and r.verification.faithfulness_score >= 0.0
    assert r.verification.is_refusal


def test_end_to_end_answer_is_grounded(pipeline):
    r = pipeline.query("What was the ADR and RevPAR for the Direct channel at Miami Beach Resort in Q3 2025?")
    assert "$182" in r.answer
    assert r.verification.faithfulness_score >= 0.95 and r.citations


@pytest.mark.live
def test_live_qwen_answer_is_faithful(pipeline):
    r = pipeline.query("Summarise Q3 2025 Direct channel performance at Miami Beach Resort.")
    assert r.verification.faithfulness_score >= 0.8


NARRATIVE = "\n".join([
    "**Team overview**",
    f"- The direct channel is covered in the case study {CITE}.",
    f"- Wholesale carried the lowest rate {CITE}.",
    f"- Zebras migrate across volcanic glaciers nightly {CITE}.",
])


def test_count_text_sentences_ignores_headings_numbers_citations():
    assert count_text_sentences(NARRATIVE) == 3
    assert count_text_sentences(GOOD) == 0          # its only sentence is carried by numeric claims
    assert count_text_sentences("**Heading only**") == 0


def test_single_judge_flag_on_narrative_answer_is_not_zero(ctx):
    judge = FakeLLM({"unsupported": ["Zebras migrate across volcanic glaciers nightly"]})
    v = Verifier(judge).verify(NARRATIVE, ctx)
    assert v.judge_unsupported and not v.claims
    assert v.faithfulness_score == pytest.approx(2 / 3, abs=1e-3)   # 1 flag out of 3 sentences, not 0.0


def test_clean_narrative_scores_one(ctx):
    v = Verifier(FakeLLM({"unsupported": []})).verify(NARRATIVE, ctx)
    assert v.faithfulness_score == 1.0


def test_judge_flag_on_supported_numeric_sentence_costs_one_unit(ctx):
    sent = "Zebras migrate across volcanic glaciers for $182"
    ans = f"{sent} {CITE}. Wholesale carried the lowest rate {CITE}."
    v = Verifier(FakeLLM({"unsupported": [sent]})).verify(ans, ctx)
    # units: 1 numeric claim + 1 text sentence + 1 flagged-numeric sentence = 3, bad = 1
    assert v.faithfulness_score == pytest.approx(2 / 3, abs=1e-3)
    assert not v.unsupported_claims
