"""Generator-Critic loop: bounded revisions, scenario outcomes, HITL interrupt/resume."""
import pytest
from agents.graph import build_graph, run_offer, resume_hitl
from agents.schemas import GuestProfile, UpsellOfferDraft, ComplianceAuditVerdict
from agents.tools import SCENARIOS


@pytest.fixture()
def graph():
    return build_graph()


def go(name, graph, **kw):
    s = SCENARIOS[name]
    return run_offer(s["guest"].model_dump(), s["injections"], s["pms_override"], graph=graph, **kw)


def drafts(o):
    return [h for h in o["history"] if h["agent"] == "strategist"]


def test_clean_first_pass(graph):
    o = go("clean_approval", graph)
    assert o["final_status"] == "APPROVED" and len(drafts(o)) == 1


@pytest.mark.parametrize("name,code", [("below_floor_price", "BELOW_RMS_FLOOR"), ("spammy_copy", "TONE_BANNED_PHRASE"), ("last_room_upgrade", "INVENTORY_BUFFER")])
def test_reject_then_revise_to_approval(graph, name, code):
    o = go(name, graph); aud = [h for h in o["history"] if h["agent"] == "auditor"]
    assert code in aud[0]["violations"] and aud[0]["decision"] == "REJECTED"
    assert o["final_status"] == "APPROVED" and len(drafts(o)) == 2 and o["verdict"]["violations"] == []


def test_vip_duplicate_perk_then_hitl(graph):
    o = go("vip_duplicate_perk", graph)
    assert o["final_status"] == "HITL_PENDING"
    first = [h for h in o["history"] if h["agent"] == "auditor"][0]
    assert "LOYALTY_PERK_DUPLICATION" in first["violations"]
    assert resume_hitl(o["thread_id"], "approve", "tester", graph=graph)["final_status"] == "HITL_APPROVED"


def test_unresolvable_stops_at_rmax_and_escalates(graph):
    o = go("unresolvable_escalation", graph)
    assert len(drafts(o)) == 4 and o["final_status"] == "HITL_PENDING"       # initial + 3 revisions
    assert o["verdict"]["decision"] == "ESCALATE_HITL"
    assert resume_hitl(o["thread_id"], "reject", graph=graph)["final_status"] == "HITL_REJECTED"


@pytest.mark.parametrize("rmax", [0, 1, 2])
def test_revision_bound_respected(graph, rmax):
    o = go("unresolvable_escalation", graph, max_revisions=rmax)
    assert len(drafts(o)) == rmax + 1


def test_schema_contracts_are_strict():
    with pytest.raises(Exception):
        UpsellOfferDraft(reservation_id="RES-00001", sku="BOGUS", offer_price=1, list_price=1, subject="s", body_copy="b", quant_rationale="q")
    with pytest.raises(Exception):
        GuestProfile(reservation_id="X1", segment="Leisure", channel="GDS", room_type="DELUXE_KING", los=1, lead_time=1, adr=1, party_size=1, past_spend=0, n_prior_stays=0)
    with pytest.raises(Exception):
        ComplianceAuditVerdict(decision="MAYBE", reservation_id="RES-00001", revision=0)
