"""Deterministic Auditor contracts (docs L7): each violation code is triggered in isolation."""
import pytest
from agents.auditor_agent import audit, heuristic_tone
from agents.schemas import UpsellOfferDraft
from agents.tools import SCENARIOS


def mk(guest, **kw):
    base = dict(reservation_id=guest.reservation_id, sku="SPA_PACKAGE", offer_price=155.0, list_price=180.0, pricing_unit="per_person", bundled_amenities=[],
                channel="EMAIL", subject="A considered touch", body_copy="Dear guest, we would be delighted to arrange a spa experience.", quant_rationale="x", revision=0)
    base.update(kw)
    return UpsellOfferDraft(**base)


G = SCENARIOS["clean_approval"]["guest"]
DIAMOND = SCENARIOS["vip_duplicate_perk"]["guest"]


def codes(v):
    return {x.code.value for x in v.violations}


def test_clean_draft_approved():
    assert audit(G, mk(G), 0, 3).decision == "APPROVED"


def test_inventory_buffer_blocks_upgrade():
    d = mk(G, sku="ROOM_UPGRADE", offer_price=150.0, list_price=132.0, pricing_unit="per_night", target_room_type="JUNIOR_OCEAN_VIEW")
    v = audit(G, d, 0, 3, {"JUNIOR_OCEAN_VIEW": 1})
    assert "INVENTORY_BUFFER" in codes(v) and "ROOM_UPGRADE" in v.directives.exclude_skus
    assert "INVENTORY_BUFFER" not in codes(audit(G, d, 0, 3, {"JUNIOR_OCEAN_VIEW": 25}))


def test_floor_directive():
    v = audit(G, mk(G, offer_price=100.0), 0, 3)
    assert "BELOW_RMS_FLOOR" in codes(v) and v.directives.min_price["SPA_PACKAGE"] >= 150


def test_margin_floor():
    v = audit(G, mk(G, sku="LATE_CHECKOUT", offer_price=38.0, list_price=45.0), 0, 3)       # (38-17)/38 = 55% ok
    assert "MARGIN_BELOW_THRESHOLD" not in codes(v)
    v = audit(G, mk(G, sku="DINING_EXPERIENCE", offer_price=115.0, list_price=140.0), 0, 3)  # (115-53)/115 = 54% ok
    assert "MARGIN_BELOW_THRESHOLD" not in codes(v)


def test_loyalty_parity_and_entitled_sku():
    v = audit(DIAMOND, mk(DIAMOND, bundled_amenities=["LATE_CHECKOUT_2PM", "WELCOME_DRINK"]), 0, 3)
    assert "LOYALTY_PERK_DUPLICATION" in codes(v) and set(v.directives.remove_perks) == {"LATE_CHECKOUT_2PM", "WELCOME_DRINK"}
    v2 = audit(DIAMOND, mk(DIAMOND, sku="LATE_CHECKOUT", offer_price=60.0, list_price=56.0), 0, 3)
    assert "SKU_ENTITLED_FREE" in codes(v2)


@pytest.mark.parametrize("txt", ["ACT NOW and save", "only 50% OFF today", "such a CHEAP upgrade", "Wonderful!!"])
def test_tone_regex(txt):
    v = audit(G, mk(G, body_copy=txt), 0, 3)
    assert "TONE_BANNED_PHRASE" in codes(v) and v.decision == "REJECTED"


def test_tone_heuristic_scores():
    assert heuristic_tone("We would be delighted to welcome you.").score > 0.9
    assert heuristic_tone("ACT NOW!! CHEAP 50% OFF").score < 0.3


def test_copy_length_and_unresolved_escalation():
    assert "COPY_TOO_LONG" in codes(audit(G, mk(G, channel="SMS", body_copy="x" * 400), 0, 3))
    assert audit(G, mk(G, offer_price=100.0), 3, 3).decision == "ESCALATE_HITL"          # R_max reached -> HITL


def test_vip_value_hitl():
    v = audit(DIAMOND, mk(DIAMOND, offer_price=281.0, list_price=225.0), 0, 3)
    assert v.decision == "ESCALATE_HITL" and "VIP" in v.hitl_reason


def test_contact_fatigue_escalates():
    g = G.model_copy(update={"contacts_7d": 3})
    assert audit(g, mk(g), 0, 3).decision == "ESCALATE_HITL"
