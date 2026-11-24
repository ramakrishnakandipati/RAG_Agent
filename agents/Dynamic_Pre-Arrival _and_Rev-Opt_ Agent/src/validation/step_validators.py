"""Input (pre-condition) / output (post-condition) contract assertions executed at every LangGraph node boundary."""
from __future__ import annotations
from datetime import date
from typing import Callable

from src.config import get_settings


class ValidationFailure(Exception):
    def __init__(self, node: str, phase: str, errors: list[str]):
        self.node, self.phase, self.errors = node, phase, errors
        super().__init__(f"[{node}:{phase}] " + "; ".join(errors))


def _check(node: str, phase: str, conds: list[tuple[bool, str]]) -> None:
    errs = [msg for ok, msg in conds if not ok]
    if errs:
        raise ValidationFailure(node, phase, errs)


def _d(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


# ------------------------------------------------------------------ Node 1: enrich_context
def pre_enrich_context(s: dict) -> None:
    _check("enrich_context", "pre", [
        (bool(s.get("guest_id")), "guest_id missing"),
        (bool(s.get("reservation_id")), "reservation_id missing"),
        (s.get("trigger_type", "pre_arrival") in ("pre_arrival", "inbound_reply", "occupancy_campaign"), "bad trigger_type"),
        (s.get("trigger_type") != "inbound_reply" or bool((s.get("inbound_text") or "").strip()), "inbound_text empty"),
    ])


def post_enrich_context(s: dict) -> None:
    g, st, env = s.get("guest"), s.get("stay"), s.get("env")
    conds = [(bool(g), "guest not found in CRM"), (bool(st), "reservation not found in PMS"), (bool(env), "env signals missing")]
    _check("enrich_context", "post", conds)
    _check("enrich_context", "post", [
        (g["guest_id"] == st["guest_id"], "reservation does not belong to guest"),
        (_d(st["check_out"]) > _d(st["check_in"]), "check_out must be after check_in"),
        (st["party_size"] >= 1, "party_size must be >= 1"),
        (0.0 <= env["precip_prob"] <= 1.0, "precip_prob outside [0,1]"),
        (0.0 <= env.get("precip_prob_max", 0.0) <= 1.0, "precip_prob_max outside [0,1]"),
        (0.0 <= env["occupancy_pct"] <= 100.0, "occupancy outside [0,100]"),
    ])


# ------------------------------------------------------------------ Node 2: retrieve_knowledge_and_offers
def pre_retrieve(s: dict) -> None:
    _check("retrieve_knowledge_and_offers", "pre", [(bool(s.get("guest")), "guest missing"), (bool(s.get("stay")), "stay missing"),
                                                    (bool(s.get("env")), "env missing")])


def post_retrieve(s: dict) -> None:
    cfg = get_settings()
    conds = [(isinstance(s.get("candidates"), list), "candidates not a list"), (len(s.get("kb_snippets", [])) > 0, "no KB snippets")]
    for o in s.get("candidates", []):
        fixed = o["category"] == "loyalty"
        conds += [
            (abs(o["final_price"] - round(o["list_price"] * (1 - o["discount_pct"] / 100), 2)) < 0.011, f"{o['sku']}: price != list*(1-disc)"),
            (fixed or o["discount_pct"] <= cfg.max_discount_pct, f"{o['sku']}: discount above cap"),
            (fixed or o["margin_pct"] >= cfg.min_margin_pct, f"{o['sku']}: margin below floor"),
            (o["available_units"] > 0, f"{o['sku']}: no inventory"),
            (0.0 <= o["propensity"] <= 1.0, f"{o['sku']}: propensity out of range"),
        ]
    _check("retrieve_knowledge_and_offers", "post", conds)


# ------------------------------------------------------------------ Node 3: draft_message
def pre_draft(s: dict) -> None:
    _check("draft_message", "pre", [(bool(s.get("selected_offer")), "no selected offer"), (bool(s.get("guest")), "guest missing")])


def post_draft(s: dict) -> None:
    d = s.get("draft") or {}
    _check("draft_message", "post", [(bool(d.get("body_template")), "draft body empty"), (bool(d.get("subject")), "draft subject empty"),
                                     (len(d.get("body_template", "")) <= 1500, "draft too long")])


# ------------------------------------------------------------------ Node 4: compliance_check
def pre_compliance(s: dict) -> None:
    d = s.get("draft") or {}
    _check("compliance_check", "pre", [(bool(d.get("body_template")), "draft template missing"), (bool(s.get("selected_offer")), "offer missing")])


def post_compliance(s: dict) -> None:
    g = s.get("guardrail") or {}
    _check("compliance_check", "post", [(g.get("status") in ("PASS", "FAIL_POLICY", "FAIL_VALIDATION"), "bad guardrail status")])


# ------------------------------------------------------------------ Node 5: hitl_gate
def pre_hitl(s: dict) -> None:
    _check("hitl_gate", "pre", [(bool(s.get("guardrail")), "guardrail result missing")])


def post_hitl(s: dict) -> None:
    if s.get("hitl_required"):
        dec = s.get("hitl_decision") or {}
        _check("hitl_gate", "post", [(dec.get("decision") in ("approve", "reject"), "HITL decision missing/invalid")])


# ------------------------------------------------------------------ Node 6: dispatch_and_log
def pre_dispatch(s: dict) -> None:
    ok = (s.get("guardrail") or {}).get("status") == "PASS" or (s.get("hitl_decision") or {}).get("decision") == "approve"
    _check("dispatch_and_log", "pre", [(ok, "dispatch requires guardrail PASS or HITL approval")])


def post_dispatch(s: dict) -> None:
    _check("dispatch_and_log", "post", [(s.get("status") in ("SENT", "BOOKED", "SOLD_OUT", "BLOCKED"), f"unexpected final status {s.get('status')}")])


VALIDATORS: dict[str, tuple[Callable, Callable]] = {
    "enrich_context": (pre_enrich_context, post_enrich_context),
    "retrieve_knowledge_and_offers": (pre_retrieve, post_retrieve),
    "draft_message": (pre_draft, post_draft),
    "compliance_check": (pre_compliance, post_compliance),
    "hitl_gate": (pre_hitl, post_hitl),
    "dispatch_and_log": (pre_dispatch, post_dispatch),
}
