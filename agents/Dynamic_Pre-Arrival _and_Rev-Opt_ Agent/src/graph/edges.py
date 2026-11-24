"""Conditional routing: PASS, FAIL_VALIDATION, FAIL_POLICY retry loop, ESCALATE_HITL."""
from __future__ import annotations
from langgraph.graph import END

from src.config import get_settings


def route_after_enrich(state: dict) -> str:
    return END if state.get("status") == "FAIL_VALIDATION" else "retrieve_knowledge_and_offers"


def route_after_retrieve(state: dict) -> str:
    if state.get("status") in ("FAIL_VALIDATION", "NO_OFFER"):
        return END
    return "draft_message"


def route_after_draft(state: dict) -> str:
    return END if state.get("status") == "FAIL_VALIDATION" else "compliance_check"


def route_after_compliance(state: dict) -> str:
    st = state.get("guardrail", {}).get("status")
    if state.get("status") == "FAIL_VALIDATION" or st == "FAIL_VALIDATION":
        return END                                   # hard stop: opt-out, caps, quiet hours, inventory
    if st == "PASS":
        return "hitl_gate"
    if st == "FAIL_POLICY":
        if state.get("retries", 0) <= get_settings().max_retries:
            return "draft_message"                   # retry loop with violation feedback
        return "hitl_gate"                           # ESCALATE_HITL after repeated policy failures
    return END


def route_after_hitl(state: dict) -> str:
    if state.get("status") in ("REJECTED", "FAIL_VALIDATION"):
        return END
    return "dispatch_and_log"


def classify_route(state: dict) -> str:
    """Label used by charts/metrics for the final outcome of a run."""
    if state.get("status") in ("SENT", "BOOKED", "SOLD_OUT") and not state.get("hitl_required"):
        return "PASS_AUTO"
    if state.get("hitl_required"):
        return "ESCALATE_HITL"
    return state.get("status", "UNKNOWN")
