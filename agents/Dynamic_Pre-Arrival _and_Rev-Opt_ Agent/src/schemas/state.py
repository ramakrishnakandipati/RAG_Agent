"""LangGraph AgentState + step trace schema."""
from __future__ import annotations
import operator
from typing import Annotated, Any, Optional, TypedDict


class StepTrace(TypedDict, total=False):
    node: str
    status: str            # OK | FAIL_VALIDATION | ERROR
    latency_ms: float
    pre_ok: bool
    post_ok: bool
    note: str
    ts: str


class AgentState(TypedDict, total=False):
    # identity / trigger
    thread_id: str
    guest_id: str
    reservation_id: str
    trigger_type: str                       # pre_arrival | inbound_reply | occupancy_campaign
    inbound_text: Optional[str]
    adversarial: str                        # none | once | persistent  (test hook: poison draft)
    now_hour: int
    # enriched context
    guest: dict
    stay: dict
    env: dict
    # retrieval + offers
    kb_snippets: list[dict]
    candidates: list[dict]
    selected_offer: Optional[dict]
    # inbound intent
    intent: dict                            # {intent, sentiment, sentiment_score, sku?, custom_request}
    # draft + guardrail
    draft: dict                             # {subject, body_template, body, source}
    guardrail: dict
    retries: int
    feedback: list[str]
    # governance
    hitl_required: bool
    hitl_reasons: list[str]
    hitl_decision: Optional[dict]
    # outcome
    status: str                             # RUNNING | SENT | BOOKED | NO_OFFER | REJECTED | FAIL_VALIDATION | ESCALATED
    booking: Optional[dict]
    dispatched: bool
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[dict], operator.add]
