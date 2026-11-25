"""Pydantic contracts and LangGraph state for the 6-node OTA-to-Direct agent."""
from __future__ import annotations
import operator
from typing import Annotated, Any, Literal, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator

Window = Literal["pre_arrival", "on_property", "post_stay"]
Channel = Literal["email", "sms", "whatsapp", "wifi_portal"]


class BookingWebhookSchema(BaseModel):
    """Incoming PMS/OTA reservation webhook."""
    booking_id: str = Field(min_length=3)
    ota: Literal["Expedia", "Booking.com", "Agoda"]
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    masked_email: str = Field(min_length=3)
    phone_last4: Optional[str] = None
    country: str = "US"
    checkin_date: str
    los: int = Field(ge=1, le=60)
    adr: float = Field(gt=0)
    ota_commission: float = Field(ge=0.10, le=0.30)
    lead_days: int = Field(ge=0)
    trip_purpose: Literal["leisure", "business", "group"] = "leisure"
    free_cancellation: bool = True
    room_type: str = "deluxe"
    ota_public_rate: Optional[float] = Field(default=None, gt=0, description="OTA displayed nightly rate (parity reference)")
    direct_public_rate: Optional[float] = Field(default=None, gt=0, description="Hotel BAR on own website")
    window: Window = "pre_arrival"
    is_ghost_holdout: Optional[bool] = None

    @field_validator("phone_last4")
    @classmethod
    def _l4(cls, v):
        if v is not None and v == "" or v is not None and not (len(str(v)) == 4 and str(v).isdigit()):
            return None
        return v


class OfferStrategySchema(BaseModel):
    """LLM structured output for strategy selection + personalised copy."""
    window: Window
    offer_id: str = Field(min_length=2)
    channel: Channel
    rationale: str = Field(min_length=5, max_length=600)
    personalized_copy: str = Field(min_length=10, max_length=700)


class DowngradeReasonSchema(BaseModel):
    summary: str = Field(min_length=5, max_length=400)


class GateResult(BaseModel):
    node: str
    phase: Literal["pre", "post"]
    passed: bool
    checks: list[dict[str, Any]]
    ms: float = 0.0


class AgentState(TypedDict, total=False):
    run_id: str
    webhook: dict
    resolved_profile: Optional[dict]
    resolution: dict
    saved_commission_usd: float
    is_holdout: bool
    features: dict
    scores: dict                      # tau, p_cancel, latency_ms, device
    strategy: dict                    # OfferStrategySchema dump
    llm_mode: str                     # "live" | "offline_fallback"
    llm_attempts: int
    rejected_offers: list[str]
    guardrail_retries: int
    guardrail: dict                   # verdict
    final_offer_id: str
    downgraded: bool
    dispatch: dict
    attribution: dict
    gates: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
    error: Optional[str]
