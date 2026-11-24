"""Strict Pydantic v2 domain + API models."""
from __future__ import annotations
from datetime import date, datetime
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

Tier = Literal["Basic", "Silver", "Gold", "Platinum"]
Channel = Literal["OTA", "Direct"]
Category = Literal["room_upgrade", "spa", "dining", "late_checkout", "loyalty"]
TriggerType = Literal["pre_arrival", "inbound_reply", "occupancy_campaign"]


class Guest360(BaseModel):
    model_config = ConfigDict(extra="forbid")
    guest_id: str
    first_name: str
    last_name: str
    email: str
    segment: Literal["Leisure", "Business", "Family", "Couple", "Group"]
    loyalty_tier: Tier
    country: str
    total_stays: int = Field(ge=0)
    ota_stays: int = Field(ge=0)
    direct_stays: int = Field(ge=0)
    ltv_usd: float = Field(ge=0)
    avg_adr: float = Field(gt=0)
    opt_in_marketing: bool
    avg_sentiment: float = Field(ge=-1, le=1)

    @property
    def is_vip(self) -> bool:
        return self.loyalty_tier == "Platinum"


class StayContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reservation_id: str
    guest_id: str
    channel: str
    channel_group: Channel
    check_in: date
    check_out: date
    nights: int = Field(ge=1)
    party_size: int = Field(ge=1)
    room_type: str
    adr: float = Field(gt=0)
    lead_days: int = Field(ge=0)
    status: str

    @model_validator(mode="after")
    def _dates(self):
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


class Offer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str
    name: str
    category: Category
    list_price: float = Field(ge=0)
    discount_pct: float = Field(ge=0, le=100)
    final_price: float = Field(ge=0)
    cost: float = Field(ge=0)
    margin_pct: float
    propensity: float = Field(ge=0, le=1)
    expected_value: float
    available_units: int = Field(ge=0)
    service_date: date
    rationale: str = ""


class GuardrailResult(BaseModel):
    status: Literal["PASS", "FAIL_POLICY", "FAIL_VALIDATION"]
    violations: list[str] = []
    checks: dict[str, bool] = {}
    tone_score: Optional[float] = None


class AuditLog(BaseModel):
    ts: datetime
    thread_id: str
    reservation_id: str
    guest_id: str
    action: str
    sku: Optional[str] = None
    amount: Optional[float] = None
    status: str
    detail: dict = {}


# ------------------------------------------------------------------ API models
class TriggerRequest(BaseModel):
    reservation_id: str = Field(min_length=3)
    trigger_type: TriggerType = "pre_arrival"
    adversarial: Literal["none", "once", "persistent"] = "none"   # test hook: inject fabricated-price draft(s) to exercise the gate


class ReplyRequest(BaseModel):
    reservation_id: str = Field(min_length=3)
    text: str = Field(min_length=1, max_length=1000)
    thread_id: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
    reviewer: str = "concierge"
    edited_body: Optional[str] = None


class WorkflowResponse(BaseModel):
    thread_id: str
    status: str
    interrupted: bool = False
    hitl_reasons: list[str] = []
    message: Optional[dict] = None
    offer: Optional[dict] = None
    guardrail: Optional[dict] = None
    booking: Optional[dict] = None
    trace: list[dict] = []
    errors: list[str] = []
