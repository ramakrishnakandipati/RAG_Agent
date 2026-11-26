"""Strict Pydantic v2 contracts and the LangGraph state schema.

Contract summary
----------------
Agent 1 -> UpsellOfferDraft          Agent 2 -> ComplianceAuditVerdict (APPROVED | REJECTED | ESCALATE_HITL)
State is kept as plain dicts (model_dump) so it serialises cleanly through the LangGraph SQLite checkpointer.
Math symbols: `revision` = r, `max_revisions` = R_max (docs L8).
"""
from __future__ import annotations
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict
import operator

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal["ROOM_UPGRADE", "SPA_PACKAGE", "DINING_EXPERIENCE", "LATE_CHECKOUT"]
Channel = Literal["EMAIL", "SMS"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GuestProfile(Strict):
    reservation_id: str = Field(pattern=r"^RES-\d{5}$")
    first_name: str = "Guest"
    email: str = "guest@example.com"
    segment: Literal["Leisure", "Corporate", "VIP_Loyalty", "Honeymoon", "Family"]
    channel: Literal["Direct_Web", "OTA_Expedia", "GDS"]
    room_type: Literal["STANDARD_QUEEN", "DELUXE_KING", "JUNIOR_OCEAN_VIEW", "OCEANFRONT_SUITE"]
    loyalty_tier: Literal["None", "Silver", "Gold", "Platinum", "Diamond"] = "None"
    existing_tier_perks: list[str] = Field(default_factory=list)
    los: int = Field(ge=1, le=30)
    lead_time: int = Field(ge=0, le=14, description="days to arrival (T-7..T-1)")
    adr: float = Field(gt=0)
    party_size: int = Field(ge=1, le=12)
    past_spend: float = Field(ge=0)
    n_prior_stays: int = Field(ge=0)
    contacts_7d: int = Field(ge=0, default=0)
    is_weekend: int = Field(ge=0, le=1, default=0)


class UpsellOfferDraft(Strict):
    reservation_id: str
    sku: Category
    target_room_type: Optional[str] = None
    offer_price: float = Field(gt=0)
    list_price: float = Field(gt=0)
    pricing_unit: Literal["per_stay", "per_night", "per_person"] = "per_stay"
    bundled_amenities: list[str] = Field(default_factory=list)
    channel: Channel = "EMAIL"
    subject: str
    body_copy: str
    quant_rationale: str
    quant: dict[str, Any] = Field(default_factory=dict, description="cate, p_accept, margin, hurdle, ucb, shadow prices")
    revision: int = Field(ge=0, default=0)
    asset_url: Optional[str] = Field(default=None, description="optional image/preview-card for multimodal visual-alignment audit")
    llm_mode: str = "fallback"

    @field_validator("bundled_amenities")
    @classmethod
    def _uniq(cls, v):
        return list(dict.fromkeys(v))


class ViolationCode(str, Enum):
    INVENTORY_BUFFER = "INVENTORY_BUFFER"
    BELOW_RMS_FLOOR = "BELOW_RMS_FLOOR"
    MARGIN_BELOW_THRESHOLD = "MARGIN_BELOW_THRESHOLD"
    LOYALTY_PERK_DUPLICATION = "LOYALTY_PERK_DUPLICATION"
    SKU_ENTITLED_FREE = "SKU_ENTITLED_FREE"
    INELIGIBLE_SKU = "INELIGIBLE_SKU"
    TONE_BANNED_PHRASE = "TONE_BANNED_PHRASE"
    TONE_SCORE_LOW = "TONE_SCORE_LOW"
    COPY_TOO_LONG = "COPY_TOO_LONG"
    CONTACT_FATIGUE = "CONTACT_FATIGUE"
    VISUAL_MISALIGNMENT = "VISUAL_MISALIGNMENT"


class Violation(Strict):
    code: ViolationCode
    detail: str
    remediation: str


class Directives(Strict):
    """Machine-actionable remediation that Agent 1 applies deterministically on the next revision."""
    min_price: dict[str, float] = Field(default_factory=dict, description="per-SKU minimum price")
    remove_perks: list[str] = Field(default_factory=list)
    exclude_skus: list[str] = Field(default_factory=list)
    refined_tone: bool = False
    shorten_copy: bool = False
    drop_asset: bool = False


class ComplianceAuditVerdict(Strict):
    decision: Literal["APPROVED", "REJECTED", "ESCALATE_HITL"]
    reservation_id: str
    revision: int
    violations: list[Violation] = Field(default_factory=list)
    remediation_instructions: list[str] = Field(default_factory=list)
    directives: Directives = Field(default_factory=Directives)
    checks: dict[str, bool] = Field(default_factory=dict)
    tone_score: Optional[float] = None
    hitl_reason: Optional[str] = None
    vip_value: Optional[float] = None
    llm_mode: str = "fallback"


class CopyOut(Strict):
    """Structured output requested from the LLM for the copywriting step (prices/SKUs are never LLM-chosen)."""
    subject: str = Field(max_length=120)
    body: str
    rationale: str


class ToneAudit(Strict):
    score: float = Field(ge=0, le=1)
    issues: list[str] = Field(default_factory=list)
    verdict: Literal["LUXURY_OK", "OFF_BRAND"]


class VisualAudit(Strict):
    aligned: bool
    notes: str = ""


class GraphState(TypedDict, total=False):
    thread_id: str
    guest: dict
    injections: dict            # stress-test hooks, e.g. {"naive_bundle": True, "price_mult_rev0": 0.7, "aggressive_rev_lt": 1}
    pms_override: dict
    revision: int
    max_revisions: int
    draft: dict
    verdict: dict
    directives: dict            # accumulated across revisions
    history: Annotated[list, operator.add]
    final_status: str           # APPROVED | HITL_PENDING | HITL_APPROVED | HITL_REJECTED
    hitl: dict
