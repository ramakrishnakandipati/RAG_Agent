"""Enterprise tool layer: CRM, live PMS inventory, RMS floors, Loyalty perks, and GPU quant inference.

Math used here (docs L6/L7):
  safety buffer   B(room) = max( B_min, ceil( z * sigma_D * sqrt(L) ) ),  sigma_D = 0.07 * capacity   (PMS)
  contribution    CM = p - c_COGS - c_HK,  CM/p >= theta_m                                          (RMS)
  candidate score S_k = tau_k(x) * (p_k - c_k) + 100*alpha*sqrt(x^T A_a^{-1} x) - h_k ,  h_k = lambda_inv,k + lambda_B * disc_k
  price           p_k* = argmax_{p >= floor, p>=p_min} (p - c) * P_hat(x,k,p)
"""
from __future__ import annotations
import json
import math
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.synthetic_generator import CATS, PRICE_MULTS, featurize, list_price, unit_margin, eligible  # noqa: E402
from models.utils import get_device, load_config, load_logs, load_policies, path, timed, jsonl_append  # noqa: E402
from .schemas import GuestProfile  # noqa: E402

CFG, POL = load_config(), load_policies()
UPGRADE_PATH = {"STANDARD_QUEEN": "DELUXE_KING", "DELUXE_KING": "JUNIOR_OCEAN_VIEW", "JUNIOR_OCEAN_VIEW": "OCEANFRONT_SUITE", "OCEANFRONT_SUITE": None}
SERVICE_CAPACITY = {"SPA_PACKAGE": 24, "DINING_EXPERIENCE": 40, "LATE_CHECKOUT": 60}
CATEGORY_PERK = {"LATE_CHECKOUT": "LATE_CHECKOUT_2PM", "ROOM_UPGRADE": "ROOM_UPGRADE_SUBJECT_TO_AVAIL"}


# ------------------------------------------------------------------------------------------------ CRM
@lru_cache(maxsize=1)
def _logs() -> pd.DataFrame:
    return load_logs().set_index("reservation_id", drop=False)


def loyalty_perks(tier: str) -> list[str]:
    return list(POL["loyalty_perks"].get(tier, []))


def crm_get_guest(reservation_id: str) -> GuestProfile:
    r = _logs().loc[reservation_id]
    return GuestProfile(
        reservation_id=reservation_id, first_name=str(r.first_name), email=str(r.email), segment=str(r.segment), channel=str(r.channel),
        room_type=str(r.room_type), loyalty_tier=str(r.loyalty_tier), existing_tier_perks=loyalty_perks(str(r.loyalty_tier)),
        los=int(r.los), lead_time=int(r.lead_time), adr=float(r.adr), party_size=int(r.party_size), past_spend=float(r.past_spend),
        n_prior_stays=int(r.n_prior_stays), contacts_7d=int(r.contacts_7d), is_weekend=int(r.is_weekend))


def guest_frame(g: GuestProfile) -> pd.DataFrame:
    d = g.model_dump(); d["loyalty_tier"] = d["loyalty_tier"]
    return pd.DataFrame([d])


# ------------------------------------------------------------------------------------------------ PMS
class LivePMS:
    """In-memory live inventory; mutated by the Flask PMS webhook. Thread-safe."""
    def __init__(self):
        self._lock = threading.Lock(); self.reset()

    def reset(self):
        t = pd.read_csv(path("data") / "pms_inventory.csv") if (path("data") / "pms_inventory.csv").exists() else None
        with self._lock:
            self.capacity = dict(zip(t.room_type, t.capacity)) if t is not None else {"STANDARD_QUEEN": 60, "DELUXE_KING": 48, "JUNIOR_OCEAN_VIEW": 30, "OCEANFRONT_SUITE": 12}
            self.available = dict(zip(t.room_type, t.live_available_rooms)) if t is not None else {r: c // 3 for r, c in self.capacity.items()}
            for k, c in SERVICE_CAPACITY.items():
                self.capacity[k] = c; self.available[k] = int(c * 0.6)

    def update(self, key: str, available: int):
        with self._lock:
            if key not in self.capacity:
                raise KeyError(key)
            self.available[key] = int(available)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: {"capacity": self.capacity[k], "available": int(self.available[k]), "buffer": self.dynamic_safety_buffer(k)} for k in self.capacity}

    def dynamic_safety_buffer(self, key: str) -> int:
        sigma = 0.07 * self.capacity[key]; p = CFG["pms"]
        return int(max(p["min_buffer_rooms"], math.ceil(p["safety_z"] * sigma * math.sqrt(p["lead_time_days"]))))

    def inventory_key(self, sku: str, room_type: str) -> str | None:
        return UPGRADE_PATH.get(room_type) if sku == "ROOM_UPGRADE" else sku

    def check(self, sku: str, room_type: str, overrides: dict | None = None) -> dict:
        key = self.inventory_key(sku, room_type)
        if key is None:
            return {"key": None, "available": 0, "buffer": 0, "ok": False, "reason": "no higher room category"}
        with self._lock:
            avail = (overrides or {}).get(key, self.available[key]); buf = self.dynamic_safety_buffer(key)
        return {"key": key, "available": int(avail), "buffer": buf, "ok": avail >= buf, "reason": ""}


PMS = LivePMS()


# ------------------------------------------------------------------------------------------------ RMS
def rms_floor(sku: str) -> float:
    return float(CFG["rms"]["min_upsell_floor"][sku])


def rms_margin_ratio() -> float:
    return float(CFG["rms"]["min_margin_ratio"])


def contribution_margin(sku: str, price: float) -> float:
    c = CFG["catalog"][sku]
    return price - c["cost"] - c["housekeeping"]


# ------------------------------------------------------------------------------------------------ quant
class QuantEngine:
    """Loads the trained GPU models once and serves candidate rankings for Agent 1."""
    def __init__(self):
        from models import elasticity_net as en, uplift_models as um, contextual_bandit as cb
        self.en, self.um, self.cb = en, um, cb
        self.elasticity = en.load(); self.dragons, self.xlearners = um.load_uplift(); self.linucb = cb.load_bandit()
        sp = path("metrics") / "shadow_prices.json"
        self.shadow = json.loads(sp.read_text()) if sp.exists() else {"lam_budget": 0.0, "lam_inv": {k: 0.0 for k in CATS}}

    def rank(self, guest: GuestProfile, exclude: set[str] | None = None, min_price: dict | None = None) -> list[dict]:
        import torch
        exclude = exclude or set(); df = guest_frame(guest); X = featurize(df); out = []
        with timed("agent1_quant_inference", len(CATS)) as tm:
            Xt = torch.as_tensor(X, device=get_device())
            A_inv = self.linucb.Ainv; alpha = self.linucb.alpha
            for j, k in enumerate(CATS):
                if not eligible(df, k)[0]:
                    continue
                pl = float(list_price(df, k)[0]); c = unit_margin(k)
                floor = max(rms_floor(k), (min_price or {}).get(k, 0.0))
                price, P, _ = self.en.optimal_price(self.elasticity, X, np.array([j]), np.array([pl]), c, floor)
                price, P = float(price[0]), float(P[0])
                if price <= 0:
                    continue
                tau = float(self.um.dragon_cate(self.dragons[k], X)[0]); tau_x = float(self.xlearners[k].predict_cate(X)[0])
                mi = int(np.argmin(np.abs(PRICE_MULTS - price / pl))); arm = 1 + j * len(PRICE_MULTS) + mi
                bonus = float(100 * alpha * torch.sqrt((Xt @ A_inv[arm] * Xt).sum(1).clamp_min(0))[0]) if Xt.shape[1] == A_inv.shape[-1] else 0.0
                disc = max(pl - price, 0.0); lam = self.shadow["lam_inv"].get(k, 0.0); h = lam + self.shadow["lam_budget"] * disc
                margin = price - c
                out.append({"sku": k, "price": round(price, 2), "list_price": round(pl, 2), "p_accept": round(P, 4), "cate_dragon": round(tau, 4),
                            "cate_xlearner": round(tau_x, 4), "unit_margin": round(margin, 2), "ucb_bonus": round(bonus, 2), "hurdle": round(h, 2),
                            "lam_inv": round(lam, 2), "lam_budget": round(float(self.shadow["lam_budget"]), 3), "discount": round(disc, 2),
                            "score": round(tau * margin + bonus - h, 3), "excluded": k in exclude})
        for r in out:
            r["gpu_latency_ms"] = tm["latency_ms"]
        out.sort(key=lambda r: (r["excluded"], -r["score"]))
        return out


_engine: QuantEngine | None = None


def get_engine() -> QuantEngine:
    global _engine
    if _engine is None:
        _engine = QuantEngine()
    return _engine


# ------------------------------------------------------------------------------------------------ pre-loaded scenarios
def _g(rid, first, seg, ch, room, tier, **kw):
    base = dict(los=3, lead_time=4, adr=300.0, party_size=2, past_spend=900.0, n_prior_stays=3, contacts_7d=0, is_weekend=1)
    base.update(kw)
    return GuestProfile(reservation_id=rid, first_name=first, email=f"{first.lower()}@example.com", segment=seg, channel=ch, room_type=room,
                        loyalty_tier=tier, existing_tier_perks=loyalty_perks(tier), **base)


SCENARIOS: dict[str, dict] = {
    "clean_approval": {
        "title": "Clean approval - Gold honeymooner, healthy inventory",
        "description": "Tier-aware offer, price above floor, refined copy: should pass on the first draft.",
        "guest": _g("RES-90001", "Amelia", "Honeymoon", "Direct_Web", "DELUXE_KING", "Gold", adr=280.0, n_prior_stays=4, past_spend=1400.0),
        "injections": {"naive_bundle": False}, "pms_override": {}},
    "vip_duplicate_perk": {
        "title": "Diamond VIP - duplicate loyalty perk + HITL sign-off",
        "description": "Strategist's tier-blind sweetener bundles complimentary 2 PM late checkout for a Diamond member who already gets it free -> rejected, revised, then high-value VIP escalation.",
        "guest": _g("RES-90002", "Sofia", "VIP_Loyalty", "Direct_Web", "JUNIOR_OCEAN_VIEW", "Diamond", adr=380.0, n_prior_stays=14, past_spend=2800.0, party_size=2),
        "injections": {"naive_bundle": True}, "pms_override": {}},
    "last_room_upgrade": {
        "title": "Last-room protection - upgrade blocked by PMS buffer",
        "description": "Honeymoon guest in DELUXE_KING; only 1 JUNIOR_OCEAN_VIEW left (< dynamic safety buffer). Room upgrade must be rejected, so the strategist pivots SKU.",
        "guest": _g("RES-90003", "Noah", "Honeymoon", "Direct_Web", "DELUXE_KING", "Silver", adr=270.0, n_prior_stays=2, past_spend=700.0),
        "injections": {"naive_bundle": False, "prefer_sku": "ROOM_UPGRADE"}, "pms_override": {"JUNIOR_OCEAN_VIEW": 1}},
    "below_floor_price": {
        "title": "ADR floor breach - over-discounted draft",
        "description": "Stress test: generator over-discounts the first draft below the RMS floor; the auditor demands a floor-compliant price.",
        "guest": _g("RES-90004", "Liam", "Leisure", "OTA_Expedia", "STANDARD_QUEEN", "None", adr=190.0, n_prior_stays=0, past_spend=150.0),
        "injections": {"naive_bundle": False, "price_mult_rev0": 0.6, "prefer_sku": "SPA_PACKAGE"}, "pms_override": {}},
    "spammy_copy": {
        "title": "Brand tone breach - discount-spam copy",
        "description": "Stress test: first draft uses 'ACT NOW', '50% OFF' and 'CHEAP' language; the auditor rejects it and Agent 1 rewrites in a refined register.",
        "guest": _g("RES-90005", "Isabella", "Family", "Direct_Web", "STANDARD_QUEEN", "Silver", adr=210.0, party_size=4, n_prior_stays=2, past_spend=500.0),
        "injections": {"naive_bundle": False, "aggressive_rev_lt": 1}, "pms_override": {}},
    "unresolvable_escalation": {
        "title": "Unresolved after 3 revisions - automatic HITL",
        "description": "Stress test: a misbehaving generator keeps emitting off-brand copy on every revision; the bounded loop stops at R_max=3 and escalates to a human.",
        "guest": _g("RES-90006", "Ethan", "Leisure", "OTA_Expedia", "STANDARD_QUEEN", "None", adr=195.0, n_prior_stays=0, past_spend=120.0),
        "injections": {"naive_bundle": False, "aggressive_rev_lt": 99}, "pms_override": {}},
}


def trace(event: dict) -> None:
    jsonl_append("agent_trace.jsonl", event)
