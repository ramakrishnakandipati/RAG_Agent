"""Deterministic compliance gate (price / discount / SKU / policy / caps) and dispatch + audit logging."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Optional

from src.config import get_settings
from src.schemas.domain import GuardrailResult
from src.tools import pms_crm_tools as pms
from src.utils.logger import get_logger

log = get_logger("compliance")

PRICE_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s?%")
SKU_RE = re.compile(r"\b[A-Z]{3}-[A-Z0-9]{2,12}\b")
PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")
BANNED = ["guarantee", "cheapest", "best price", "hurry", "last chance", "act now", "limited time only", "don't miss out",
          "100% free", "no strings"]


def fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def build_facts(state: dict) -> dict[str, str]:
    g, st, o = state["guest"], state["stay"], state["selected_offer"]
    disc = o["discount_pct"]
    return {
        "GUEST_FIRST_NAME": g["first_name"], "HOTEL": "Azure Bay Resort & Spa",
        "CHECK_IN": str(st["check_in"]), "CHECK_OUT": str(st["check_out"]),
        "OFFER_NAME": o["name"], "OFFER_PRICE": fmt_money(o["final_price"]), "OFFER_LIST_PRICE": fmt_money(o["list_price"]),
        "OFFER_DISCOUNT": f"{disc:g}%" if disc > 0 else "", "OFFER_DATE": str(o["service_date"]),
    }


def render(template: str, facts: dict[str, str]) -> str:
    out = template
    for k, v in facts.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def allowed_tokens(offer: dict) -> tuple[set[float], set[float], set[str]]:
    prices = {round(offer["final_price"], 2), round(offer["list_price"], 2)}
    pcts = {float(offer["discount_pct"])} if offer["discount_pct"] > 0 else set()
    return prices, pcts, {offer["sku"]}


def check_text_tokens(text: str, offer: dict) -> list[str]:
    """Core 0%-hallucination gate: every $, % and SKU token in the final text must come from the RMS offer payload."""
    prices, pcts, skus = allowed_tokens(offer)
    v = []
    for m in PRICE_RE.findall(text):
        if not any(abs(float(m.replace(",", "")) - p) < 0.005 for p in prices):
            v.append(f"price_hallucination:${m}")
    for m in PCT_RE.findall(text):
        if not any(abs(float(m) - p) < 0.005 for p in pcts):
            v.append(f"discount_hallucination:{m}%")
    for m in SKU_RE.findall(text):
        if m not in skus:
            v.append(f"sku_hallucination:{m}")
    return v


def verify_compliance_and_caps(state: dict) -> dict:
    """Return GuardrailResult dict. FAIL_POLICY -> redraft/retry; FAIL_VALIDATION -> hard stop."""
    s = get_settings()
    g, o, d = state["guest"], state["selected_offer"], state["draft"]
    text = f"{d.get('subject', '')}\n{d['body']}"
    checks: dict[str, bool] = {}
    viol: list[str] = []

    tok = check_text_tokens(text, o)
    checks["no_price_discount_sku_hallucination"] = not tok
    viol += tok

    unresolved = PLACEHOLDER_RE.findall(text)
    checks["placeholders_resolved"] = not unresolved
    viol += [f"unresolved_placeholder:{u}" for u in unresolved]

    low = text.lower()
    bad = [b for b in BANNED if b in low]
    if "free" in low and o["category"] != "loyalty":
        bad.append("free")
    checks["brand_tone_banned_phrases"] = not bad
    viol += [f"banned_phrase:{b}" for b in bad]
    shout = [w for w in re.findall(r"\b[A-Z]{5,}\b", text) if w not in ("SPA", "LATE")]
    checks["no_all_caps"] = not shout
    viol += [f"all_caps:{w}" for w in shout]
    checks["length_ok"] = len(d["body"]) <= 1200
    if not checks["length_ok"]:
        viol.append("body_too_long")

    fixed = o["category"] == "loyalty"
    checks["discount_within_cap"] = fixed or o["discount_pct"] <= s.max_discount_pct
    checks["margin_above_floor"] = fixed or o["margin_pct"] >= s.min_margin_pct
    checks["offer_price_consistent"] = abs(o["final_price"] - round(o["list_price"] * (1 - o["discount_pct"] / 100), 2)) < 0.011
    for k in ("discount_within_cap", "margin_above_floor", "offer_price_consistent"):
        if not checks[k]:
            viol.append(f"rms_rule_violation:{k}")

    hard = False
    inbound = state.get("trigger_type") == "inbound_reply"
    checks["opt_in"] = inbound or bool(g["opt_in_marketing"])
    if not checks["opt_in"]:
        viol.append("guest_not_opted_in"); hard = True
    if not inbound:
        recent = pms.recent_message_count(g["guest_id"])
        checks["frequency_cap"] = recent < s.max_msgs_per_7d
        if not checks["frequency_cap"]:
            viol.append(f"frequency_cap_exceeded:{recent}"); hard = True
        hr = state.get("now_hour", 10)
        quiet = hr >= s.quiet_hours_start or hr < s.quiet_hours_end
        checks["quiet_hours"] = not quiet
        if quiet:
            viol.append(f"quiet_hours:{hr}"); hard = True
    action = d.get("action", "offer")
    if action in ("offer", "confirm"):
        avail = pms.get_availability(o["sku"], str(o["service_date"]))
        checks["inventory_available"] = avail > 0 or o["category"] == "loyalty"
        if not checks["inventory_available"]:
            viol.append("inventory_unavailable"); hard = True

    status = "PASS" if not viol else ("FAIL_VALIDATION" if hard else "FAIL_POLICY")
    return GuardrailResult(status=status, violations=viol, checks=checks).model_dump()


def dispatch_and_log(state: dict) -> dict:
    """Send (outbox) + audit; book to folio on confirmed inbound acceptance."""
    g, st, o, d = state["guest"], state["stay"], state["selected_offer"], state["draft"]
    tid = state["thread_id"]
    final_status, booking = "SENT", None
    if d.get("action") == "confirm":
        booking = pms.book_ancillary_item(st["reservation_id"], o["sku"], o["final_price"], str(o["service_date"]))
        if booking["ok"]:
            final_status = "BOOKED"
            pms.write_audit(tid, st["reservation_id"], g["guest_id"], "folio_charge", "OK", o["sku"], o["final_price"], booking)
        else:
            final_status = "SOLD_OUT"
            d = dict(d, subject="About your request", body=(
                f"Hi {g['first_name']}, thank you for your reply. Unfortunately {o['name']} has just sold out for your stay, "
                "so we have not added it to your reservation. Our team would be glad to suggest an alternative on arrival. "
                "The Azure Bay Team"))
    pms.record_message(tid, st["reservation_id"], g["guest_id"], d["subject"], d["body"], o["sku"],
                       status="SENT", channel="email")
    pms.write_audit(tid, st["reservation_id"], g["guest_id"], "message_sent", final_status, o["sku"], o["final_price"],
                    {"action": d.get("action"), "source": d.get("source"), "retries": state.get("retries", 0)})
    log.info("dispatched", thread=tid, status=final_status, sku=o["sku"])
    return {"status": final_status, "booking": booking, "dispatched": True, "draft": d}
