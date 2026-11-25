"""Deterministic tools: CDP lookup/identity, scoring service, CRS inventory, gated links, holdout assignment, dispatch, JSONL logs."""
from __future__ import annotations
import hashlib, hmac, json, threading
from datetime import datetime, timezone
from functools import lru_cache
import pandas as pd
from config.settings import get_settings

_SECRET = b"ota-to-direct-demo-signing-key"
_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonl_append(name: str, rec: dict) -> None:
    p = get_settings().path("logs", name)
    with _lock, p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


@lru_cache
def cdp_store() -> dict:
    from data.synthetic_generator import load, generate, GEN_DIR
    from models import identity_resolution as ir
    if not (GEN_DIR / "cdp_profiles.csv").exists():
        generate()
    _, cdp = load()
    cdp = cdp.reset_index(drop=True)
    return {"cdp": cdp, "cache": ir.prepare(cdp), "by_id": cdp.set_index("profile_id").to_dict("index")}


def resolve_identity(wh: dict) -> tuple[dict | None, dict]:
    from models import identity_resolution as ir
    st = cdp_store()
    row = pd.DataFrame([{"first_name": wh["first_name"], "last_name": wh["last_name"], "masked_email": wh["masked_email"],
                         "phone_last4": wh.get("phone_last4"), "country": wh.get("country", "US")}])
    out, tel = ir.resolve(row, st["cdp"], cache=st["cache"])
    pid = out.resolved_profile_id.iloc[0]
    meta = {"method": out.resolution_method.iloc[0], "confidence": float(out.resolution_confidence.iloc[0]), "profile_id": pid, "ms": tel["t_total_s"] * 1000}
    if pid is None or (isinstance(pid, float) and pd.isna(pid)):
        return None, meta
    prof = dict(st["by_id"][pid]); prof["profile_id"] = pid
    return prof, meta


@lru_cache
def scoring_service():
    from models.uplift_pipeline import ScoringService
    return ScoringService()


def crs_inventory(room_type: str, checkin_date: str) -> dict:
    h = int(hashlib.sha256(f"{room_type}|{checkin_date}".encode()).hexdigest(), 16)
    avail = h % 12
    return {"room_type": room_type, "date": checkin_date, "rooms_available": int(avail), "upgrade_available": avail >= 3}


def assign_holdout(booking_id: str, frac: float | None = None) -> bool:
    frac = get_settings().HOLDOUT_FRAC if frac is None else frac
    return int(hashlib.sha256(("holdout|" + booking_id).encode()).hexdigest(), 16) % 10000 < int(frac * 10000)


def gated_link(booking_id: str, offer_id: str) -> str:
    sig = hmac.new(_SECRET, f"{booking_id}|{offer_id}".encode(), hashlib.sha256).hexdigest()[:16]
    return f"https://book.example-hotel.com/member?b={booking_id}&o={offer_id}&sig={sig}"


def verify_link(url: str) -> bool:
    try:
        q = dict(kv.split("=", 1) for kv in url.split("?", 1)[1].split("&"))
        return hmac.compare_digest(q["sig"], hmac.new(_SECRET, f"{q['b']}|{q['o']}".encode(), hashlib.sha256).hexdigest()[:16])
    except Exception:
        return False


def dispatch(channel: str, booking_id: str, copy: str, link: str, holdout: bool) -> dict:
    status = "suppressed_ghost_holdout" if holdout else "sent"
    rec = {"ts": now(), "booking_id": booking_id, "channel": channel, "status": status, "link": link, "copy": copy}
    jsonl_append("outbox.jsonl", rec)
    return {"status": status, "channel": channel}
