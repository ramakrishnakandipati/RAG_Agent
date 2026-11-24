"""Synthetic hospitality data: PMS reservations, CRM Guest360, RMS inventory/occupancy, weather/events,
SKU catalog, historical offer outcomes (NBA training data) and the property knowledge base.

`true_accept_logit()` is the hidden "guest behaviour" used (a) to label historical offers and (b) by the offline
replay as the acceptance simulator. XGBoost never sees the hidden guest random-effect, so AUC is realistic (<1).
"""
from __future__ import annotations
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 24)
HOTEL = "Azure Bay Resort & Spa"

SEGMENTS = ["Leisure", "Business", "Family", "Couple", "Group"]
TIERS = ["Basic", "Silver", "Gold", "Platinum"]
OTA_CHANNELS = ["Booking.com", "Expedia", "Agoda"]
ROOM_TYPES = {"Standard": 190, "Deluxe": 260, "Suite": 380}
ROOMS_TOTAL = 150

SKU_CATALOG = [
    {"sku": "UPG-OCEAN", "name": "Ocean View Upgrade", "category": "room_upgrade", "price": 95.0, "cost": 20.0,
     "capacity": 12, "service_date": "stay", "desc": "Upgrade to a Deluxe Ocean View room with balcony, subject to availability."},
    {"sku": "UPG-SUITE", "name": "Junior Suite Upgrade", "category": "room_upgrade", "price": 160.0, "cost": 45.0,
     "capacity": 6, "service_date": "stay", "desc": "Upgrade to a Junior Suite with separate lounge and soaking tub."},
    {"sku": "SPA-RELAX", "name": "Signature Relaxation Massage (60 min)", "category": "spa", "price": 140.0, "cost": 45.0,
     "capacity": 20, "service_date": "checkin", "desc": "60-minute signature massage at the Tidewater Spa."},
    {"sku": "SPA-COUPLE", "name": "Couples Spa Ritual", "category": "spa", "price": 290.0, "cost": 95.0,
     "capacity": 6, "service_date": "checkin", "desc": "Side-by-side massage and private hydrotherapy suite for two."},
    {"sku": "DIN-CHEF", "name": "Chef's Tasting Dinner for Two", "category": "dining", "price": 180.0, "cost": 70.0,
     "capacity": 16, "service_date": "checkin", "desc": "Five-course tasting menu at Salt & Ember with wine pairing."},
    {"sku": "DIN-BRKFST", "name": "Daily Breakfast Package", "category": "dining", "price": 60.0, "cost": 22.0,
     "capacity": 200, "service_date": "checkin", "desc": "Full breakfast buffet at the Terrace Cafe for every morning of your stay."},
    {"sku": "LCO-4PM", "name": "Late Checkout until 4 PM", "category": "late_checkout", "price": 50.0, "cost": 8.0,
     "capacity": 25, "service_date": "checkout", "desc": "Keep your room until 4 PM on departure day."},
    {"sku": "LOY-DIRECT", "name": "Azure Circle Membership", "category": "loyalty", "price": 0.0, "cost": 0.0,
     "capacity": 99999, "service_date": "checkin", "fixed_discount_pct": 10.0,
     "desc": "Free loyalty membership: 10% off your next direct booking plus member-only perks."},
]
CATEGORIES = ["room_upgrade", "spa", "dining", "late_checkout", "loyalty"]

FIRST = ["Ava", "Liam", "Noah", "Mia", "Ethan", "Sofia", "Lucas", "Isla", "Mateo", "Zara", "Omar", "Priya", "Chen",
         "Hana", "Diego", "Elena", "Kofi", "Nora", "Ravi", "Tara", "Jonas", "Leila", "Marco", "Yuki", "Amara"]
LAST = ["Smith", "Garcia", "Patel", "Nguyen", "Brown", "Kim", "Silva", "Rossi", "Khan", "Mueller", "Okafor", "Sato",
        "Dubois", "Novak", "Haddad", "Larsen", "Costa", "Ivanov", "Reyes", "Walker"]
EVENTS = [("Coastal Jazz Festival", 0.9), ("Harbour Marathon", 0.7), ("Food & Wine Weekend", 0.8),
          ("Regatta Week", 0.6), ("New Year Gala", 1.0)]


# ----------------------------------------------------------------------------------------------- behaviour model
def true_accept_logit(f: dict, guest_effect: float = 0.0) -> float:
    """Hidden ground-truth acceptance logit for an offer (f = feature dict, see nba_rms_engine.FEATURES)."""
    c = f["category"]
    base = {"room_upgrade": -2.3, "spa": -1.7, "dining": -1.5, "late_checkout": -1.9, "loyalty": -2.0}[c]
    z = base
    seg = f["segment"]
    seg_fx = {
        ("Couple", "spa"): 0.9, ("Couple", "dining"): 0.6, ("Couple", "room_upgrade"): 0.3,
        ("Family", "late_checkout"): 0.7, ("Family", "dining"): 0.4, ("Family", "spa"): -0.5,
        ("Business", "late_checkout"): 1.0, ("Business", "room_upgrade"): 0.3, ("Business", "spa"): -0.3,
        ("Leisure", "room_upgrade"): 0.3, ("Leisure", "spa"): 0.2, ("Group", "dining"): 0.3, ("Group", "spa"): -0.4,
    }
    z += seg_fx.get((seg, c), 0.0)
    z += {"Basic": 0.0, "Silver": 0.15, "Gold": 0.3, "Platinum": 0.55}[f["loyalty_tier"]]
    if c in ("spa", "dining"):
        z += 1.3 * (f["precip_prob"] - 0.3)
    if c == "late_checkout":
        z += 0.3 * f["has_event"]
    z += 0.7 * math.log(max(f["adr"], 50) / 230.0) * (1 if c in ("room_upgrade", "spa", "dining") else 0.3)
    z += 0.5 * math.log1p(max(f["ltv_usd"], 0) / 2500.0) - 0.2
    if c == "room_upgrade" and f["lead_days"] < 7:
        z += 0.35
    if c == "loyalty":
        z += 0.8 * f["is_ota"] - 0.6 * f["is_direct"]
        z += 0.2 * f["total_stays"] / 5
    z += 0.05 * f["discount_pct"] if c != "loyalty" else 0.0
    z -= 0.0035 * (f["price"] - 100.0) if c != "loyalty" else 0.0
    z += 0.08 * (f["party_size"] - 2) * (1 if c in ("dining", "late_checkout") else 0)
    z -= 0.5 * max(0.0, f["occupancy_pct"] - 85) / 15 * (1 if c == "late_checkout" else 0)
    return z + guest_effect


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------------------------------------------------------------------------- generators
def _weather(dates: list[date], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in dates:
        doy = d.timetuple().tm_yday
        temp = 22 + 7 * math.cos((doy - 200) / 365 * 2 * math.pi) + rng.normal(0, 2.2)
        precip = float(np.clip(0.28 + 0.15 * math.cos((doy - 330) / 365 * 2 * math.pi) + rng.normal(0, 0.18), 0.0, 1.0))
        cond = "rain" if precip > 0.6 else ("cloudy" if precip > 0.35 else "sunny")
        rows.append({"date": d.isoformat(), "temp_c": round(temp, 1), "precip_prob": round(precip, 3),
                     "wind_kph": round(abs(rng.normal(14, 6)), 1), "condition": cond})
    return pd.DataFrame(rows)


def _events(dates: list[date], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in dates:
        if d.month == 12 and d.day in (30, 31):
            rows.append({"date": d.isoformat(), "event_name": "New Year Gala", "demand_boost": 1.0})
        elif rng.random() < 0.035:
            name, boost = EVENTS[int(rng.integers(0, len(EVENTS) - 1))]
            rows.append({"date": d.isoformat(), "event_name": name, "demand_boost": boost})
    return pd.DataFrame(rows, columns=["date", "event_name", "demand_boost"])


def generate_all(n_guests: int = 6000, n_res: int = 18000, seed: int = 42, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir or ROOT / "data" / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- guests (CRM Guest360)
    seg = rng.choice(SEGMENTS, n_guests, p=[0.32, 0.2, 0.18, 0.22, 0.08])
    tier = rng.choice(TIERS, n_guests, p=[0.55, 0.22, 0.15, 0.08])
    total_stays = rng.poisson(np.where(tier == "Platinum", 7, np.where(tier == "Gold", 4, np.where(tier == "Silver", 2.5, 1.2)))) + 1
    ota_ratio = np.clip(rng.beta(4, 3, n_guests) - np.where(tier == "Platinum", 0.35, np.where(tier == "Gold", 0.2, 0.0)), 0, 1)
    ota_stays = np.round(total_stays * ota_ratio).astype(int)
    direct_stays = total_stays - ota_stays
    avg_adr = np.round(rng.lognormal(math.log(230), 0.28, n_guests), 2)
    ltv = np.round(total_stays * avg_adr * rng.uniform(2.0, 3.2, n_guests) * (1 + rng.uniform(0, 0.4, n_guests)), 2)
    guests = pd.DataFrame({
        "guest_id": [f"G{100000 + i}" for i in range(n_guests)],
        "first_name": rng.choice(FIRST, n_guests), "last_name": rng.choice(LAST, n_guests),
        "segment": seg, "loyalty_tier": tier,
        "country": rng.choice(["US", "UK", "DE", "FR", "IN", "AU", "AE", "JP"], n_guests, p=[.4, .12, .08, .07, .1, .08, .07, .08]),
        "total_stays": total_stays, "ota_stays": ota_stays, "direct_stays": direct_stays,
        "ltv_usd": ltv, "avg_adr": avg_adr,
        "opt_in_marketing": rng.random(n_guests) < 0.9,
        "avg_sentiment": np.round(np.clip(rng.normal(0.35, 0.35, n_guests), -1, 1), 3),
    })
    guests["email"] = [f"{f.lower()}.{l.lower()}{i}@example.com" for i, (f, l) in enumerate(zip(guests.first_name, guests.last_name))]

    # ---- reservations (PMS)
    start, end = TODAY - timedelta(days=365), TODAY + timedelta(days=95)
    span = (end - start).days
    gidx = rng.integers(0, n_guests, n_res)
    ci_off = rng.integers(0, span, n_res)
    nights = np.clip(rng.poisson(2.6, n_res) + 1, 1, 10)
    check_in = [start + timedelta(days=int(o)) for o in ci_off]
    check_out = [c + timedelta(days=int(n)) for c, n in zip(check_in, nights)]
    lead = np.clip(rng.gamma(2.0, 22, n_res).astype(int), 0, 240)
    g = guests.iloc[gidx].reset_index(drop=True)
    p_ota = np.clip(0.62 - 0.25 * (g.loyalty_tier.isin(["Gold", "Platinum"])) - 0.1 * (g.loyalty_tier == "Silver")
                    + 0.05 * (g.segment == "Leisure") - 0.1 * (g.segment == "Business"), 0.05, 0.95)
    is_ota = rng.random(n_res) < p_ota
    channel = np.where(is_ota, rng.choice(OTA_CHANNELS, n_res, p=[0.5, 0.35, 0.15]), "Direct")
    room = rng.choice(list(ROOM_TYPES), n_res, p=[0.5, 0.33, 0.17])
    adr = np.round(np.array([ROOM_TYPES[r] for r in room]) * rng.lognormal(0, 0.12, n_res), 2)
    party = np.where(g.segment == "Family", rng.integers(3, 6, n_res),
                     np.where(g.segment == "Couple", 2, np.where(g.segment == "Group", rng.integers(4, 9, n_res),
                                                                 rng.integers(1, 3, n_res))))
    reservations = pd.DataFrame({
        "reservation_id": [f"R{500000 + i}" for i in range(n_res)], "guest_id": g.guest_id.values,
        "channel": channel, "channel_group": np.where(is_ota, "OTA", "Direct"),
        "check_in": [c.isoformat() for c in check_in], "check_out": [c.isoformat() for c in check_out],
        "nights": nights, "party_size": party, "room_type": room, "adr": adr, "lead_days": lead,
        "room_revenue": np.round(adr * nights, 2),
        "status": ["completed" if c < TODAY else "confirmed" for c in check_in],
    })

    # ---- weather / events / occupancy
    all_dates = [start + timedelta(days=i) for i in range(span + 12)]
    weather = _weather(all_dates, rng)
    events = _events(all_dates, rng)
    ev_map = dict(zip(events.date, events.demand_boost))
    sold = {}
    for ci, co in zip(check_in, check_out):
        d = ci
        while d < co:
            sold[d.isoformat()] = sold.get(d.isoformat(), 0) + 1
            d += timedelta(days=1)
    # scale up: reservations sample is a slice of total demand -> add base load so occupancy is realistic
    occ_rows = []
    for d in all_dates:
        k = d.isoformat()
        dow = 1.12 if d.weekday() in (4, 5) else 1.0
        base = 0.55 + 0.1 * math.cos((d.timetuple().tm_yday - 200) / 365 * 2 * math.pi) + 0.15 * ev_map.get(k, 0)
        rooms_sold = int(np.clip(ROOMS_TOTAL * base * dow + sold.get(k, 0) * 0.6 + rng.normal(0, 5), 20, ROOMS_TOTAL))
        occ_rows.append({"date": k, "rooms_total": ROOMS_TOTAL, "rooms_sold": rooms_sold,
                         "occupancy_pct": round(100 * rooms_sold / ROOMS_TOTAL, 1),
                         "rms_adr": round(210 * (0.8 + 0.5 * rooms_sold / ROOMS_TOTAL), 2)})
    occupancy = pd.DataFrame(occ_rows)
    occ_map = dict(zip(occupancy.date, occupancy.occupancy_pct))

    # ---- ancillary inventory (RMS) per sku/date
    inv_rows = []
    for d in all_dates:
        k = d.isoformat()
        for s in SKU_CATALOG:
            cap = s["capacity"]
            fill = float(np.clip(occ_map[k] / 100 * rng.uniform(0.55, 1.15), 0, 1)) if s["sku"] != "LOY-DIRECT" else 0.0
            inv_rows.append({"date": k, "sku": s["sku"], "capacity": cap, "booked": int(cap * fill)})
    inventory = pd.DataFrame(inv_rows)

    # ---- historical offers -> NBA training data (exploratory random policy)
    wmap = weather.set_index("date")
    emap = set(events.date)
    hist = reservations[reservations.status == "completed"].reset_index(drop=True)
    gi = guests.set_index("guest_id")
    sku_df = pd.DataFrame(SKU_CATALOG)
    rows = []
    guest_fx = {gid: rng.normal(0, 0.3) for gid in guests.guest_id}
    for r in hist.itertuples():
        gr = gi.loc[r.guest_id]
        for _ in range(int(rng.integers(1, 3))):
            s = sku_df.iloc[int(rng.integers(0, len(sku_df)))]
            if s.category == "loyalty" and r.channel_group != "OTA" and rng.random() < 0.85:
                continue
            disc = 0.0 if s.category == "loyalty" else float(rng.choice([0, 5, 10, 15, 20]))
            f = build_feature_row(gr, r._asdict(), s.to_dict(), wmap.loc[r.check_in], int(r.check_in in emap),
                                  occ_map.get(r.check_in, 60.0), disc)
            p = sigmoid(true_accept_logit(f, guest_fx[r.guest_id]))
            f.update({"reservation_id": r.reservation_id, "guest_id": r.guest_id, "sku": s.sku,
                      "accepted": int(rng.random() < p)})
            rows.append(f)
    offers = pd.DataFrame(rows)

    data = {"guests": guests, "reservations": reservations, "weather": weather, "events": events,
            "occupancy": occupancy, "inventory": inventory, "sku_catalog": sku_df, "offer_history": offers}
    for name, df in data.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    guests.to_parquet(out_dir / "guests.parquet", index=False)
    reservations.to_parquet(out_dir / "reservations.parquet", index=False)
    offers.to_parquet(out_dir / "offer_history.parquet", index=False)
    (out_dir / "guest_effects.json").write_text(json.dumps({k: round(v, 4) for k, v in guest_fx.items()}))
    write_knowledge_base()
    return data


def build_feature_row(guest, res: dict, sku: dict, wx, has_event: int, occ_pct: float, discount_pct: float) -> dict:
    """Feature dict shared by training (generator), scoring (RMS engine) and replay."""
    price = float(sku["price"])
    return {
        "category": sku["category"], "segment": guest["segment"], "loyalty_tier": guest["loyalty_tier"],
        "channel_group": res["channel_group"], "room_type": res["room_type"],
        "is_ota": int(res["channel_group"] == "OTA"), "is_direct": int(res["channel_group"] == "Direct"),
        "adr": float(res["adr"]), "nights": int(res["nights"]), "party_size": int(res["party_size"]),
        "lead_days": int(res["lead_days"]), "ltv_usd": float(guest["ltv_usd"]), "total_stays": int(guest["total_stays"]),
        "ota_ratio": float(guest["ota_stays"]) / max(int(guest["total_stays"]), 1),
        "precip_prob": float(wx["precip_prob"]), "temp_c": float(wx["temp_c"]), "has_event": int(has_event),
        "occupancy_pct": float(occ_pct), "price": price, "discount_pct": float(discount_pct),
        "final_price": round(price * (1 - discount_pct / 100), 2),
    }


# ----------------------------------------------------------------------------------------------- knowledge base
def write_knowledge_base(kb: Path | None = None) -> None:
    kb = Path(kb or ROOT / "data" / "knowledge_base")
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "sku_catalog.json").write_text(json.dumps(SKU_CATALOG, indent=2))
    docs = {
        "spa_menu.md": f"""# {HOTEL} - Tidewater Spa
doc_type: spa
The Tidewater Spa is open daily 9 AM to 8 PM. Treatments must be booked at least 4 hours ahead.
Signature Relaxation Massage (60 min) uses warm sea-salt oil and is our most requested treatment.
The Couples Spa Ritual includes a private hydrotherapy suite and herbal tea service for two.
Spa guests receive complimentary access to the thermal pools on the day of treatment.
Rain-day tip: the spa and thermal pools are fully indoors, a favourite when the weather turns.
""",
        "dining_menu.md": f"""# {HOTEL} - Dining
doc_type: dining
Salt & Ember (fine dining) serves dinner 6 PM to 10 PM, reservations recommended.
The Chef's Tasting Dinner is a five-course menu with optional wine pairing; vegetarian menu on request.
The Terrace Cafe serves a breakfast buffet 6:30 AM to 10:30 AM with ocean views.
Dietary requirements (gluten-free, vegan, nut allergy) can be accommodated with 24 hours notice.
""",
        "room_tiers.md": f"""# {HOTEL} - Room Tiers
doc_type: rooms
Standard rooms (garden view, 28 sqm), Deluxe rooms (ocean view with balcony, 34 sqm), Suites (60 sqm with lounge and soaking tub).
Upgrades are subject to availability on the night of arrival and confirmed at the time of booking.
Ocean View Upgrade moves a Standard room to Deluxe. Junior Suite Upgrade moves a Standard or Deluxe room to a Suite.
""",
        "policies.md": f"""# {HOTEL} - Policies
doc_type: policy
Standard check-in is 3 PM and check-out is 11 AM. Late checkout until 4 PM is offered as a paid add-on subject to availability.
Ancillary purchases are charged to the room folio and are refundable up to 24 hours before arrival.
Discounts cannot be combined. Offers are valid only for the dates and prices stated in the message.
Guests may opt out of marketing messages at any time by replying STOP.
Loyalty: Azure Circle membership is free. Members receive 10% off future direct bookings, priority upgrades and late checkout when available.
Booking direct avoids third-party fees and unlocks Azure Circle benefits.
""",
        "brand_voice.md": f"""# {HOTEL} - Brand Voice Guidelines
doc_type: brand
Tone: warm, relaxed, coastal, concise. Address the guest by first name. Two short paragraphs maximum.
Do: mention the experience (sea air, calm, indulgence); give one clear call to action; sign off as "The Azure Bay Team".
Do not: use pressure language (hurry, last chance, act now), ALL CAPS, exclamation-mark clusters, or the words guarantee, free (unless the offer is a free membership), cheapest, best price.
Never write any price, percentage, SKU code or date yourself; use the provided placeholders only.
""",
    }
    for name, body in docs.items():
        (kb / name).write_text(body, encoding="utf-8")


if __name__ == "__main__":
    d = generate_all()
    for k, v in d.items():
        print(f"{k:16s} {len(v):>7d} rows  cols={len(v.columns)}")
    print("accept rate:", round(d["offer_history"].accepted.mean(), 3))
