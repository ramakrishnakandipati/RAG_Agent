"""Dataset validation: schema, nulls, ranges, referential integrity, duplicate keys and PSI drift -> validation/*.json."""
from __future__ import annotations
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import get_settings

SCHEMAS = {
    "guests": {"guest_id": "object", "first_name": "object", "last_name": "object", "segment": "object", "loyalty_tier": "object",
               "country": "object", "total_stays": "int", "ota_stays": "int", "direct_stays": "int", "ltv_usd": "float",
               "avg_adr": "float", "opt_in_marketing": "bool", "avg_sentiment": "float", "email": "object"},
    "reservations": {"reservation_id": "object", "guest_id": "object", "channel": "object", "channel_group": "object",
                     "check_in": "object", "check_out": "object", "nights": "int", "party_size": "int", "room_type": "object",
                     "adr": "float", "lead_days": "int", "room_revenue": "float", "status": "object"},
    "weather": {"date": "object", "temp_c": "float", "precip_prob": "float", "wind_kph": "float", "condition": "object"},
    "occupancy": {"date": "object", "rooms_total": "int", "rooms_sold": "int", "occupancy_pct": "float", "rms_adr": "float"},
    "inventory": {"date": "object", "sku": "object", "capacity": "int", "booked": "int"},
}
RANGES = {
    "guests": {"total_stays": (0, 200), "ltv_usd": (0, 1e6), "avg_adr": (1, 5000), "avg_sentiment": (-1, 1)},
    "reservations": {"nights": (1, 30), "party_size": (1, 20), "adr": (1, 5000), "lead_days": (0, 400)},
    "weather": {"precip_prob": (0, 1), "temp_c": (-30, 55), "wind_kph": (0, 200)},
    "occupancy": {"occupancy_pct": (0, 100), "rooms_sold": (0, 100000)},
    "inventory": {"capacity": (0, 1e6), "booked": (0, 1e6)},
}
ENUMS = {"guests": {"segment": ["Leisure", "Business", "Family", "Couple", "Group"], "loyalty_tier": ["Basic", "Silver", "Gold", "Platinum"]},
         "reservations": {"channel_group": ["OTA", "Direct"], "room_type": ["Standard", "Deluxe", "Suite"], "status": ["confirmed", "completed"]}}
PK = {"guests": "guest_id", "reservations": "reservation_id"}


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    qs = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    e = np.histogram(expected, qs)[0] / len(expected) + 1e-6
    a = np.histogram(np.clip(actual, qs[0], qs[-1]), qs)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def validate_all(data: dict[str, pd.DataFrame] | None = None) -> dict:
    s = get_settings()
    if data is None:
        data = {k: pd.read_csv(s.raw_dir / f"{k}.csv", keep_default_na=False, na_values=[""]) for k in
                [*SCHEMAS, "offer_history", "sku_catalog", "events"]}
    checks: list[dict] = []

    def add(table, name, ok, detail=""):
        checks.append({"table": table, "check": name, "passed": bool(ok), "detail": str(detail)})

    for t, schema in SCHEMAS.items():
        df = data[t]
        miss = [c for c in schema if c not in df.columns]
        add(t, "schema_columns_present", not miss, miss)
        for c, kind in schema.items():
            if c not in df.columns:
                continue
            ok = {"int": pd.api.types.is_integer_dtype, "float": pd.api.types.is_numeric_dtype,
                  "bool": lambda x: pd.api.types.is_bool_dtype(x), "object": lambda x: True}[kind](df[c])
            if not ok:
                add(t, f"dtype_{c}", False, f"expected {kind}, got {df[c].dtype}")
        nulls = df[[c for c in schema if c in df.columns]].isna().sum()
        add(t, "no_nulls", nulls.sum() == 0, nulls[nulls > 0].to_dict())
        for c, (lo, hi) in RANGES.get(t, {}).items():
            bad = int(((df[c] < lo) | (df[c] > hi)).sum())
            add(t, f"range_{c}[{lo},{hi}]", bad == 0, f"{bad} violations")
        for c, allowed in ENUMS.get(t, {}).items():
            bad = int((~df[c].isin(allowed)).sum())
            add(t, f"enum_{c}", bad == 0, f"{bad} invalid")
        if t in PK:
            add(t, f"unique_{PK[t]}", df[PK[t]].is_unique, "")
    g, r, inv, occ, w = (data[k] for k in ("guests", "reservations", "inventory", "occupancy", "weather"))
    add("reservations", "ref_guest_exists", r.guest_id.isin(g.guest_id).all(), f"{int((~r.guest_id.isin(g.guest_id)).sum())} orphans")
    add("reservations", "check_out_after_check_in", (pd.to_datetime(r.check_out) > pd.to_datetime(r.check_in)).all(), "")
    add("reservations", "nights_matches_dates", ((pd.to_datetime(r.check_out) - pd.to_datetime(r.check_in)).dt.days == r.nights).all(), "")
    add("guests", "stays_consistent", (g.ota_stays + g.direct_stays == g.total_stays).all(), "")
    add("inventory", "booked_le_capacity", (inv.booked <= inv.capacity).all(), "")
    add("inventory", "ref_sku_in_catalog", inv.sku.isin(data["sku_catalog"].sku).all(), "")
    add("occupancy", "sold_le_total", (occ.rooms_sold <= occ.rooms_total).all(), "")
    add("weather", "covers_reservation_dates", r.check_in.isin(w.date).all(), "")
    oh = data["offer_history"]
    add("offer_history", "ref_reservation_exists", oh.reservation_id.isin(r.reservation_id).all(), "")
    add("offer_history", "label_binary", oh.accepted.isin([0, 1]).all(), "")
    add("offer_history", "final_price_consistent", ((oh.price * (1 - oh.discount_pct / 100)).round(2) - oh.final_price).abs().lt(0.011).all(), "")
    # drift: completed (training) vs confirmed (upcoming) reservations
    hist, up = r[r.status == "completed"], r[r.status == "confirmed"]
    drift = {c: round(psi(hist[c], up[c]), 4) for c in ("adr", "lead_days", "nights", "party_size")}
    for c, v in drift.items():
        add("reservations", f"psi_{c}<0.25", v < 0.25, f"psi={v}")
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "n_checks": len(checks),
              "n_passed": sum(c["passed"] for c in checks), "all_passed": all(c["passed"] for c in checks),
              "psi": drift, "row_counts": {k: len(v) for k, v in data.items()}, "checks": checks}
    (s.validation_dir / "data_validation_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = validate_all()
    print(f"{rep['n_passed']}/{rep['n_checks']} checks passed")
    for c in rep["checks"]:
        if not c["passed"]:
            print("FAIL", c)
