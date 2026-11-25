"""Step 0 / Gate D1 - ingestion schema & integrity validation (pandas only; no extra deps)."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings

REQUIRED = {"booking_id": "object", "ota": "object", "masked_email": "object", "first_name": "object", "last_name": "object",
            "los": "int", "adr": "float", "ota_commission": "float", "lead_days": "int", "is_ghost_holdout": "bool"}
KINDS = {"int": "iu", "float": "iuf", "bool": "b", "object": "OUS"}


def _chk(name, ok, detail):
    return {"check": name, "passed": bool(ok), "detail": detail}


def gate_d1(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> dict:
    s = get_settings(); checks = []
    missing = [c for c in REQUIRED if c not in df.columns]
    checks.append(_chk("required_columns", not missing, {"missing": missing}))
    if not missing:
        bad = [c for c, t in REQUIRED.items() if df[c].dtype.kind not in KINDS[t]]
        checks.append(_chk("dtypes", not bad, {"bad": bad}))
        checks.append(_chk("pk_non_null_unique", df.booking_id.notna().all() and df.booking_id.is_unique,
                           {"dupes": int(df.booking_id.duplicated().sum())}))
        nn = df[["ota", "masked_email", "first_name", "last_name", "adr", "los", "ota_commission"]].isna().sum().to_dict()
        checks.append(_chk("no_nulls_critical", sum(nn.values()) == 0, nn))
        checks.append(_chk("ota_values", df.ota.isin(["Expedia", "Booking.com", "Agoda"]).all(), {"values": sorted(df.ota.unique().tolist())}))
        checks.append(_chk("commission_range_0.10_0.30", df.ota_commission.between(0.10, 0.30).all(),
                           {"min": float(df.ota_commission.min()), "max": float(df.ota_commission.max())}))
        checks.append(_chk("adr_positive", (df.adr > 0).all(), {"min": float(df.adr.min())}))
        checks.append(_chk("los_ge_1", (df.los >= 1).all(), {"min": int(df.los.min())}))
        frac = float(df.is_ghost_holdout.mean())
        checks.append(_chk("ghost_holdout_15pct_pm1", abs(frac - s.HOLDOUT_FRAC) <= 0.01, {"holdout_frac": round(frac, 4)}))
        strata = df.groupby("ota").is_ghost_holdout.mean().round(4).to_dict()
        checks.append(_chk("holdout_stratified_by_ota", all(abs(v - s.HOLDOUT_FRAC) <= 0.01 for v in strata.values()), strata))
        if ref is not None:
            d = abs(df.adr.mean() / ref.adr.mean() - 1)
            checks.append(_chk("drift_adr_mean_lt_20pct", d < 0.20, {"rel_diff": round(float(d), 4)}))
    return {"gate": "D1_ingestion_schema_integrity", "passed": all(c["passed"] for c in checks), "rows": int(len(df)),
            "ts": datetime.now(timezone.utc).isoformat(), "checks": checks}


def update_report(section: str, rep: dict) -> Path:
    p = get_settings().path("validation", "data_validation.json")
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[section] = rep
    cur["all_passed"] = all(v.get("passed", True) for k, v in cur.items() if isinstance(v, dict))
    p.write_text(json.dumps(cur, indent=2, default=str))
    return p


if __name__ == "__main__":
    from data.synthetic_generator import load
    b, _ = load(); r = gate_d1(b); update_report("D1", r)
    print(r["passed"], [c["check"] for c in r["checks"] if not c["passed"]])
