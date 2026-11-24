"""Next-Best-Action: GPU XGBoost propensity model + deterministic RMS margin / inventory / discount rules."""
from __future__ import annotations
import json
import warnings
from datetime import date, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import get_settings
from src.tools import pms_crm_tools as pms
from src.utils import device as dev
from src.utils.logger import get_logger

log = get_logger("nba")
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

CAT_COLS = ["category", "segment", "loyalty_tier", "channel_group", "room_type"]
NUM_COLS = ["adr", "nights", "party_size", "lead_days", "ltv_usd", "total_stays", "ota_ratio", "precip_prob", "temp_c",
            "has_event", "occupancy_pct", "price", "discount_pct"]
FEATURES = CAT_COLS + NUM_COLS
CAT_LEVELS = {
    "category": ["room_upgrade", "spa", "dining", "late_checkout", "loyalty"],
    "segment": ["Leisure", "Business", "Family", "Couple", "Group"],
    "loyalty_tier": ["Basic", "Silver", "Gold", "Platinum"],
    "channel_group": ["OTA", "Direct"],
    "room_type": ["Standard", "Deluxe", "Suite"],
}


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


# --------------------------------------------------------------------------------------------------- training
def train_model(df: Optional[pd.DataFrame] = None, seed: int = 42) -> dict:
    """Train on GPU (tree_method=hist, device=cuda; CPU fallback). Returns metrics + saves model & holdout preds."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    s = get_settings()
    df = df if df is not None else pd.read_parquet(s.raw_dir / "offer_history.parquet")
    tr, te = train_test_split(df, test_size=0.25, random_state=seed, stratify=df["accepted"])
    tr, va = train_test_split(tr, test_size=0.15, random_state=seed, stratify=tr["accepted"])
    params = dict(n_estimators=600, learning_rate=0.04, max_depth=4, subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=3, reg_lambda=2.0, objective="binary:logistic", eval_metric="auc",
                  enable_categorical=True, early_stopping_rounds=40, random_state=seed, **dev.xgb_device_params())
    model = xgb.XGBClassifier(**params)
    model.fit(to_matrix(tr), tr["accepted"], eval_set=[(to_matrix(va), va["accepted"])], verbose=False)
    p = model.predict_proba(to_matrix(te))[:, 1]
    holdout = te[["reservation_id", "guest_id", "sku", "category", "accepted", "final_price", "discount_pct"]].copy()
    holdout["p"] = p
    holdout.to_parquet(s.eval_dir / "holdout_predictions.parquet", index=False)
    joblib.dump(model, s.models_dir / "nba_xgb.joblib")
    info = {"device": params["device"], "tree_method": params["tree_method"], "best_iteration": int(model.best_iteration),
            "n_train": len(tr), "n_test": len(te), "auc": round(float(roc_auc_score(te["accepted"], p)), 4)}
    (s.eval_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    log.info("xgb_trained", **info)
    return info


_model = None


def get_model():
    global _model
    if _model is None:
        path = get_settings().models_dir / "nba_xgb.joblib"
        if not path.exists():
            train_model()
        _model = joblib.load(path)
        try:
            _model.set_params(**dev.xgb_device_params())
        except Exception:  # noqa: BLE001
            pass
    return _model


def score_rows(rows: list[dict]) -> np.ndarray:
    return get_model().predict_proba(to_matrix(pd.DataFrame(rows)))[:, 1]


# --------------------------------------------------------------------------------------------------- RMS rules
def load_catalog() -> list[dict]:
    return json.loads((get_settings().kb_dir / "sku_catalog.json").read_text())


def service_date(sku: dict, stay: dict) -> str:
    ci = date.fromisoformat(str(stay["check_in"]))
    co = date.fromisoformat(str(stay["check_out"]))
    return {"checkout": (co).isoformat()}.get(sku["service_date"], ci.isoformat())


def margin_pct(final_price: float, cost: float) -> float:
    return round(100 * (final_price - cost) / final_price, 2) if final_price > 0 else 100.0


def rms_discount(propensity: float, occupancy_pct: float, tier: str, cap: float) -> float:
    """Deterministic discount ladder: only discount when demand is soft AND propensity is borderline."""
    d = 0.0
    if occupancy_pct < 60:
        d = 15.0 if propensity < 0.35 else 10.0
    elif occupancy_pct < 75:
        d = 10.0 if propensity < 0.30 else 5.0
    elif occupancy_pct < 88:
        d = 5.0 if propensity < 0.22 else 0.0
    if tier == "Platinum" and d > 0:
        d += 5.0
    return float(min(d, cap))


def eligible(sku: dict, guest: dict, stay: dict, env: dict, avail: int) -> tuple[bool, str]:
    cat = sku["category"]
    if avail <= 0:
        return False, "sold_out"
    if cat == "loyalty" and (stay["channel_group"] != "OTA" or guest["loyalty_tier"] not in ("Basic", "Silver")):
        return False, "loyalty_only_for_ota_non_members"
    if cat == "room_upgrade" and sku["sku"] == "UPG-OCEAN" and stay["room_type"] != "Standard":
        return False, "already_ocean_or_better"
    if sku["sku"] == "UPG-SUITE" and stay["room_type"] == "Suite":
        return False, "already_suite"
    if cat == "room_upgrade" and env.get("occupancy_pct", 0) > 97:
        return False, "property_full"
    if cat == "late_checkout" and env.get("occupancy_pct", 0) > 92:
        return False, "high_occupancy_no_late_checkout"
    if sku["sku"] == "DIN-CHEF" and stay["party_size"] > 4:
        return False, "chef_table_max_party_4"
    if sku["sku"] == "SPA-COUPLE" and stay["party_size"] < 2:
        return False, "needs_two_guests"
    return True, "ok"


def generate_offers(guest: dict, stay: dict, env: dict, catalog: Optional[list[dict]] = None,
                    top_k: int = 3, threshold: Optional[float] = None) -> list[dict]:
    """Eligible SKUs -> XGBoost propensity -> RMS discount/margin -> expected-value ranking."""
    from data.synthetic_generator import build_feature_row
    s = get_settings()
    thr = s.propensity_threshold if threshold is None else threshold
    catalog = catalog or load_catalog()
    wx = {"precip_prob": env["precip_prob"], "temp_c": env["temp_c"]}
    base_rows, metas = [], []
    for sku in catalog:
        sdate = service_date(sku, stay)
        avail = pms.get_availability(sku["sku"], sdate)
        ok, why = eligible(sku, guest, stay, env, avail)
        if not ok:
            continue
        fixed = float(sku.get("fixed_discount_pct", 0.0))
        row = build_feature_row(guest, stay, sku, wx, env["has_event"], env["occupancy_pct"], 0.0)
        base_rows.append(row)
        metas.append((sku, sdate, avail, fixed))
    if not base_rows:
        return []
    p0 = score_rows(base_rows)
    final_rows = []
    for row, (sku, sdate, avail, fixed), p in zip(base_rows, metas, p0):
        d = fixed if sku["category"] == "loyalty" else rms_discount(float(p), env["occupancy_pct"], guest["loyalty_tier"], s.max_discount_pct)
        # margin floor: reduce discount until margin >= floor
        while d > 0 and sku["category"] != "loyalty" and margin_pct(sku["price"] * (1 - d / 100), sku["cost"]) < s.min_margin_pct:
            d -= 5.0
        final_rows.append(dict(row, discount_pct=float(max(d, 0.0)), final_price=round(sku["price"] * (1 - max(d, 0) / 100), 2)))
    p1 = score_rows(final_rows)
    offers = []
    for row, (sku, sdate, avail, fixed), p in zip(final_rows, metas, p1):
        fp = row["final_price"]
        if sku["category"] == "loyalty":   # value = avoided OTA commission on next direct stay (~18%)
            ev = float(p) * 0.18 * stay["adr"] * stay["nights"]
        else:
            ev = float(p) * (fp - sku["cost"])
        offers.append({
            "sku": sku["sku"], "name": sku["name"], "category": sku["category"], "list_price": sku["price"],
            "discount_pct": row["discount_pct"], "final_price": fp, "cost": sku["cost"],
            "margin_pct": margin_pct(fp, sku["cost"]), "propensity": round(float(p), 4), "expected_value": round(ev, 2),
            "available_units": int(avail), "service_date": sdate, "desc": sku["desc"],
            "rationale": f"p={p:.2f}, occ={env['occupancy_pct']:.0f}%, discount={row['discount_pct']:.0f}%"})
    offers = [o for o in offers if o["propensity"] >= thr and (o["category"] == "loyalty" or o["margin_pct"] >= s.min_margin_pct)
              and o["discount_pct"] <= max(s.max_discount_pct, float(next((c.get("fixed_discount_pct", 0) for c in catalog if c["sku"] == o["sku"]), 0)))]
    offers.sort(key=lambda o: -o["expected_value"])
    return offers[:top_k]
