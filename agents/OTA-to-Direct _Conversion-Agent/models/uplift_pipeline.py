"""Feature building + CUDA XGBoost T-/X-Learner uplift and cancellation-risk models + inference service."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings, xgb_device

TIER = {"none": 0, "member": 1, "silver": 2, "gold": 3}
FEATURES = ["adr", "los", "lead_days", "ota_commission", "saved_commission_usd", "free_cancellation", "is_resolved", "tier_n",
            "past_direct_stays", "past_ota_stays", "ltv", "email_opt_in", "msgs_30d",
            "ota_expedia", "ota_booking", "ota_agoda", "purpose_leisure", "purpose_business", "purpose_group"]
CANCEL_FEATURES = ["lead_days", "free_cancellation", "ota_booking", "ota_expedia", "ota_agoda", "los", "tier_n", "adr", "purpose_leisure",
                   "purpose_business", "purpose_group", "is_resolved"]


def row_features(ota: str, adr: float, los: int, lead_days: int, comm: float, free_cancel: bool, purpose: str, profile: dict | None) -> dict:
    p = profile or {}
    return {"adr": float(adr), "los": int(los), "lead_days": int(lead_days), "ota_commission": float(comm),
            "saved_commission_usd": round(float(adr) * int(los) * float(comm), 2), "free_cancellation": int(bool(free_cancel)),
            "is_resolved": int(profile is not None), "tier_n": TIER.get(p.get("loyalty_tier", "none"), 0),
            "past_direct_stays": int(p.get("past_direct_stays", 0)), "past_ota_stays": int(p.get("past_ota_stays", 0)),
            "ltv": float(p.get("lifetime_value_usd", 0.0)), "email_opt_in": int(bool(p.get("email_opt_in", False))),
            "msgs_30d": int(p.get("msgs_30d", 0)),
            "ota_expedia": int(ota == "Expedia"), "ota_booking": int(ota == "Booking.com"), "ota_agoda": int(ota == "Agoda"),
            "purpose_leisure": int(purpose == "leisure"), "purpose_business": int(purpose == "business"), "purpose_group": int(purpose == "group")}


def build_features(resolved: pd.DataFrame, cdp: pd.DataFrame) -> pd.DataFrame:
    m = resolved.merge(cdp.add_prefix("c_"), left_on="resolved_profile_id", right_on="c_profile_id", how="left")
    f = pd.DataFrame(index=m.index)
    r = m.c_profile_id.notna()
    f["adr"], f["los"], f["lead_days"], f["ota_commission"] = m.adr, m.los, m.lead_days, m.ota_commission
    f["saved_commission_usd"] = m.saved_commission_usd; f["free_cancellation"] = m.free_cancellation.astype(int)
    f["is_resolved"] = r.astype(int); f["tier_n"] = m.c_loyalty_tier.map(TIER).fillna(0).astype(int)
    f["past_direct_stays"] = m.c_past_direct_stays.fillna(0); f["past_ota_stays"] = m.c_past_ota_stays.fillna(0)
    f["ltv"] = m.c_lifetime_value_usd.fillna(0); f["email_opt_in"] = m.c_email_opt_in.fillna(False).astype(int)
    f["msgs_30d"] = m.c_msgs_30d.fillna(0)
    f["ota_expedia"], f["ota_booking"], f["ota_agoda"] = (m.ota == "Expedia").astype(int), (m.ota == "Booking.com").astype(int), (m.ota == "Agoda").astype(int)
    for p in ("leisure", "business", "group"): f[f"purpose_{p}"] = (m.trip_purpose == p).astype(int)
    return f[FEATURES].astype(float).reset_index(drop=True)


def _params(device: str, **kw) -> dict:
    return dict(tree_method="hist", device=device, n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, random_state=get_settings().SEED, n_jobs=4, **kw)


class UpliftModel:
    """T- and X-learner on XGBoost. predict() returns CATE tau(x) in [-1,1]."""
    def __init__(self, device: str | None = None, kind: str = "x"):
        self.device = device or xgb_device(); self.kind = kind
        self.m0 = self.m1 = self.t0 = self.t1 = None; self.g = 0.85

    def fit(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray):
        X1, X0, y1, y0 = X[t == 1], X[t == 0], y[t == 1], y[t == 0]
        self.g = float(t.mean())
        self.m1 = xgb.XGBClassifier(**_params(self.device)).fit(X1, y1)
        self.m0 = xgb.XGBClassifier(**_params(self.device, )).fit(X0, y0)
        if self.kind == "x":
            d1 = y1 - self.m0.predict_proba(X1)[:, 1]; d0 = self.m1.predict_proba(X0)[:, 1] - y0
            rp = _params(self.device, ); rp.update(n_estimators=150, max_depth=3)
            self.t1 = xgb.XGBRegressor(**rp).fit(X1, d1); self.t0 = xgb.XGBRegressor(**rp).fit(X0, d0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "t":
            tau = self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]
        else:
            tau = self.g * self.t0.predict(X) + (1 - self.g) * self.t1.predict(X)
        return np.clip(tau, -1, 1)

    def save(self, d: Path, name: str):
        d.mkdir(parents=True, exist_ok=True)
        for k in ("m0", "m1", "t0", "t1"):
            if getattr(self, k) is not None: getattr(self, k).save_model(d / f"{name}_{k}.json")
        (d / f"{name}_meta.json").write_text(json.dumps({"kind": self.kind, "g": self.g, "device_trained": self.device}))

    @classmethod
    def load(cls, d: Path, name: str, device: str | None = None):
        meta = json.loads((d / f"{name}_meta.json").read_text()); o = cls(device, meta["kind"]); o.g = meta["g"]
        for k, C in (("m0", xgb.XGBClassifier), ("m1", xgb.XGBClassifier), ("t0", xgb.XGBRegressor), ("t1", xgb.XGBRegressor)):
            p = d / f"{name}_{k}.json"
            if p.exists():
                m = C(); m.load_model(p); m.set_params(device=o.device); setattr(o, k, m)
        return o


def train_cancel(Xtr, ytr, device: str):
    return xgb.XGBClassifier(**_params(device, eval_metric="auc")).fit(Xtr[CANCEL_FEATURES], ytr)


def qini(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_pts: int = 100):
    """Qini curve + coefficient (area between model curve and random-targeting line, normalised by N)."""
    order = np.argsort(-score); y, t = y[order], t[order]
    cT, cC = np.cumsum(y * t), np.cumsum(y * (1 - t)); nT, nC = np.cumsum(t), np.cumsum(1 - t)
    q = cT - cC * np.divide(nT, nC, out=np.zeros_like(nT, dtype=float), where=nC > 0)
    idx = np.linspace(0, len(y) - 1, n_pts).astype(int)
    x = (idx + 1) / len(y); qc = q[idx]
    x = np.r_[0, x]; qc = np.r_[0, qc]
    rand = qc[-1] * x
    coef = float(np.trapezoid(qc - rand, x) / len(y))
    return x, qc, rand, coef


def calibration_bins(y, p, n=10):
    b = pd.DataFrame({"y": y, "p": p}); b["bin"] = pd.qcut(b.p, n, duplicates="drop")
    g = b.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return g, float((g.n * (g.pred - g.obs).abs()).sum() / g.n.sum())


class ScoringService:
    """Loads promoted models once; scores a single feature dict with latency telemetry."""
    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        s = get_settings(); d = model_dir or s.path("models", "x")  .parent
        self.device = device or xgb_device()
        self.uplift = UpliftModel.load(d, "uplift", self.device)
        self.cancel = xgb.XGBClassifier(); self.cancel.load_model(d / "cancel.json"); self.cancel.set_params(device=self.device)
        self.iso = None
        p = d / "cancel_isotonic.json"
        if p.exists():
            from sklearn.isotonic import IsotonicRegression
            j = json.loads(p.read_text()); self.iso = IsotonicRegression(out_of_bounds="clip").fit(j["x"], j["y"])
        self.warm = False

    def score(self, feats: dict) -> dict:
        X = pd.DataFrame([feats])[FEATURES].astype(float)
        t0 = time.perf_counter()
        tau = float(self.uplift.predict(X)[0]); pc = float(self.cancel.predict_proba(X[CANCEL_FEATURES])[:, 1][0])
        if self.iso is not None: pc = float(self.iso.predict([pc])[0])
        return {"tau": tau, "p_cancel": float(np.clip(pc, 0.005, 0.995)), "latency_ms": round((time.perf_counter() - t0) * 1000, 3), "device": self.device}
