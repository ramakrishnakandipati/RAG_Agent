"""Synthetic hospitality data engine (reservations, CRM, PMS, RMS and bandit logs).

Mathematical model (see docs/MATHEMATICAL_FOUNDATIONS.md, L1)
-------------------------------------------------------------
Guest context x, category k, price multiplier m (offer price p = m * p0_k * room_factor):

  organic purchase at list price :  p0_k(x)  = sigma( a0_k + s_k[seg] + 0.30*tier + 0.35*spend + 0.15*prior )
  offer-driven take-up           :  u_k(x,m) = sigma( a1_k + g_k[seg] + 0.25*tier + 0.30*spend - 4*eps_seg * ln m )
  acceptance given offer         :  P(Y=1|x,k,m) = p0_k + (1 - p0_k) * u_k(x,m)       (>= organic, decreasing in m)
  true CATE at list price        :  tau_k(x) = P(Y=1|x,k,1) - p0_k(x)
  behaviour policy (confounded)  :  pi0(k|x) = softmax(gamma_k . x),  m ~ Uniform(M)   => logged propensity
  realised reward                :  r = Y * (p - c_k - hk_k)   (offer)   /   Y0 * (p_list - c_k - hk_k)   (no offer)

Code symbol <-> math symbol: `p0` = p0_k(x), `u` = u_k(x,m), `eps_seg` = eps_s, `pi0` = pi_0(a|x).
Action a in {0 = no offer} U {(k, m)} => 1 + K*|M| joint actions (21 with K=4, |M|=5).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.utils import load_config, path, seed_all, write_json  # noqa: E402

CFG = load_config()
CATS = CFG["data"]["categories"]
SEGS = CFG["data"]["segments"]
CHANNELS = CFG["data"]["channels"]
ROOMS = CFG["data"]["room_types"]
TIERS = CFG["data"]["tiers"]
PRICE_MULTS = np.array([0.75, 0.875, 1.0, 1.125, 1.25])
ROOM_FACTOR = {"STANDARD_QUEEN": 1.0, "DELUXE_KING": 1.1, "JUNIOR_OCEAN_VIEW": 1.25, "OCEANFRONT_SUITE": 1.5}
ROOM_ADR = {"STANDARD_QUEEN": 200, "DELUXE_KING": 260, "JUNIOR_OCEAN_VIEW": 350, "OCEANFRONT_SUITE": 600}
# price elasticity coefficient on ln m (logit scale), per segment: higher = more price sensitive
EPS = {"Leisure": 1.6, "Corporate": 0.7, "VIP_Loyalty": 0.5, "Honeymoon": 0.9, "Family": 1.5}
# category intercepts and segment tilts
A0 = {"ROOM_UPGRADE": -3.2, "SPA_PACKAGE": -3.6, "DINING_EXPERIENCE": -3.4, "LATE_CHECKOUT": -2.6}
A1 = {"ROOM_UPGRADE": -2.7, "SPA_PACKAGE": -3.0, "DINING_EXPERIENCE": -2.9, "LATE_CHECKOUT": -2.0}
SEG_TILT = {  # (organic tilt, offer tilt) per category per segment
    "ROOM_UPGRADE": {"Leisure": (0.0, 0.2), "Corporate": (-0.2, -0.2), "VIP_Loyalty": (0.3, 0.6), "Honeymoon": (0.4, 0.9), "Family": (0.2, 0.5)},
    "SPA_PACKAGE": {"Leisure": (0.0, 0.2), "Corporate": (-0.6, -0.8), "VIP_Loyalty": (0.2, 0.4), "Honeymoon": (0.7, 1.2), "Family": (-0.2, 0.0)},
    "DINING_EXPERIENCE": {"Leisure": (0.0, 0.1), "Corporate": (0.2, 0.2), "VIP_Loyalty": (0.5, 0.8), "Honeymoon": (0.6, 1.0), "Family": (0.1, 0.3)},
    "LATE_CHECKOUT": {"Leisure": (0.0, 0.2), "Corporate": (0.8, 0.9), "VIP_Loyalty": (0.2, 0.2), "Honeymoon": (0.1, 0.2), "Family": (0.4, 0.7)},
}
NUM_COLS = ["los", "lead_time", "adr", "party_size", "past_spend", "n_prior_stays", "contacts_7d", "is_weekend"]
NUM_SCALE = np.array([10, 7, 600, 6, 3000, 20, 3, 1], dtype=np.float32)
FEATURE_COLS = NUM_COLS + [f"seg_{s}" for s in SEGS] + [f"ch_{c}" for c in CHANNELS] + [f"room_{r}" for r in ROOMS] + [f"tier_{t}" for t in TIERS]
N_ACTIONS = 1 + len(CATS) * len(PRICE_MULTS)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def featurize(df: pd.DataFrame) -> np.ndarray:
    """Fixed, scaler-free feature map phi(x) in R^25 (numeric / constant + one-hots)."""
    num = df[NUM_COLS].to_numpy(dtype=np.float32) / NUM_SCALE
    oh = [(df["segment"].to_numpy()[:, None] == np.array(SEGS)[None]).astype(np.float32),
          (df["channel"].to_numpy()[:, None] == np.array(CHANNELS)[None]).astype(np.float32),
          (df["room_type"].to_numpy()[:, None] == np.array(ROOMS)[None]).astype(np.float32),
          (df["loyalty_tier"].to_numpy()[:, None] == np.array(TIERS)[None]).astype(np.float32)]
    return np.concatenate([num] + oh, axis=1).astype(np.float32)


def list_price(df: pd.DataFrame, k: str) -> np.ndarray:
    return CFG["catalog"][k]["base_price"] * df["room_type"].map(ROOM_FACTOR).to_numpy()


def eligible(df: pd.DataFrame, k: str) -> np.ndarray:
    if k == "ROOM_UPGRADE":                                  # top room has nothing above it
        return (df["room_type"] != "OCEANFRONT_SUITE").to_numpy()
    return np.ones(len(df), dtype=bool)


def entitled(df: pd.DataFrame, k: str) -> np.ndarray:
    """Guests whose loyalty tier already grants category k free (Platinum/Diamond: late checkout & upgrade) -> zero revenue."""
    pol_perks = {"LATE_CHECKOUT": "LATE_CHECKOUT_2PM", "ROOM_UPGRADE": "ROOM_UPGRADE_SUBJECT_TO_AVAIL"}
    from models.utils import load_policies
    perks = load_policies()["loyalty_perks"]
    if k not in pol_perks:
        return np.zeros(len(df), dtype=bool)
    tiers_with = [t for t, pl in perks.items() if pol_perks[k] in pl]
    return df["loyalty_tier"].isin(tiers_with).to_numpy()


def truth_probs(df: pd.DataFrame, k: str, mult) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth (organic p0, acceptance-given-offer P1) for category k at price multiplier(s) `mult`."""
    tier = df["loyalty_tier"].map({t: i for i, t in enumerate(TIERS)}).to_numpy() / 4.0
    spend = df["past_spend"].to_numpy() / 3000.0
    prior = df["n_prior_stays"].to_numpy() / 20.0
    s0 = df["segment"].map({s: SEG_TILT[k][s][0] for s in SEGS}).to_numpy()
    s1 = df["segment"].map({s: SEG_TILT[k][s][1] for s in SEGS}).to_numpy()
    eps = df["segment"].map(EPS).to_numpy()
    p0 = _sigmoid(A0[k] + s0 + 1.2 * tier + 0.9 * spend + 0.6 * prior)
    u = _sigmoid(A1[k] + s1 + 1.0 * tier + 0.9 * spend - 4.0 * eps * np.log(mult))
    return p0, p0 + (1 - p0) * u


def unit_margin(k: str) -> float:
    c = CFG["catalog"][k]
    return c["cost"] + c["housekeeping"]


def generate_reservations(n: int, rng: np.random.Generator) -> pd.DataFrame:
    seg = rng.choice(SEGS, n, p=[0.36, 0.22, 0.14, 0.10, 0.18])
    ch = np.where(seg == "Corporate", rng.choice(CHANNELS, n, p=[0.3, 0.1, 0.6]), rng.choice(CHANNELS, n, p=[0.45, 0.50, 0.05]))
    room = rng.choice(ROOMS, n, p=[0.35, 0.33, 0.22, 0.10])
    tier_p = {"VIP_Loyalty": [0.0, 0.1, 0.2, 0.3, 0.4], "Corporate": [0.3, 0.3, 0.2, 0.15, 0.05]}
    default_p = [0.5, 0.22, 0.14, 0.09, 0.05]
    tier = np.empty(n, dtype=object)
    for s in SEGS:
        idx = seg == s
        tier[idx] = rng.choice(TIERS, idx.sum(), p=tier_p.get(s, default_p))
    tier_i = pd.Series(tier).map({t: i for i, t in enumerate(TIERS)}).to_numpy()
    df = pd.DataFrame({
        "reservation_id": [f"RES-{i + 1:05d}" for i in range(n)],
        "segment": seg, "channel": ch, "room_type": room, "loyalty_tier": tier,
        "los": np.clip(rng.poisson(2.8, n) + 1, 1, 10),
        "lead_time": rng.integers(1, 8, n),                              # T-7 .. T-1
        "party_size": np.clip(rng.poisson(1.6, n) + 1, 1, 6) + (seg == "Family") * 2,
        "n_prior_stays": np.clip(rng.poisson(1 + 2.2 * tier_i, n), 0, 20),
        "contacts_7d": rng.poisson(0.7, n).clip(0, 3),
        "is_weekend": rng.integers(0, 2, n),
    })
    df["adr"] = df["room_type"].map(ROOM_ADR) * rng.normal(1.0, 0.12, n)
    df["past_spend"] = np.clip(rng.gamma(2.0, 180 + 120 * tier_i + 60 * df["n_prior_stays"].to_numpy() / 5, n), 0, 3000)
    df["email"] = [f"guest{i + 1}@example.com" for i in range(n)]
    df["first_name"] = rng.choice(["Amelia", "Noah", "Sofia", "Liam", "Isabella", "Ethan", "Mei", "Arjun", "Elena", "Omar"], n)
    return df


def behaviour_logits(X: np.ndarray, rng_w: np.random.Generator) -> np.ndarray:
    """gamma_k . x : confounded targeting - tier/spend/segment drive WHO gets offered WHAT."""
    W = rng_w.normal(0, 0.35, (X.shape[1], len(CATS) + 1))
    W[FEATURE_COLS.index("past_spend"), 1:] += 1.2
    for t in ("Platinum", "Diamond"):
        W[FEATURE_COLS.index(f"tier_{t}"), 1:] += 0.6
    W[:, 0] -= 0.0
    return X @ W + np.array([0.9, 0, 0, 0, 0])      # bias toward "no offer"


def simulate_logs(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    X = featurize(df)
    logits = behaviour_logits(X, np.random.default_rng(CFG["seed"] + 1))
    for j, k in enumerate(CATS):                       # eligibility mask
        logits[~eligible(df, k), j + 1] = -1e9
    z = logits - logits.max(1, keepdims=True)
    pi_cat = np.exp(z); pi_cat /= pi_cat.sum(1, keepdims=True)           # pi0(k | x), k=0 is "no offer"
    cum = pi_cat.cumsum(1); u = rng.random((n, 1))
    a_cat = (u > cum).sum(1).clip(0, len(CATS))
    m_idx = rng.integers(0, len(PRICE_MULTS), n)
    mult = np.where(a_cat > 0, PRICE_MULTS[m_idx], 1.0)
    pi0_joint = np.where(a_cat == 0, pi_cat[:, 0], pi_cat[np.arange(n), a_cat] / len(PRICE_MULTS))
    action = np.where(a_cat == 0, 0, 1 + (a_cat - 1) * len(PRICE_MULTS) + m_idx)

    y = np.zeros(n); price = np.zeros(n); reward = np.zeros(n); tau_cols = {}; p0_cols = {}
    for j, k in enumerate(CATS):
        p0, p1_list = truth_probs(df, k, 1.0)
        _, p1 = truth_probs(df, k, mult)
        tau_cols[f"tau_true_{k}"] = np.where(eligible(df, k), p1_list - p0, 0.0)
        p0_cols[f"p0_true_{k}"] = p0
        sel = a_cat == j + 1
        pl = list_price(df, k)
        y[sel] = rng.random(sel.sum()) < p1[sel]
        price[sel] = (pl * mult)[sel]
        reward[sel] = y[sel] * (price[sel] * (~entitled(df, k))[sel] - unit_margin(k))
    sel0 = a_cat == 0                                  # no-offer: organic purchase of a random "reference" category
    ref = rng.integers(0, len(CATS), n)
    for j, k in enumerate(CATS):
        s = sel0 & (ref == j)
        p0, _ = truth_probs(df, k, 1.0)
        y[s] = rng.random(s.sum()) < p0[s] * eligible(df, k)[s]
        price[s] = list_price(df, k)[s]
        reward[s] = y[s] * (price[s] * (~entitled(df, k))[s] - unit_margin(k))
    out = df.copy()
    out["action"] = action; out["category_idx"] = a_cat; out["price_mult"] = mult; out["offer_price"] = price
    out["propensity"] = pi0_joint; out["accepted"] = y.astype(int); out["reward"] = reward
    out["ref_category_idx"] = np.where(sel0, ref + 1, 0)   # which category an organic purchase refers to (no-offer rows)
    for d in (tau_cols, p0_cols):
        for c, v in d.items():
            out[c] = v
    return out


def reward_tables(df: pd.DataFrame):
    """Ground-truth tables over the 21 joint actions: (P, M, E, mask), each [n, 21].
    P = acceptance prob, M = unit margin if accepted, E = P*M expected reward, mask = action eligibility.
    Column 0 = no offer (P=1, M=E=expected organic margin averaged over the 4 reference categories)."""
    n = len(df); P = np.ones((n, N_ACTIONS)); M = np.zeros((n, N_ACTIONS)); mask = np.ones((n, N_ACTIONS), dtype=bool)
    org = np.zeros(n)
    for j, k in enumerate(CATS):
        pl = list_price(df, k); el = eligible(df, k); p0, _ = truth_probs(df, k, 1.0)
        ent = entitled(df, k)
        org += (p0 * (pl * (~ent) - unit_margin(k)) * el) / len(CATS)
        for mi, m in enumerate(PRICE_MULTS):
            a = 1 + j * len(PRICE_MULTS) + mi
            _, p1 = truth_probs(df, k, m)
            P[:, a] = p1; M[:, a] = pl * m * (~ent) - unit_margin(k); mask[:, a] = el
    M[:, 0] = org
    return P, M, P * M, mask


def pms_rms_tables(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = {"STANDARD_QUEEN": 60, "DELUXE_KING": 48, "JUNIOR_OCEAN_VIEW": 30, "OCEANFRONT_SUITE": 12}
    pms = pd.DataFrame([{"room_type": r, "capacity": c, "live_available_rooms": int(rng.integers(max(1, c // 8), c // 2)),
                         "daily_demand_sigma": round(float(c * 0.07), 2)} for r, c in cap.items()])
    rms = pd.DataFrame([{"category": k, "min_upsell_floor": v} for k, v in CFG["rms"]["min_upsell_floor"].items()])
    return pms, rms


def main(n: int | None = None) -> pd.DataFrame:
    seed_all()
    n = n or CFG["data"]["n_reservations"]
    rng = np.random.default_rng(CFG["seed"])
    logs = simulate_logs(generate_reservations(n, rng), rng)
    pms, rms = pms_rms_tables(rng)
    out = path("data")
    logs.to_parquet(out / "reservations_logs.parquet", index=False)
    pms.to_csv(out / "pms_inventory.csv", index=False); rms.to_csv(out / "rms_floors.csv", index=False)
    summ = {"n_rows": int(len(logs)), "offer_rate": float((logs.action > 0).mean()), "accept_rate_given_offer": float(logs.loc[logs.action > 0, "accepted"].mean()),
            "mean_reward": float(logs.reward.mean()), "n_actions": N_ACTIONS,
            "mean_true_tau": {k: float(logs[f"tau_true_{k}"].mean()) for k in CATS}}
    write_json("data_summary.json", summ)
    print(f"[data] {len(logs):,} reservations -> {out}\n{summ}")
    return logs


if __name__ == "__main__":
    main()
