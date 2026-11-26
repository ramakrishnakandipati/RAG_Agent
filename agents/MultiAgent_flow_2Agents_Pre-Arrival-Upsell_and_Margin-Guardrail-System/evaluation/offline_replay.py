"""Offline policy evaluation (OPE): IPS / SNIPS / Doubly Robust, Qini & AUUC, regret, guardrail impact.

Math (docs L9)
--------------
Logged tuples (x_i, a_i, r_i, pi0(a_i|x_i)); target policy pi (deterministic => pi(a|x)=1[a=pi(x)]); w_i = pi(a_i|x_i)/pi0(a_i|x_i).
  IPS   : V = 1/N sum w_i r_i                    (weights clipped at M)
  SNIPS : V = sum w_i r_i / sum w_i
  DR    : V = 1/N sum [ q_pi(x_i) + w_i (r_i - q(x_i,a_i)) ],   q_pi(x) = sum_a pi(a|x) q(x,a)
  ESS   : (sum w)^2 / sum w^2
Qini  : Q(phi) = Y_t(phi) - Y_c(phi) * n_t(phi)/n_c(phi)  (customers sorted by predicted uplift, top phi fraction);
        Qini coefficient = area between Q and the random line  Q(1)*phi ;  AUUC = area under Q  (both /N, phi in [0,1]).
Regret: Reg(pi) = E[ max_a E[r|x,a] - E[r|x,pi(x)] ]  using simulator ground truth.
Symbols: `w` = w_i, `clip` = M, `q_hat` = q(x,a).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.utils import load_config, load_logs, load_policies, split_idx, write_json, path, seed_all  # noqa: E402


# ----------------------------------------------------------------------------------- estimators
def ips(r, w, clip=50.0):
    return float(np.mean(np.minimum(w, clip) * r))


def snips(r, w, clip=50.0):
    w = np.minimum(w, clip)
    return float((w * r).sum() / max(w.sum(), 1e-9))


def doubly_robust(r, w, q_pi, q_logged, clip=50.0):
    return float(np.mean(q_pi + np.minimum(w, clip) * (r - q_logged)))


def ess(w):
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-9))


def bootstrap_ci(fn, n, B=200, seed=0):
    rng = np.random.default_rng(seed); vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def qini_curve(score, t, y, n_points=101):
    order = np.argsort(-score); t, y = np.asarray(t)[order], np.asarray(y)[order]
    ct, cc = np.cumsum(t), np.cumsum(1 - t); yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = yt - yc * ct / np.maximum(cc, 1)
    idx = np.linspace(0, len(y) - 1, n_points).astype(int)
    return np.linspace(0, 1, n_points), q[idx] / len(y)


def qini_auuc(score, t, y):
    phi, q = qini_curve(score, t, y)
    auuc = float(np.trapezoid(q, phi)); rand = float(np.trapezoid(q[-1] * phi, phi))
    return {"auuc": auuc, "qini_coefficient": auuc - rand, "qini_curve": q.tolist(), "phi": phi.tolist()}


# ----------------------------------------------------------------------------------- policies
def guardrail_mask(df: pd.DataFrame, price_mults, cats, catalog, rms_floor, min_margin_ratio, perk_map) -> np.ndarray:
    """Vectorised replay of the Auditor's deterministic contracts over the 21 joint actions: [n,21] allowed."""
    from data.synthetic_generator import list_price, entitled, eligible
    n = len(df); ok = np.ones((n, 1 + len(cats) * len(price_mults)), dtype=bool)
    for j, k in enumerate(cats):
        pl = list_price(df, k); c = catalog[k]["cost"] + catalog[k]["housekeeping"]; ent = entitled(df, k); el = eligible(df, k)
        for mi, m in enumerate(price_mults):
            a = 1 + j * len(price_mults) + mi; price = pl * m
            ok[:, a] = el & (price >= rms_floor[k]) & ((price - c) / price >= min_margin_ratio) & (~ent)
    return ok


def evaluate_policy(name, actions, df, E_true, mask_elig, q_hat, w_clip):
    """actions: [n] chosen joint actions. Returns OPE estimates + ground truth."""
    a_log = df["action"].to_numpy(); r = df["reward"].to_numpy(); pi0 = df["propensity"].to_numpy()
    if name == "logged_behaviour":            # on-policy: pi == pi0 so w_i == 1 (IPS = DR = sample mean)
        w = np.ones(len(df))
    else:
        w = (actions == a_log).astype(float) / pi0
    q_pi = q_hat[np.arange(len(df)), actions]; q_log = q_hat[np.arange(len(df)), a_log]
    best = np.where(mask_elig, E_true, -1e9).max(1)
    true_v = float(E_true[np.arange(len(df)), actions].mean())
    dr_fn = lambda ix: doubly_robust(r[ix], w[ix], q_pi[ix], q_log[ix], w_clip)
    return {"policy": name, "true_value": true_v, "ips": ips(r, w, w_clip), "snips": snips(r, w, w_clip),
            "dr": doubly_robust(r, w, q_pi, q_log, w_clip), "dr_ci95": bootstrap_ci(dr_fn, len(df)), "ess": ess(np.minimum(w, w_clip)),
            "regret": float((best - E_true[np.arange(len(df)), actions]).mean())}


def main():
    import torch
    from data.synthetic_generator import (CATS, PRICE_MULTS, featurize, reward_tables, list_price, unit_margin, entitled, eligible, N_ACTIONS)
    from models import elasticity_net as en
    from models.utils import timed
    seed_all(); cfg = load_config(); pol = load_policies()
    df_all = load_logs(); _, va = split_idx(len(df_all)); df = df_all.iloc[va].reset_index(drop=True)
    X = featurize(df); n = len(df)
    P_true, M_true, E_true, elig = reward_tables(df)
    el_model = en.load()

    # q_hat(x,a): elasticity-net acceptance prob x unit margin (cannibalisation-unaware, as a real ML model would be)
    q_hat = np.zeros((n, N_ACTIONS)); P_hat = np.ones((n, N_ACTIONS)); M_hat = np.zeros((n, N_ACTIONS))
    with timed("ope_q_hat_inference", n * (N_ACTIONS - 1)):
        for j, k in enumerate(CATS):
            pl = list_price(df, k)
            for mi, m in enumerate(PRICE_MULTS):
                a = 1 + j * len(PRICE_MULTS) + mi
                P_hat[:, a] = en.predict_np(el_model, X, np.full(n, j), np.full(n, np.log(m), dtype=np.float32))
                M_hat[:, a] = pl * m - unit_margin(k)
    # no-offer arm: observed mean organic reward per segment in the logs (what a model would learn from control rows)
    ctrl = df[df.action == 0].groupby("segment")["reward"].mean()
    M_hat[:, 0] = df["segment"].map(ctrl).to_numpy()
    q_hat = np.where(elig, P_hat * M_hat, -1e9); q_hat_fit = np.where(elig, q_hat, 0.0)

    catalog = cfg["catalog"]; floors = cfg["rms"]["min_upsell_floor"]
    allowed = guardrail_mask(df, PRICE_MULTS, CATS, catalog, floors, cfg["rms"]["min_margin_ratio"], pol["loyalty_perks"])
    allowed[:, 0] = True
    ent_mat = np.zeros((n, N_ACTIONS), dtype=bool)                 # actions that give away a perk the tier already holds free
    for j, k in enumerate(CATS):
        ent_mat[:, 1 + j * len(PRICE_MULTS):1 + (j + 1) * len(PRICE_MULTS)] = entitled(df, k)[:, None]
    rng = np.random.default_rng(0)

    policies = {
        "logged_behaviour": df["action"].to_numpy(),
        "no_offer": np.zeros(n, dtype=int),
        "random_eligible": np.array([rng.choice(np.flatnonzero(m)) for m in elig]),
        "ml_only(elasticity greedy)": np.where(elig, q_hat, -1e9).argmax(1),
        "ml+agent_guardrails": np.where(elig & allowed, q_hat, -1e9).argmax(1),
        "oracle": np.where(elig, E_true, -1e9).argmax(1),
    }
    rows, viol = [], {}
    for name, a in policies.items():
        res = evaluate_policy(name, a, df, E_true, elig, q_hat_fit, w_clip=50.0); rows.append(res)
        res["cannibalisation_rate"] = float(ent_mat[np.arange(n), a].mean())
        viol[name] = float((~allowed[np.arange(n), a]).mean())
        res["guardrail_violation_rate"] = viol[name]
    # Qini / AUUC from uplift models on observed data per category (holdout rows only)
    uplift = {}
    try:
        from models.uplift_models import load_uplift, dragon_cate
        dragons, xls = load_uplift()
        for j, k in enumerate(CATS):
            sel = ((df.category_idx == j + 1) & (df.price_mult == 1.0)) | ((df.action == 0) & (df.ref_category_idx == j + 1))
            sel &= eligible(df, k); sub = df[sel]; Xs = featurize(sub); t = (sub.action > 0).to_numpy().astype(int); y = sub.accepted.to_numpy()
            uplift[k] = {"dragonnet": {m: v for m, v in qini_auuc(dragon_cate(dragons[k], Xs), t, y).items() if m not in ("qini_curve", "phi")},
                         "x_learner": {m: v for m, v in qini_auuc(xls[k].predict_cate(Xs), t, y).items() if m not in ("qini_curve", "phi")},
                         "random": {m: v for m, v in qini_auuc(np.random.default_rng(1).random(len(y)), t, y).items() if m not in ("qini_curve", "phi")},
                         "n": int(len(y))}
    except FileNotFoundError:
        pass
    report = {"n_eval": n, "policies": rows, "uplift_qini": uplift,
              "improvement_vs_logged_pct": {r["policy"]: 100 * (r["true_value"] / rows[0]["true_value"] - 1) for r in rows}}
    write_json("ope_report.json", report)
    _plots(rows, report)
    for r in rows:
        print(f"  {r['policy']:<30} true={r['true_value']:7.2f} IPS={r['ips']:7.2f} DR={r['dr']:7.2f} ESS={r['ess']:8.0f} viol={r['guardrail_violation_rate']:.3f} cannib={r['cannibalisation_rate']:.3f} regret={r['regret']:.2f}")
    return report


def _plots(rows, report):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5)); names = [r["policy"] for r in rows]; x = np.arange(len(rows)); wdt = 0.2
    for i, (key, lab) in enumerate([("true_value", "Ground truth"), ("ips", "IPS"), ("snips", "SNIPS"), ("dr", "Doubly Robust")]):
        ax.bar(x + (i - 1.5) * wdt, [r[key] for r in rows], wdt, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right"); ax.set_ylabel("Value ($ margin / guest)"); ax.legend(); ax.set_title("Offline policy value: estimators vs ground truth")
    fig.tight_layout(); fig.savefig(path("plots") / "ope_policy_values.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
