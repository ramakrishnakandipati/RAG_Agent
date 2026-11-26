"""End-to-end GPU training of the quantitative suite: elasticity net, uplift learners, bandits, convex allocator.

Writes checkpoints (.pt) to outputs/checkpoints, JSON metrics to outputs/metrics, 300-DPI plots to outputs/plots,
and GPU telemetry (device, VRAM, latency ms) to outputs/logs/gpu_telemetry.jsonl.
Run:  python models/train_quant_suite.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from data.synthetic_generator import (CATS, SEGS, PRICE_MULTS, featurize, list_price, truth_probs, eligible,  # noqa: E402
                                      reward_tables, unit_margin, N_ACTIONS)
from evaluation.offline_replay import qini_auuc, qini_curve  # noqa: E402
from models import contextual_bandit as cb, convex_allocator as ca, elasticity_net as en, uplift_models as um  # noqa: E402
from models.utils import (get_device, gpu_snapshot, load_config, load_logs, path, seed_all, split_idx, timed, write_json)  # noqa: E402

DPI = 300


def train_elasticity(df, X, tr, va):
    off = (df.action > 0).to_numpy(); idx = np.flatnonzero(off)
    cat = np.where(off, df.category_idx.to_numpy() - 1, 0); lr = np.log(df.price_mult.to_numpy()).astype(np.float32); y = df.accepted.to_numpy().astype(np.float32)
    tr_o = np.intersect1d(tr, idx); va_o = np.intersect1d(va, idx)
    t0 = time.time(); model, hist = en.fit(X, cat, lr, y, tr_o, va_o); train_s = time.time() - t0
    with timed("elasticity_inference", len(va_o)) as tm:
        p = en.predict_np(model, X[va_o], cat[va_o], lr[va_o])
    # ground-truth comparison + monotonicity audit on a dense price grid
    true_p = np.zeros(len(va_o))
    for j, k in enumerate(CATS):
        m = cat[va_o] == j; sub = df.iloc[va_o[m]]
        true_p[m] = truth_probs(sub, k, np.exp(lr[va_o][m]))[1]
    grid = np.log(np.linspace(0.7, 1.3, 25)).astype(np.float32); sample = va_o[:2000]; viol = 0
    for j in range(len(CATS)):
        P = np.stack([en.predict_np(model, X[sample], np.full(len(sample), j), np.full(len(sample), g, dtype=np.float32)) for g in grid], 1)
        viol += int((np.diff(P, axis=1) > 1e-6).sum())
    yv = y[va_o]; bce = float(-np.mean(yv * np.log(p + 1e-9) + (1 - yv) * np.log(1 - p + 1e-9)))
    metrics = {"val_bce": bce, "rmse_vs_true_prob": float(np.sqrt(np.mean((p - true_p) ** 2))), "ece": en.ece(p, yv),
               "monotonicity_violations": viol, "train_seconds": round(train_s, 1), "inference_latency_ms": tm["latency_ms"],
               "vram_peak_mib": tm.get("vram_peak_mib"), "n_val": int(len(va_o))}
    en.save(model, X.shape[1], len(CATS))
    # plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(hist["train"], label="train"); ax[0].plot(hist["val"], label="val"); ax[0].set_title("Elasticity net: BCE loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    cal = en.calibration_curve(p, yv); ax[1].plot([0, 1], [0, 1], "k--"); ax[1].plot(cal["conf"], cal["acc"], "o-"); ax[1].set_title(f"Calibration (ECE={metrics['ece']:.4f})"); ax[1].set_xlabel("predicted"); ax[1].set_ylabel("observed")
    ratios = np.linspace(0.7, 1.3, 40)
    for s in SEGS:                                               # demand curve per segment, ROOM_UPGRADE
        sub = df[(df.segment == s) & (df.room_type != "OCEANFRONT_SUITE")].head(1500); Xs = featurize(sub)
        curve = [en.predict_np(model, Xs, np.zeros(len(sub), dtype=int), np.full(len(sub), np.log(r), dtype=np.float32)).mean() for r in ratios]
        ax[2].plot(ratios, curve, label=s)
    ax[2].set_title("Demand curves P(accept | price/list) - ROOM_UPGRADE"); ax[2].set_xlabel("price / list"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path("plots") / "elasticity_diagnostics.png", dpi=DPI); plt.close(fig)
    return model, metrics


def train_uplift(df, X, tr, va):
    cfg_u = load_config()["uplift"]; dragons, xls, metrics = {}, {}, {}
    fig, axes = plt.subplots(1, len(CATS), figsize=(17, 3.8))
    for j, k in enumerate(CATS):
        el = eligible(df, k)
        treated = (df.category_idx == j + 1) & (df.price_mult == 1.0)
        control = (df.action == 0) & (df.ref_category_idx == j + 1)
        sel = np.flatnonzero((treated | control).to_numpy() & el)
        s_tr, s_va = np.intersect1d(sel, tr), np.intersect1d(sel, va)
        t = treated.to_numpy().astype(np.float32); y = df.accepted.to_numpy().astype(np.float32)
        Xtr, ttr, ytr = X[s_tr], t[s_tr], y[s_tr]
        with timed(f"uplift_train_{k}", len(s_tr)) as tm:
            xl = um.XLearner(X.shape[1]).fit(Xtr, ttr.astype(int), ytr); dn = um.fit_dragonnet(Xtr, ttr, ytr)
        # CATE evaluation on every eligible validation guest, against ground truth tau_k(x)
        ev = np.intersect1d(np.flatnonzero(el), va); tau_true = df[f"tau_true_{k}"].to_numpy()[ev]
        tau_x, tau_t, tau_d = xl.predict_cate(X[ev]), xl.predict_t_learner(X[ev]), um.dragon_cate(dn, X[ev])
        Xv, tv, yv = X[s_va], t[s_va], y[s_va]
        qd = qini_auuc(um.dragon_cate(dn, Xv), tv, yv); qx = qini_auuc(xl.predict_cate(Xv), tv, yv)
        qr = qini_auuc(np.random.default_rng(0).random(len(yv)), tv, yv)
        metrics[k] = {"sqrt_pehe": {"x_learner": um.sqrt_pehe(tau_x, tau_true), "t_learner": um.sqrt_pehe(tau_t, tau_true), "dragonnet": um.sqrt_pehe(tau_d, tau_true),
                                    "constant_ate": um.sqrt_pehe(np.full_like(tau_true, df[f"tau_true_{k}"].to_numpy()[tr][el[tr]].mean()), tau_true)},
                      "corr_with_true_tau": {"x_learner": float(np.corrcoef(tau_x, tau_true)[0, 1]), "dragonnet": float(np.corrcoef(tau_d, tau_true)[0, 1])},
                      "qini_coefficient": {"dragonnet": qd["qini_coefficient"], "x_learner": qx["qini_coefficient"], "random": qr["qini_coefficient"]},
                      "auuc": {"dragonnet": qd["auuc"], "x_learner": qx["auuc"], "random": qr["auuc"]},
                      "n_train": int(len(s_tr)), "n_treated_train": int(ttr.sum()), "train_latency_ms": tm["latency_ms"]}
        dragons[k], xls[k] = dn, xl
        for nm, q in (("DragonNet", qd), ("X-Learner", qx), ("Random", qr)):
            axes[j].plot(q["phi"], q["qini_curve"], label=nm)
        axes[j].set_title(f"Qini - {k}", fontsize=9); axes[j].set_xlabel("fraction targeted")
        print(f"  [uplift] {k}: sqrtPEHE x={metrics[k]['sqrt_pehe']['x_learner']:.3f} dragon={metrics[k]['sqrt_pehe']['dragonnet']:.3f} const={metrics[k]['sqrt_pehe']['constant_ate']:.3f}")
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(path("plots") / "uplift_qini_curves.png", dpi=DPI); plt.close(fig)
    um.save_uplift(dragons, xls, X.shape[1])
    return dragons, xls, metrics


def train_bandits(df, X, tr):
    cfg = load_config()["bandit"]; sub = df.iloc[tr].reset_index(drop=True).head(40000); Xs = featurize(sub)
    P, M, E, mask = reward_tables(sub)
    logged = (Xs[:30000], sub.action.to_numpy()[:30000], sub.reward.to_numpy()[:30000])
    res, curves = {}, {}
    for name in ("random", "linucb", "neural_ts"):
        with timed(f"bandit_{name}", cfg["n_rounds"]) as tm:
            out = cb.simulate(name, Xs, P, M, mask, cfg["n_rounds"], cfg["batch"], seed=1, logged=logged if name == "neural_ts" else None)
        curves[name] = out["cumulative_regret"]
        res[name] = {"final_cumulative_regret": out["final_regret"], "mean_reward_last": out["mean_reward"], "rounds": out["rounds"], "latency_ms": tm["latency_ms"]}
        if name == "linucb":
            cb.save_bandit(out["_pol"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, v in curves.items():
        ax.plot(np.arange(1, len(v) + 1) * cfg["batch"], v, label=k)
    ax.set_xlabel("round"); ax.set_ylabel("cumulative regret ($)"); ax.set_title("Contextual bandits: cumulative regret (GPU batched)"); ax.legend()
    fig.tight_layout(); fig.savefig(path("plots") / "bandit_regret.png", dpi=DPI); plt.close(fig)
    return res


def run_allocator(df, el_model, n_guests=5000):
    cfg = load_config()["allocator"]; cat_cfg = load_config()["catalog"]; floors = load_config()["rms"]["min_upsell_floor"]
    sub = df.sample(n_guests, random_state=3).reset_index(drop=True); Xs = featurize(sub); K = len(CATS)
    P, Mg, D = (np.zeros((n_guests, K)) for _ in range(3))
    for j, k in enumerate(CATS):
        pl = list_price(sub, k); price, Pk, _ = en.optimal_price(el_model, Xs, np.full(n_guests, j), pl, unit_margin(k), floors[k])
        P[:, j] = np.where(eligible(sub, k), Pk, 0.0); Mg[:, j] = price - unit_margin(k); D[:, j] = np.clip(pl - price, 0, None)
    inv = [cfg["inventory_per_category"][k] for k in CATS]
    with timed("convex_allocator", n_guests * K) as tm:
        res = ca.solve(P, Mg, D, inv, cfg["promo_budget"])
    free = ca.solve(P, Mg, D, [1e9] * K, 1e12)                                            # unconstrained reference
    out = {"lam_inv": dict(zip(CATS, map(float, res["lam_inv"]))), "lam_budget": res["lam_budget"], "usage_inv": dict(zip(CATS, map(float, res["usage_inv"]))),
           "inventory_caps": dict(zip(CATS, inv)), "usage_budget": res["usage_budget"], "budget": cfg["promo_budget"], "objective": res["objective"],
           "unconstrained_objective": free["objective"], "primal_violation_max": res["primal_violation_max"],
           "comp_slackness_max": res["comp_slackness_max"], "latency_ms": tm["latency_ms"], "n_guests": n_guests}
    write_json("shadow_prices.json", ca.hurdle_table(res, CATS) | {"budget_usage": res["usage_budget"]})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ax[0].bar(CATS, [out["lam_inv"][k] for k in CATS]); ax[0].set_title("Inventory shadow prices lambda_inv ($/acceptance)"); ax[0].tick_params(axis="x", rotation=20)
    x = np.arange(K); ax[1].bar(x - 0.2, [out["usage_inv"][k] for k in CATS], 0.4, label="expected usage"); ax[1].bar(x + 0.2, inv, 0.4, label="cap"); ax[1].set_xticks(x); ax[1].set_xticklabels(CATS, rotation=20); ax[1].legend(); ax[1].set_title("Inventory usage vs cap")
    fig.tight_layout(); fig.savefig(path("plots") / "allocator_shadow_prices.png", dpi=DPI); plt.close(fig)
    return out


def main():
    seed_all(); dev = get_device(); print(f"[train] device={dev} {gpu_snapshot(dev)} cuda={torch.cuda.is_available()}")
    df = load_logs(); X = featurize(df); tr, va = split_idx(len(df)); report = {"device": gpu_snapshot(dev), "n_rows": len(df)}
    print("[1/4] price elasticity net"); el_model, report["elasticity"] = train_elasticity(df, X, tr, va); print("  ", report["elasticity"])
    print("[2/4] uplift learners"); _, _, report["uplift"] = train_uplift(df, X, tr, va)
    print("[3/4] contextual bandits"); report["bandit"] = train_bandits(df, X, tr); print("  ", report["bandit"])
    print("[4/4] convex allocator"); report["allocator"] = run_allocator(df.iloc[va], el_model); print("  ", report["allocator"])
    report["gpu_final"] = gpu_snapshot(dev); write_json("quant_suite_report.json", report); print("[train] done ->", path("metrics"))


if __name__ == "__main__":
    main()
