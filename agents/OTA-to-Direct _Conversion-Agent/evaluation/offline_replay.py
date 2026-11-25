"""Offline replay: policy-targeted bookings, treated vs 15% ghost holdout -> measured lift, ROI, Qini and margin-waterfall charts."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import get_settings
from data.synthetic_generator import GEN_DIR
from agent import policy_engine as P

C = {"blue": "#2a6fdb", "gray": "#8a94a6", "green": "#1f9d6b", "red": "#d1495b", "amber": "#e3a008", "ink": "#1b2333"}


def _policy(row) -> tuple[str, float, float]:
    """Deterministic offline policy mirroring the agent's fallback + guardrails (pre_arrival window)."""
    kw = dict(tau=row.tau_pred, p_cancel=row.p_cancel, saved_commission=row.saved_commission_usd, stay_revenue=row.adr * row.los,
              ota_rate=row.adr, direct_rate=row.adr, msgs_30d=int(row.msgs_30d), opt_in=bool(row.email_opt_in) if row.is_resolved else False, resolved=bool(row.is_resolved))
    if row.is_resolved == 0:       # shadow profile: consent unknown -> only on-property wifi/post-stay allowed; pre-arrival blocked by R5
        pass
    oid, v = P.best_compliant_offer("pre_arrival", [], **kw)
    cost = P.catalog_offers()[oid]["cost_usd"]
    return oid, v.enrd, cost


def run(seed: int = 7) -> dict:
    s = get_settings(); rng = np.random.default_rng(seed)
    f = pd.read_csv(GEN_DIR / "scored_features.csv"); r = pd.read_csv(GEN_DIR / "resolved_bookings.csv", usecols=["booking_id", "treated", "converted_direct", "true_tau", "is_ghost_holdout", "ota", "cancelled"])
    d = pd.concat([f, r], axis=1); d = d[d.split == "test"].reset_index(drop=True)
    pol = d.apply(_policy, axis=1, result_type="expand"); d["offer"], d["enrd"], d["cost"] = pol[0], pol[1], pol[2]
    d["target"] = d.offer != P.NO_OFFER
    tgt = d[d.target]; T_, C_ = tgt[tgt.treated == 1], tgt[tgt.treated == 0]
    lift = T_.converted_direct.mean() - C_.converted_direct.mean()
    boots = [(lambda a, b: a.converted_direct.mean() - b.converted_direct.mean())(T_.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))), C_.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) for _ in range(300)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    avg_sc = float(tgt.saved_commission_usd.mean()); n_t = len(T_)
    incr = lift * n_t; gross = incr * avg_sc
    leakage = gross * float(T_.p_cancel.mean()); cost = float((T_.converted_direct * T_.cost).sum())
    net = gross - leakage - cost

    # blast baseline: everyone treated with cheapest-perk offer, no model
    allT, allC = d[d.treated == 1], d[d.treated == 0]
    blast_lift = allT.converted_direct.mean() - allC.converted_direct.mean()
    blast_net = blast_lift * len(allT) * d.saved_commission_usd.mean() * (1 - d.p_cancel.mean()) - float(allT.converted_direct.sum()) * 12.0

    # counterfactual lift by predicted-tau decile
    d["decile"] = pd.qcut(d.tau_pred.rank(method="first"), 10, labels=False)
    dec = d.groupby("decile").apply(lambda g: pd.Series({
        "pred_tau": g.tau_pred.mean(), "true_tau": g.true_tau.mean(),
        "measured_lift": g[g.treated == 1].converted_direct.mean() - g[g.treated == 0].converted_direct.mean(), "n": len(g)}), include_groups=False).reset_index()
    dec.to_csv(s.path("evaluation", "counterfactual_lift_by_decile.csv"), index=False)
    rep = {"n_test": int(len(d)), "n_targeted": int(d.target.sum()), "target_rate": round(float(d.target.mean()), 4),
           "treated_targeted": int(n_t), "holdout_targeted": int(len(C_)), "holdout_share_overall": round(float((d.treated == 0).mean()), 4),
           "conv_treated": round(float(T_.converted_direct.mean()), 4), "conv_holdout": round(float(C_.converted_direct.mean()), 4),
           "incremental_lift": round(float(lift), 4), "lift_ci95": [round(c, 4) for c in ci], "incremental_direct_bookings": round(float(incr), 1),
           "avg_saved_commission_usd": round(avg_sc, 2), "gross_saved_commission_usd": round(float(gross), 2), "cancellation_leakage_usd": round(float(leakage), 2),
           "incentive_cost_usd": round(cost, 2), "net_revenue_delta_usd": round(float(net), 2),
           "roi_x": round(float(net / cost), 2) if cost else None, "blast_all_net_revenue_delta_usd": round(float(blast_net), 2),
           "offer_mix": d.offer.value_counts().to_dict(), "decile_rank_corr_pred_vs_measured": round(float(dec[["pred_tau", "measured_lift"]].corr(method="spearman").iloc[0, 1]), 3)}
    s.path("evaluation", "offline_replay_roi.json").write_text(json.dumps(rep, indent=2))
    charts(rep, dec)
    print(json.dumps(rep, indent=2)); return rep


def charts(rep: dict, dec: pd.DataFrame):
    s = get_settings(); q = pd.read_csv(s.path("metrics", "qini_curve.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=220)
    ax.plot(q.frac * 100, q.qini, color=C["blue"], lw=2.4, label="X/T-learner uplift model")
    ax.plot(q.frac * 100, q.random, color=C["gray"], lw=1.6, ls="--", label="Random targeting")
    ax.fill_between(q.frac * 100, q.random, q.qini, color=C["blue"], alpha=.12)
    ax.set_title("Qini curve (held-out test set)", fontsize=14, color=C["ink"], loc="left"); ax.set_xlabel("% of guests targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative incremental direct bookings"); ax.grid(alpha=.25); ax.legend(frameon=False); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.tight_layout(); fig.savefig(s.path("charts", "qini_curve.png")); plt.close(fig)

    steps = [("Gross saved\ncommission", rep["gross_saved_commission_usd"]), ("Cancellation\nleakage", -rep["cancellation_leakage_usd"]), ("Incentive\ncost", -rep["incentive_cost_usd"])]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=220); cum = 0
    for i, (lab, v) in enumerate(steps):
        ax.bar(i, v, bottom=cum if v < 0 else 0, color=C["green"] if v > 0 else C["red"], width=.6); ax.text(i, (cum + v if v < 0 else v) + 2, f"${v:,.0f}", ha="center", fontsize=10, color=C["ink"]); cum += v
    ax.bar(len(steps), cum, color=C["blue"], width=.6); ax.text(len(steps), cum + 2, f"${cum:,.0f}", ha="center", fontsize=10, color=C["ink"], fontweight="bold")
    ax.set_ylim(min(0, cum) - 0.08 * abs(steps[0][1]), steps[0][1] * 1.18); ax.set_xticks(range(len(steps) + 1)); ax.set_xticklabels([l for l, _ in steps] + ["Net revenue\ndelta"]); ax.axhline(0, color=C["gray"], lw=.8)
    ax.set_title("Net margin waterfall (targeted guests, test replay)", fontsize=14, color=C["ink"], loc="left"); ax.set_ylabel("USD"); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.tight_layout(); fig.savefig(s.path("charts", "net_margin_waterfall.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=220)
    ax.plot(dec.decile + 1, dec.true_tau, "o-", color=C["gray"], label="True mean tau (synthetic oracle)"); ax.plot(dec.decile + 1, dec.pred_tau, "s-", color=C["blue"], label="Predicted mean tau")
    ax.plot(dec.decile + 1, dec.measured_lift, "^--", color=C["green"], label="Measured lift (treated - ghost holdout)")
    ax.set_title("Counterfactual lift by predicted-uplift decile", fontsize=14, color=C["ink"], loc="left"); ax.set_xlabel("Decile (10 = highest predicted uplift)"); ax.set_ylabel("Uplift in P(direct rebook)")
    ax.grid(alpha=.25); ax.legend(frameon=False); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.tight_layout(); fig.savefig(s.path("charts", "lift_by_decile.png")); plt.close(fig)


if __name__ == "__main__":
    run()
