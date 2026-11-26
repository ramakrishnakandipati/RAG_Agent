"""Numerical checks of the mathematics in docs/MATHEMATICAL_FOUNDATIONS.md."""
import numpy as np
import pytest
from data.synthetic_generator import CATS, PRICE_MULTS, featurize, reward_tables, list_price
from evaluation.offline_replay import ips, doubly_robust, ess, qini_auuc, guardrail_mask
from models import convex_allocator as ca, elasticity_net as en
from models.utils import load_logs, load_config


@pytest.fixture(scope="module")
def df():
    return load_logs().head(6000)


def test_elasticity_strictly_monotone_in_price(df):
    m = en.load(); X = featurize(df.head(500)); grid = np.log(np.linspace(0.7, 1.3, 30)).astype(np.float32)
    for j in range(len(CATS)):
        P = np.stack([en.predict_np(m, X, np.full(len(X), j), np.full(len(X), g, dtype=np.float32)) for g in grid], 1)
        assert (np.diff(P, axis=1) <= 1e-6).all()


def test_allocator_kkt():
    rng = np.random.default_rng(0); n, K = 800, 3
    P = rng.uniform(0.1, 0.6, (n, K)); M = rng.uniform(20, 120, (n, K)); D = rng.uniform(0, 25, (n, K)); inv = [60, 90, 120]; B = 600.0
    r = ca.solve(P, M, D, inv, B, iters=2500)
    assert r["primal_violation_max"] < 5e-3 and r["comp_slackness_max"] < 0.5
    assert (r["lam_inv"] >= 0).all() and r["lam_budget"] >= 0
    assert r["objective"] <= ca.solve(P, M, D, [1e9] * K, 1e12)["objective"] + 1e-3         # constraints cannot raise value


def test_ips_dr_on_known_truth(df):
    """Uniform-random logging: mean IPS over repeated log draws -> truth (unbiased); DR with exact q is exact and low-variance (docs L9)."""
    P, M, E, mask = reward_tables(df); n = len(df); pi0 = 1.0 / mask.sum(1)
    target = np.where(mask, E, -1e9).argmax(1); truth = E[np.arange(n), target].mean(); q = np.where(mask, E, 0)
    ips_vals, dr_vals = [], []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        a_log = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
        r = np.where(a_log == 0, E[:, 0], (rng.random(n) < P[np.arange(n), a_log]) * M[np.arange(n), a_log]); w = (target == a_log) / pi0
        ips_vals.append(ips(r, w, clip=1e9)); dr_vals.append(doubly_robust(r, w, q[np.arange(n), target], q[np.arange(n), a_log], 1e9))
    assert abs(np.mean(ips_vals) - truth) < 0.05 * truth                      # unbiasedness
    assert abs(np.mean(dr_vals) - truth) < 0.01 * truth and np.std(dr_vals) < np.std(ips_vals)   # DR variance reduction
    assert ess(w) > 1


def test_qini_informative_score_beats_random():
    rng = np.random.default_rng(1); n = 20000; tau = rng.uniform(0, 0.4, n); t = rng.integers(0, 2, n); y = (rng.random(n) < 0.1 + tau * t).astype(int)
    good = qini_auuc(tau, t, y)["qini_coefficient"]; rand = qini_auuc(rng.random(n), t, y)["qini_coefficient"]
    assert good > 0 and good > 5 * abs(rand)


def test_vectorised_guardrail_matches_auditor(df):
    """The OPE action filter and the Auditor's deterministic contracts must agree (single source of truth)."""
    from agents.auditor_agent import audit
    from agents.tools import crm_get_guest
    from agents.schemas import UpsellOfferDraft
    cfg = load_config(); sub = df.head(40).reset_index(drop=True)
    mask = guardrail_mask(sub, PRICE_MULTS, CATS, cfg["catalog"], cfg["rms"]["min_upsell_floor"], cfg["rms"]["min_margin_ratio"], None)
    big = {key: 99 for key in ("DELUXE_KING", "JUNIOR_OCEAN_VIEW", "OCEANFRONT_SUITE", "SPA_PACKAGE", "DINING_EXPERIENCE", "LATE_CHECKOUT")}
    for i in range(len(sub)):
        g = crm_get_guest(sub.reservation_id[i]).model_copy(update={"contacts_7d": 0})
        for j, k in enumerate(CATS):
            for mi, m in enumerate(PRICE_MULTS):
                price = float(list_price(sub, k)[i] * m); a = 1 + j * len(PRICE_MULTS) + mi
                d = UpsellOfferDraft(reservation_id=g.reservation_id, sku=k, offer_price=price, list_price=price / m, subject="Hello", body_copy="Dear guest, a considered offer.", quant_rationale="x")
                v = audit(g, d, 0, 3, big); got = {x.code.value for x in v.violations}
                ok = not (got & {"BELOW_RMS_FLOOR", "MARGIN_BELOW_THRESHOLD", "SKU_ENTITLED_FREE", "INELIGIBLE_SKU"})
                assert ok == bool(mask[i, a]), (sub.reservation_id[i], k, m, got)
