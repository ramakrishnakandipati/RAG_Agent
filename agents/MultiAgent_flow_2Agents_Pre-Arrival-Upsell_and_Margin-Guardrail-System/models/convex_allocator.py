"""Constrained convex budget & inventory allocator with GPU dual ascent -> shadow prices.

Math (docs L5)
--------------
Guests i=1..N, categories k=1..K, q_ik in [0,1] = offer intensity, sum_k q_ik <= 1 (one offer per guest).
  P_ik  acceptance prob.,  m_ik unit margin if accepted,  d_ik discount dollars given if accepted.

  max_q  sum_ik q_ik P_ik m_ik  +  tau * sum_i H(q_i)                       (entropy-smoothed => strictly concave)
  s.t.   sum_i q_ik P_ik <= I_k          [inventory]       multiplier  lambda_inv,k >= 0
         sum_ik q_ik P_ik d_ik <= B      [promo budget]    multiplier  lambda_B     >= 0

Per-guest closed form given duals:   q_i = softmax_{k, 0}( P_ik (m_ik - lambda_inv,k - lambda_B d_ik) / tau ),  (slot 0 = no offer, value 0).
Dual g(lambda) = sum_i tau * logsumexp(.) + lambda_inv . I + lambda_B B  is convex; dual gradient = cap - usage.
Projected gradient descent on lambda >= 0.   Hurdle rate:  h_ik = lambda_inv,k + lambda_B d_ik   (offer only if m_ik > h_ik).
KKT diagnostics: primal violation max(0, usage - cap);  complementary slackness  lambda * (cap - usage) ~ 0.
Symbols: `lam_inv` = lambda_inv, `lam_budget` = lambda_B, `tau` = tau.
"""
from __future__ import annotations
import numpy as np
import torch

from .utils import get_device, load_config


def solve(P, M, D, inventory, budget, iters: int | None = None, lr: float | None = None, tau: float = 6.0) -> dict:
    cfg = load_config()["allocator"]; iters = iters or cfg["iters"]; lr = lr or cfg["lr"]; dev = get_device()
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    P, M, D, I = t(P), t(M), t(D), t(inventory)
    lam = torch.zeros(P.shape[1] + 1, device=dev, requires_grad=False)       # [K inventory duals, 1 budget dual]
    cap = torch.cat([I, t([budget])]); scale = cap.clamp_min(1.0)
    ms = float(M.abs().mean())                              # margin scale ($) so lambda steps are in $ units
    for it in range(iters):
        v = P * (M - lam[:-1][None] - lam[-1] * D) / tau
        logits = torch.cat([torch.zeros(P.shape[0], 1, device=dev), v], 1)
        q = torch.softmax(logits, 1)[:, 1:]
        usage = torch.cat([(q * P).sum(0), (q * P * D).sum().reshape(1)])
        grad = (cap - usage) / scale                       # d g / d lambda (normalised)
        step = lr * (1.0 / (1 + it / 200))
        lam = torch.clamp(lam - step * ms * grad, min=0)
    v = P * (M - lam[:-1][None] - lam[-1] * D) / tau
    q = torch.softmax(torch.cat([torch.zeros(P.shape[0], 1, device=dev), v], 1), 1)[:, 1:]
    usage = torch.cat([(q * P).sum(0), (q * P * D).sum().reshape(1)])
    viol = torch.clamp(usage - cap, min=0) / scale
    cs = (lam * (cap - usage).abs() / scale)
    return {"q": q.cpu().numpy(), "lam_inv": lam[:-1].cpu().numpy(), "lam_budget": float(lam[-1]),
            "usage_inv": usage[:-1].cpu().numpy(), "usage_budget": float(usage[-1]),
            "primal_violation_max": float(viol.max()), "comp_slackness_max": float(cs.max()),
            "objective": float((q * P * M).sum()), "hurdle": (lam[:-1][None] + lam[-1] * D).cpu().numpy()}


def hurdle_table(res: dict, categories: list[str], discount_per_cat: dict | None = None) -> dict:
    """Compact shadow-price summary handed to the agents."""
    return {"lam_budget": res["lam_budget"], "lam_inv": {k: float(v) for k, v in zip(categories, res["lam_inv"])},
            "usage_inv": {k: float(v) for k, v in zip(categories, res["usage_inv"])}}
