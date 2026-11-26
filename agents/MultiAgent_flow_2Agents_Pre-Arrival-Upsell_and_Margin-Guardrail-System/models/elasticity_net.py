"""Deep neural price-elasticity network with a monotone price-response layer (PyTorch / CUDA).

Math (docs L2)
--------------
Let h = trunk([emb(x), onehot(k)]) with emb(x)=W_e x (guest embedding), r = ln(p / p_list) the log price ratio.

    logit(x,k,p) = g(h) - sum_j  w_j(h) * phi_j(r),      w_j(h) = softplus(v_j(h)) >= 0
    phi_j(r)     = softplus( s * (r - kappa_j) )         (non-decreasing in r)
    P_hat        = sigmoid(logit)                          =>  dP_hat/dp < 0   for every x   (strict monotonicity)

Loss: BCE = -(1/N) sum [ y ln P + (1-y) ln(1-P) ].
Elasticity: eta = dlnP/dlnp = -(1-P) * sum_j w_j * s * sigmoid(s (r-kappa_j)).
Calibration: ECE = sum_b |B_b|/N * |acc(B_b) - conf(B_b)|.
Symbols: `w` = w_j, `knots` = kappa_j, `slope` = s, `g` = g(h).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_device, load_config, path


class ElasticityNet(nn.Module):
    def __init__(self, n_features: int, n_cats: int, hidden: int = 128, embed_dim: int = 16, n_basis: int = 8, slope: float = 6.0):
        super().__init__()
        self.n_cats = n_cats
        self.embed = nn.Linear(n_features, embed_dim)
        self.trunk = nn.Sequential(nn.Linear(embed_dim + n_cats, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.g = nn.Linear(hidden, 1)
        self.v = nn.Linear(hidden, n_basis)
        self.register_buffer("knots", torch.linspace(-0.4, 0.4, n_basis))
        self.slope = slope

    def forward(self, x, cat_idx, log_ratio):
        h = self.trunk(torch.cat([self.embed(x), F.one_hot(cat_idx, self.n_cats).float()], 1))
        w = F.softplus(self.v(h))                                           # >= 0
        phi = F.softplus(self.slope * (log_ratio[:, None] - self.knots[None]))
        return self.g(h).squeeze(1) - (w * phi).sum(1)                      # logit

    def predict(self, x, cat_idx, log_ratio):
        return torch.sigmoid(self.forward(x, cat_idx, log_ratio))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def calibration_curve(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    q = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, q) - 1, 0, bins - 1)
    return {"conf": [float(p[idx == b].mean()) for b in range(bins) if (idx == b).any()],
            "acc": [float(y[idx == b].mean()) for b in range(bins) if (idx == b).any()]}


def fit(X, cat_idx, log_ratio, y, tr, va, cfg=None, verbose=True):
    cfg = cfg or load_config()["elasticity"]; dev = get_device()
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    X_, c_, r_, y_ = T(X), T(cat_idx, torch.long), T(log_ratio), T(y)
    model = ElasticityNet(X.shape[1], int(cat_idx.max()) + 1, cfg["hidden"], cfg["embed_dim"], cfg["n_price_basis"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    tr_t, va_t = T(tr, torch.long), T(va, torch.long); hist = {"train": [], "val": []}; best, best_state = 1e9, None
    for ep in range(cfg["epochs"]):
        model.train(); perm = tr_t[torch.randperm(len(tr_t), device=dev)]; tot = 0.0
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            loss = F.binary_cross_entropy_with_logits(model(X_[b], c_[b], r_[b]), y_[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(model(X_[va_t], c_[va_t], r_[va_t]), y_[va_t]).item()
        hist["train"].append(tot / len(tr)); hist["val"].append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == cfg["epochs"] - 1):
            print(f"  [elasticity] ep{ep:02d} train={hist['train'][-1]:.4f} val={vl:.4f}")
    model.load_state_dict(best_state); model.eval()
    return model, hist


@torch.no_grad()
def predict_np(model: ElasticityNet, X: np.ndarray, cat_idx: np.ndarray, log_ratio: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=dev)
    return model.predict(t(X), t(cat_idx, torch.long), t(log_ratio)).cpu().numpy()


def save(model: ElasticityNet, n_features: int, n_cats: int, name="elasticity_net.pt"):
    cfg = load_config()["elasticity"]
    torch.save({"state": model.state_dict(), "n_features": n_features, "n_cats": n_cats, "cfg": cfg}, path("checkpoints") / name)


def load(name="elasticity_net.pt") -> ElasticityNet:
    ck = torch.load(path("checkpoints") / name, map_location=get_device())
    c = ck["cfg"]; m = ElasticityNet(ck["n_features"], ck["n_cats"], c["hidden"], c["embed_dim"], c["n_price_basis"]).to(get_device())
    m.load_state_dict(ck["state"]); return m.eval()


def optimal_price(model: ElasticityNet, X: np.ndarray, cat_idx: np.ndarray, p_list: np.ndarray, unit_cost: float,
                  floor: float, mults=(0.75, 0.875, 1.0, 1.125, 1.25), n_grid: int = 41):
    """Grid solve of  p* = argmax_{p >= floor} (p - c) * P_hat(x,p)   (docs L6). Returns (price, P_hat, margin) per row."""
    grid = np.linspace(min(mults), max(mults), n_grid)
    n = len(X); best = np.full(n, -1e9); bp = np.zeros(n); bP = np.zeros(n)
    for m in grid:
        price = p_list * m
        P = predict_np(model, X, cat_idx, np.full(n, np.log(m), dtype=np.float32))
        obj = np.where(price >= floor, (price - unit_cost) * P, -1e9)
        upd = obj > best
        best = np.where(upd, obj, best); bp = np.where(upd, price, bp); bP = np.where(upd, P, bP)
    return bp, bP, best
