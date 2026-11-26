"""GPU uplift meta-learners: X-Learner (and T-Learner baseline) + DragonNet for CATE tau_k(x).

Math (docs L3)
--------------
X-Learner:  mu_t(x) = E[Y|x,T=t];  D1 = Y1 - mu_0(X1),  D0 = mu_1(X0) - Y0;
            tau_t(x) = E[D_t | x];  tau(x) = g(x) tau_0(x) + (1-g(x)) tau_1(x),  g = e_hat (propensity).
T-Learner:  tau(x) = mu_1(x) - mu_0(x).
DragonNet:  shared Phi(x); heads e(Phi), mu_0(Phi), mu_1(Phi);
            L = BCE(y, mu_t) + alpha*BCE(t, e) + beta * MSE(y, mu_t + eps*h),   h = t/e - (1-t)/(1-e)   (targeted reg.)
            tau = mu_1 - mu_0.
Metric: sqrt(PEHE) = sqrt( mean (tau_hat - tau_true)^2 ).
Symbols: `g` = g(x), `eps` = epsilon, `alpha`,`beta` = DragonNet weights.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_device, load_config, path


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def _fit_mlp(net, X, y, loss_fn, cfg, dev):
    X = torch.as_tensor(X, dtype=torch.float32, device=dev); y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            loss = loss_fn(net(X[b]).squeeze(1), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def _pred(net, X, dev, sig=False):
    o = net(torch.as_tensor(X, dtype=torch.float32, device=dev)).squeeze(1)
    return (torch.sigmoid(o) if sig else o).cpu().numpy()


class XLearner:
    def __init__(self, n_features: int, cfg=None):
        self.cfg = cfg or load_config()["uplift"]; self.dev = get_device(); h = self.cfg["hidden"]; self.nf = n_features
        self.mu0, self.mu1, self.tau0, self.tau1, self.prop = (mlp(n_features, h, 1).to(self.dev) for _ in range(5))

    def fit(self, X, t, y):
        bce = F.binary_cross_entropy_with_logits; mse = F.mse_loss
        _fit_mlp(self.mu0, X[t == 0], y[t == 0], bce, self.cfg, self.dev)
        _fit_mlp(self.mu1, X[t == 1], y[t == 1], bce, self.cfg, self.dev)
        _fit_mlp(self.prop, X, t, bce, self.cfg, self.dev)
        d1 = y[t == 1] - _pred(self.mu0, X[t == 1], self.dev, True)
        d0 = _pred(self.mu1, X[t == 0], self.dev, True) - y[t == 0]
        _fit_mlp(self.tau1, X[t == 1], d1, mse, self.cfg, self.dev)
        _fit_mlp(self.tau0, X[t == 0], d0, mse, self.cfg, self.dev)
        return self

    def predict_cate(self, X):
        g = _pred(self.prop, X, self.dev, True).clip(0.02, 0.98)
        return g * _pred(self.tau0, X, self.dev) + (1 - g) * _pred(self.tau1, X, self.dev)

    def predict_t_learner(self, X):
        return _pred(self.mu1, X, self.dev, True) - _pred(self.mu0, X, self.dev, True)

    def state(self):
        return {n: getattr(self, n).state_dict() for n in ("mu0", "mu1", "tau0", "tau1", "prop")}

    def load_state(self, st):
        for n, s in st.items():
            getattr(self, n).load_state_dict(s); getattr(self, n).eval()
        return self


class DragonNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.e = nn.Linear(hidden, 1)
        self.m0 = mlp(hidden, hidden // 2, 1, 1); self.m1 = mlp(hidden, hidden // 2, 1, 1)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.phi(x)
        return self.e(z).squeeze(1), self.m0(z).squeeze(1), self.m1(z).squeeze(1)

    def cate(self, x):
        _, m0, m1 = self(x)
        return torch.sigmoid(m1) - torch.sigmoid(m0)


def fit_dragonnet(X, t, y, cfg=None, verbose=False) -> DragonNet:
    cfg = cfg or load_config()["uplift"]; dev = get_device()
    X_, t_, y_ = (torch.as_tensor(a, dtype=torch.float32, device=dev) for a in (X, t, y))
    net = DragonNet(X.shape[1], cfg["hidden"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    a, b = cfg["dragon_alpha"], cfg["dragon_beta"]
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(X_), device=dev)
        for i in range(0, len(X_), cfg["batch_size"]):
            ix = perm[i:i + cfg["batch_size"]]; x, tt, yy = X_[ix], t_[ix], y_[ix]
            e_l, m0, m1 = net(x)
            mu_t = torch.sigmoid(torch.where(tt == 1, m1, m0))
            e = torch.sigmoid(e_l).clamp(0.02, 0.98)
            h = tt / e - (1 - tt) / (1 - e)
            mu_tilde = (mu_t + net.eps * h).clamp(1e-4, 1 - 1e-4)                     # targeted regularisation
            loss = F.binary_cross_entropy(mu_t.clamp(1e-4, 1 - 1e-4), yy) + a * F.binary_cross_entropy_with_logits(e_l, tt) \
                + b * F.mse_loss(mu_tilde, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 5 == 0:
            print(f"  [dragonnet] ep{ep} loss={loss.item():.4f} eps={net.eps.item():.4f}")
    return net.eval()


@torch.no_grad()
def dragon_cate(net: DragonNet, X) -> np.ndarray:
    return net.cate(torch.as_tensor(X, dtype=torch.float32, device=next(net.parameters()).device)).cpu().numpy()


def sqrt_pehe(tau_hat, tau_true) -> float:
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))


def save_uplift(dragons: dict, xlearners: dict, n_features: int, name="uplift_models.pt"):
    torch.save({"n_features": n_features, "hidden": load_config()["uplift"]["hidden"],
                "dragon": {k: v.state_dict() for k, v in dragons.items()},
                "xlearner": {k: v.state() for k, v in xlearners.items()}}, path("checkpoints") / name)


def load_uplift(name="uplift_models.pt") -> tuple[dict, dict]:
    ck = torch.load(path("checkpoints") / name, map_location=get_device()); d, x = {}, {}
    for k, s in ck["dragon"].items():
        n = DragonNet(ck["n_features"], ck["hidden"]).to(get_device()); n.load_state_dict(s); d[k] = n.eval()
    for k, s in ck["xlearner"].items():
        x[k] = XLearner(ck["n_features"]).load_state(s)
    return d, x
