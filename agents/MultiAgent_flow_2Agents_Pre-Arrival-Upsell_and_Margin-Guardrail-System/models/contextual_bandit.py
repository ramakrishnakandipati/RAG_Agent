"""GPU tensor-batched contextual bandits: LinUCB and Neural Thompson Sampling.

Math (docs L4)
--------------
Arm a (joint action: category x price multiplier, or no-offer), context x in R^d, reward r = realised margin
(= Y * (price - cost)), scaled by `reward_scale`.

LinUCB:   A_a = I + sum x x^T,  b_a = sum r x,  theta_a = A_a^{-1} b_a
          a_t = argmax_a  theta_a^T x + alpha * sqrt( x^T A_a^{-1} x )
          (single update  A^{-1} <- A^{-1} - A^{-1} x x^T A^{-1} / (1 + x^T A^{-1} x)  -- Sherman-Morrison;
           the batched update recomputes A^{-1} by Cholesky for all arms at once.)
Neural TS: phi(x) = penultimate layer of a network fit to logged margins; Bayesian linear head per arm
          theta_tilde ~ N(theta_hat, nu^2 A^{-1});  a_t = argmax_a theta_tilde_a^T phi(x).
Regret:   R_T = sum_t ( r*_t - r_{a_t,t} ),  r*_t = max_a E[r | x_t, a]   (oracle from the simulator).
Symbols: `Ainv` = A^{-1}, `alpha` = alpha, `nu` = nu.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .utils import get_device, load_config, path

REWARD_SCALE = 100.0


class BatchedLinUCB:
    def __init__(self, n_arms: int, d: int, alpha: float = 0.6, device=None):
        self.dev = device or get_device(); self.K, self.d, self.alpha = n_arms, d, alpha
        self.A = torch.eye(d, device=self.dev).repeat(n_arms, 1, 1)
        self.b = torch.zeros(n_arms, d, device=self.dev)
        self.Ainv = self.A.clone(); self.theta = torch.zeros(n_arms, d, device=self.dev)

    def scores(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:        # x [B,d] -> [B,K]
        mean = x @ self.theta.T
        var = torch.einsum("bd,kde,be->bk", x, self.Ainv, x).clamp_min(0)
        s = mean + self.alpha * var.sqrt()
        return s if mask is None else s.masked_fill(~mask, -1e9)

    def select(self, x, mask=None):
        return self.scores(x, mask).argmax(1)

    def update(self, x, arm, reward):
        oh = torch.nn.functional.one_hot(arm, self.K).float()                                  # [B,K]
        self.A += torch.einsum("bk,bd,be->kde", oh, x, x)
        self.b += oh.T @ (x * reward[:, None])
        L = torch.linalg.cholesky(self.A)
        self.Ainv = torch.cholesky_inverse(L)
        self.theta = torch.einsum("kde,ke->kd", self.Ainv, self.b)


class FeatureNet(nn.Module):
    """phi(x) with per-arm linear read-out; trained on logged (x, a, r)."""
    def __init__(self, d_in: int, n_arms: int, feat: int = 32, hidden: int = 96):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, feat), nn.Tanh())
        self.head = nn.Linear(feat, n_arms)

    def forward(self, x):
        return self.head(self.phi(x))


def pretrain_features(X, arms, rewards, n_arms, epochs=15, device=None) -> FeatureNet:
    dev = device or get_device()
    X, arms, r = (torch.as_tensor(a, device=dev, dtype=t) for a, t in ((X, torch.float32), (arms, torch.long), (rewards / REWARD_SCALE, torch.float32)))
    net = FeatureNet(X.shape[1], n_arms).to(dev); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), 2048):
            b = perm[i:i + 2048]
            loss = ((net(X[b]).gather(1, arms[b, None]).squeeze(1) - r[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


class NeuralThompson:
    def __init__(self, net: FeatureNet, n_arms: int, nu: float = 0.4):
        self.net, self.nu, self.K = net, nu, n_arms; self.dev = next(net.parameters()).device
        f = net.head.in_features
        self.lin = BatchedLinUCB(n_arms, f, 0.0, self.dev)

    @torch.no_grad()
    def select(self, x, mask=None):
        z = self.net.phi(x)                                                                    # [B,f]
        L = torch.linalg.cholesky(self.lin.Ainv + 1e-6 * torch.eye(z.shape[1], device=self.dev))
        eps = torch.randn(self.K, z.shape[1], device=self.dev)
        theta_t = self.lin.theta + self.nu * torch.einsum("kde,ke->kd", L, eps)                # theta ~ N(theta, nu^2 A^-1)
        s = z @ theta_t.T
        return (s if mask is None else s.masked_fill(~mask, -1e9)).argmax(1)

    @torch.no_grad()
    def update(self, x, arm, reward):
        self.lin.update(self.net.phi(x), arm, reward)


def simulate(policy_name: str, X, P, M, mask, n_rounds=4000, batch=64, seed=0, logged=None) -> dict:
    """Online simulation on ground-truth reward tables (P: accept prob, M: margin, mask: eligibility) [N,K]."""
    cfg = load_config()["bandit"]; dev = get_device(); g = torch.Generator(device="cpu").manual_seed(seed)
    n, K = P.shape; Xt = torch.as_tensor(X, device=dev); Pt, Mt, mk = (torch.as_tensor(a, device=dev) for a in (P, M, mask))
    Pt, Mt = Pt.float(), Mt.float()
    E = Pt * Mt; E_masked = E.masked_fill(~mk, -1e9); best = E_masked.max(1).values
    if policy_name == "linucb":
        pol = BatchedLinUCB(K, X.shape[1], cfg["linucb_alpha"], dev)
    elif policy_name == "neural_ts":
        net = pretrain_features(*logged, n_arms=K, device=dev); pol = NeuralThompson(net, K, cfg["ts_nu"])
    elif policy_name == "random":
        pol = None
    else:
        raise ValueError(policy_name)
    cum, regret, reward_hist = 0.0, [], []
    for t in range(n_rounds // batch):
        idx = torch.randint(0, n, (batch,), generator=g).to(dev); x = Xt[idx]; m = mk[idx]
        if pol is None:
            arm = torch.multinomial(m.float(), 1).squeeze(1)
        else:
            arm = pol.select(x, m)
        p = Pt[idx, arm]; r = torch.where(arm == 0, E[idx, 0], (torch.rand(batch, device=dev) < p).float() * Mt[idx, arm])
        if pol is not None:
            pol.update(x, arm, r / REWARD_SCALE)
        cum += (best[idx] - E[idx, arm]).sum().item(); regret.append(cum); reward_hist.append(r.mean().item())
    return {"policy": policy_name, "cumulative_regret": regret, "mean_reward": float(np.mean(reward_hist[-10:])),
            "final_regret": regret[-1], "rounds": len(regret) * batch, "_pol": pol}


def save_bandit(pol: BatchedLinUCB, name="linucb.pt"):
    torch.save({"A": pol.A.cpu(), "b": pol.b.cpu(), "alpha": pol.alpha}, path("checkpoints") / name)


def load_bandit(name="linucb.pt") -> BatchedLinUCB:
    ck = torch.load(path("checkpoints") / name, map_location="cpu"); K, d = ck["b"].shape
    p = BatchedLinUCB(K, d, ck["alpha"]); p.A = ck["A"].to(p.dev); p.b = ck["b"].to(p.dev)
    p.Ainv = torch.cholesky_inverse(torch.linalg.cholesky(p.A)); p.theta = torch.einsum("kde,ke->kd", p.Ainv, p.b); return p
