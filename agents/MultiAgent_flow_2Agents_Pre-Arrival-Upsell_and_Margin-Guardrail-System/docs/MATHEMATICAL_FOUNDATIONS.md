# Mathematical Foundations

Every layer of the system has a formal specification here, a docstring in its module repeating the equations, and a numerical test
(`tests/test_models_math.py`, `tests/test_guardrails.py`, `tests/test_graph.py`) that checks the key identity.

## 0. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $x\in\mathbb R^{25}$ | guest context features $\phi(\text{guest})$ (numerics + one-hots) | `featurize()` |
| $k\in\{1..K\}$, $K=4$ | upsell category (ROOM_UPGRADE, SPA_PACKAGE, DINING_EXPERIENCE, LATE_CHECKOUT) | `CATS` |
| $m\in\mathcal M$ | price multiplier, $\mathcal M=\{0.75,0.875,1,1.125,1.25\}$; offer price $p=m\,p^{\text{list}}_k$ | `PRICE_MULTS` |
| $a\in\{0,\dots,20\}$ | joint action: $0$ = no offer, otherwise $(k,m)$ | `action` |
| $T,\;Y$ | treatment (offer received), outcome (purchase) | `treated`, `accepted` |
| $e(x)=P(T{=}1\mid x)$ | propensity | `propensity` |
| $\tau_k(x)$ | CATE of an offer on purchase probability | `tau_true_*`, `cate_dragon` |
| $c_k,\;h_k^{HK}$ | COGS and housekeeping cost | `catalog` in `config.yaml` |
| $\lambda_{inv,k},\lambda_B$ | shadow prices of inventory and promo budget | `lam_inv`, `lam_budget` |
| $R_{\max}=3$ | revision bound | `agents.max_revisions` |
| $\sigma(z)=1/(1+e^{-z})$ | sigmoid | |

---

## L1 — Synthetic data generation (`data/synthetic_generator.py`)

For guest $x$ with segment $s$, loyalty tier index $t\in[0,1]$, normalised past spend $u$ and prior stays $n$:

$$p_{0,k}(x)=\sigma\!\big(a^0_k+s^0_{k,s}+1.2\,t+0.9\,u+0.6\,n\big)\qquad\text{(organic purchase at list price)}$$

$$u_k(x,m)=\sigma\!\big(a^1_k+s^1_{k,s}+1.0\,t+0.9\,u-4\,\varepsilon_s\ln m\big)\qquad\text{(offer-driven take-up)}$$

$$P(Y{=}1\mid x,k,m,T{=}1)=p_{0,k}+(1-p_{0,k})\,u_k(x,m)\ \ge p_{0,k},\qquad \tau_k(x)=P(Y{=}1\mid x,k,1,T{=}1)-p_{0,k}(x).$$

$\varepsilon_s$ is the segment price-sensitivity (Leisure 1.6, Family 1.5, Honeymoon 0.9, Corporate 0.7, VIP 0.5), so acceptance is strictly decreasing in $m$.

**Confounded behaviour policy.** $\pi_0(k\mid x)=\operatorname{softmax}(\Gamma^\top x)$ with tier/spend loadings, so high-value guests are targeted more
(confounding for uplift); $m\sim\mathrm{Uniform}(\mathcal M)$ independently, hence the *logged joint propensity*

$$\pi_0(a{=}(k,m)\mid x)=\pi_0(k\mid x)/|\mathcal M|,\qquad \pi_0(0\mid x)=\pi_0(\text{no offer}\mid x).$$

**Reward with cannibalisation.** For tier-entitled guests $\mathbb 1[\text{ent}_k(x)]$ (e.g. Platinum/Diamond ask for the free 2 PM late checkout):

$$r=Y\cdot\big(p\,(1-\mathbb 1[\text{ent}_k(x)])-c_k-h^{HK}_k\big).$$

Closed-form expected rewards $E[r\mid x,a]=P\cdot M$ are available for ground-truth policy values (`reward_tables`).

---

## L2 — Deep monotone price-elasticity network (`models/elasticity_net.py`)

Trunk $h=\mathrm{MLP}([W_e x,\;\mathrm{onehot}(k)])$ with guest embedding $W_e x$; log price ratio $r=\ln(p/p^{\text{list}})$.

$$\text{logit}(x,k,p)=g(h)-\sum_{j=1}^{J}w_j(h)\,\phi_j(r),\quad w_j(h)=\mathrm{softplus}(v_j(h))\ge0,\quad \phi_j(r)=\mathrm{softplus}\!\big(s\,(r-\kappa_j)\big).$$

**Monotonicity proof.** $\phi_j'(r)=s\,\sigma(s(r-\kappa_j))>0$ and $w_j\ge0$, so $\partial\,\text{logit}/\partial r\le0$ and, because $\sigma$ is increasing,
$\partial\hat P/\partial p<0$ for *every* guest (verified on a 30-point price grid: 0 violations).

**Loss** $\mathcal L=-\tfrac1N\sum_i\big[y_i\ln\hat P_i+(1-y_i)\ln(1-\hat P_i)\big]$.

**Elasticity** (log-log slope): $\displaystyle\eta=\frac{\partial\ln\hat P}{\partial\ln p}=-(1-\hat P)\sum_j w_j\,s\,\sigma\!\big(s(r-\kappa_j)\big)$.

**Calibration** (expected calibration error, 10 bins): $\mathrm{ECE}=\sum_b\frac{|B_b|}{N}\,\big|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\big|$.

---

## L3 — Uplift meta-learners (`models/uplift_models.py`)

Potential-outcome setting, assuming unconfoundedness $(Y(0),Y(1))\perp T\mid x$ and overlap $0<e(x)<1$:
$\tau(x)=E[Y(1)-Y(0)\mid x]$.

**T-learner.** $\hat\tau(x)=\hat\mu_1(x)-\hat\mu_0(x)$ with $\mu_t(x)=E[Y\mid x,T{=}t]$.

**X-learner.** Impute individual effects
$$D^1_i=Y_i-\hat\mu_0(X_i)\ (T_i{=}1),\qquad D^0_i=\hat\mu_1(X_i)-Y_i\ (T_i{=}0),$$
regress $D^t$ on $x$ to get $\hat\tau_t(x)$, then blend with the propensity
$$\hat\tau(x)=g(x)\,\hat\tau_0(x)+\big(1-g(x)\big)\,\hat\tau_1(x),\qquad g=\hat e.$$
(The blend weights the effect model fitted on the *larger* arm more heavily.)

**DragonNet.** Shared representation $\Phi(x)$, heads $\hat e(\Phi),\hat\mu_0(\Phi),\hat\mu_1(\Phi)$:

$$\mathcal L=\underbrace{\mathrm{BCE}(y,\hat\mu_t)}_{\text{outcome}}+\alpha\,\underbrace{\mathrm{BCE}(t,\hat e)}_{\text{propensity}}+\beta\,\underbrace{\big\|y-\tilde\mu_t\big\|^2}_{\text{targeted reg.}},\qquad
\tilde\mu_t=\hat\mu_t+\epsilon\,h,\quad h=\frac{t}{\hat e}-\frac{1-t}{1-\hat e}.$$

$\epsilon$ is learned; the clever-covariate term $h$ pushes the fit towards the efficient-influence-function solution (doubly-robust flavour). $\hat\tau=\hat\mu_1-\hat\mu_0$.

**Metrics.** Because the simulator knows $\tau$: $\sqrt{\mathrm{PEHE}}=\sqrt{\tfrac1N\sum_i(\hat\tau_i-\tau_i)^2}$. On observed data: Qini / AUUC (L9).

---

## L4 — GPU batched contextual bandits (`models/contextual_bandit.py`)

Reward $r=Y\,(p-c)$ (scaled by 1/100), arm $a$ over the 21 joint actions, context $x$.

**LinUCB.**
$$A_a=I+\sum_{t:a_t=a}x_tx_t^\top,\quad b_a=\sum_{t:a_t=a}r_tx_t,\quad\hat\theta_a=A_a^{-1}b_a,\qquad a_t=\arg\max_a\ \hat\theta_a^\top x_t+\alpha\sqrt{x_t^\top A_a^{-1}x_t}.$$
Single-observation update by Sherman–Morrison:
$A^{-1}\leftarrow A^{-1}-\dfrac{A^{-1}xx^\top A^{-1}}{1+x^\top A^{-1}x}$. The GPU implementation updates *all arms for a whole batch at once*
($A\mathrel{+}=\sum_b \mathbb 1[a_b{=}a]\,x_bx_b^\top$) and recomputes $A^{-1}$ by batched Cholesky.

**Neural Thompson Sampling.** $\varphi(x)$ is the penultimate layer of a network fit to logged margins; each arm has a Bayesian linear head
$$\tilde\theta_a\sim\mathcal N\!\big(\hat\theta_a,\ \nu^2A_a^{-1}\big),\qquad a_t=\arg\max_a\ \tilde\theta_a^\top\varphi(x_t)$$
(sampled with the Cholesky factor of $A_a^{-1}$).

**Regret.** $R_T=\sum_{t\le T}\big(\max_aE[r\mid x_t,a]-E[r\mid x_t,a_t]\big)$, using simulator ground truth.

---

## L5 — Constrained convex allocator & shadow prices (`models/convex_allocator.py`)

Let $q_{ik}\in[0,1]$ be the offer intensity of category $k$ to guest $i$ with $\sum_kq_{ik}\le1$; $P_{ik}$ acceptance probability,
$m_{ik}$ unit margin when accepted, $d_{ik}$ discount dollars given when accepted.

$$\max_{q}\ \sum_{i,k}q_{ik}P_{ik}m_{ik}+\tau\sum_iH(q_i)\quad\text{s.t.}\quad
\underbrace{\sum_iq_{ik}P_{ik}\le I_k}_{\lambda_{inv,k}},\ \ \underbrace{\sum_{i,k}q_{ik}P_{ik}d_{ik}\le B}_{\lambda_B}.$$

The entropy term $H(q_i)=-\sum_{k\in\{0..K\}}q_{ik}\ln q_{ik}$ (slot 0 = no offer) makes the problem strictly concave, so the dual is smooth. Lagrangian per guest:

$$q_{i\cdot}=\operatorname{softmax}_{k\in\{0..K\}}\!\Big(\tfrac{1}{\tau}\,P_{ik}\big(m_{ik}-\lambda_{inv,k}-\lambda_Bd_{ik}\big)\Big),\ \text{ with value of slot 0 fixed at }0.$$

Dual $g(\lambda)=\tau\sum_i\ln\!\big(1+\sum_ke^{v_{ik}/\tau}\big)+\lambda_{inv}^\top I+\lambda_BB$ is convex with gradient $\partial g/\partial\lambda=\text{cap}-\text{usage}$.
**Projected gradient descent:** $\lambda\leftarrow[\lambda-\eta_t\,(\text{cap}-\text{usage})/\text{cap}]_+$ (GPU, 2000 iterations, $\eta_t\propto1/(1+t/200)$).

**Hurdle rate** handed to both agents: $h_{ik}=\lambda_{inv,k}+\lambda_Bd_{ik}$; an offer is economically worthwhile only if $m_{ik}>h_{ik}$.

**KKT diagnostics (tested):** primal feasibility $\max(0,\text{usage}-\text{cap})/\text{cap}\approx0$, dual feasibility $\lambda\ge0$,
complementary slackness $\lambda\cdot|\text{cap}-\text{usage}|\approx0$.
Interpretation: $\lambda_{inv,k}$ is the marginal profit of one more unit of inventory $k$ — the price of selling a scarce room today rather than at BAR.

---

## L6 — Strategist decision rule (`agents/strategist_agent.py`, `agents/tools.py`)

**Price.** For category $k$ the strategist solves
$$p_k^*=\arg\max_{p\ge\max(\text{floor}_k,\,p^{\min}_k)}\ (p-c_k)\,\hat P(x,k,p)$$
on a 41-point grid. Without the floor the first-order condition is $\hat P+(p-c)\partial_p\hat P=0\Rightarrow\ \frac{p-c}{p}=-\frac1\eta$ (Lerner rule): the markup is the inverse of the elasticity.

**Candidate score.**
$$S_k=\underbrace{\hat\tau_k(x)\,(p_k^*-c_k)}_{\text{incremental margin}}+\underbrace{100\,\alpha\sqrt{x^\top A_{a}^{-1}x}}_{\text{exploration bonus (\$)}}-\underbrace{(\lambda_{inv,k}+\lambda_Bd_k)}_{\text{shadow-price hurdle}}.$$
Candidates with $m_k\le h_k$ are dropped unless nothing viable remains. The LLM only writes the copy; price and SKU are always tool outputs.

---

## L7 — Auditor contracts (`agents/auditor_agent.py`)

Each contract is a deterministic predicate producing a coded violation with a machine-actionable remediation (`Directives`).

| # | Contract | Predicate (violation if false) | Remediation |
|---|---|---|---|
| 1 | PMS inventory buffer | $A_{live}\ge B=\max\big(B_{\min},\lceil z\,\sigma_D\sqrt L\rceil\big)$, $\sigma_D=0.07\cdot\text{capacity}$, $z=1.28$, $L=3$ | exclude SKU |
| 2a | RMS floor | $p\ge p^{RMS}_{floor,k}$ | $p^{\min}=\lceil\text{floor}\rceil$ |
| 2b | Contribution margin | $\mathrm{CM}/p=\dfrac{p-c-h^{HK}}{p}\ge\theta_m=0.30$ | $p^{\min}=\dfrac{c+h^{HK}}{1-\theta_m}$ |
| 3a | Loyalty parity | $\text{bundled\_perks}\cap\text{tier\_perks}(g)=\varnothing$; cannibalisation cost $=\sum_{j\in\cap}v_j$ | remove perks |
| 3b | Entitled SKU | $\text{perk}(k)\notin\text{tier\_perks}(g)$ | exclude SKU |
| 4 | Brand tone | no regex hit $\wedge\ s_{tone}\ge\theta_{tone}=0.7$ $\wedge$ len $\le L_{ch}$ | rewrite refined / shorten |
| 5 | Visual alignment (multimodal) | Qwen3-VL `aligned = true` | drop asset |
| 6 | Contact fatigue | $n_{contacts,7d}<n_{\max}=2$ | HITL |

Heuristic tone fallback: $s=\mathrm{clip}\big(1-0.3\,n_{hits}-0.2\,\mathbb 1[!]-0.2\,\mathbb 1[\text{caps}>0.18],\,0,\,1\big)$.

**Decision rule.**
$$\text{decision}=\begin{cases}\text{ESCALATE\_HITL}&\text{fatigue, or violations}\neq\varnothing\wedge r\ge R_{\max}\\ \text{REJECTED}&\text{violations}\ne\varnothing\wedge r<R_{\max}\\ \text{ESCALATE\_HITL}&\text{no violations}\wedge w_{tier}\cdot p\ge V_{VIP}\ (=\$250)\\ \text{APPROVED}&\text{otherwise}\end{cases}$$
with CLV weights $w_{tier}\in\{1.0,1.2,1.6,2.2,3.0\}$ (None→Diamond).

---

## L8 — Bounded Generator–Critic loop (`agents/graph.py`)

State $(r,\mathcal D_r)$ with revision counter $r$ and accumulated directives $\mathcal D_r=\mathcal D_{r-1}\sqcup\text{verdict}_r.\text{directives}$ (union of excluded SKUs/perks, per-SKU max of $p^{\min}$).

Transitions: REJECTED $\Rightarrow r\leftarrow r+1$ and back to the strategist; APPROVED $\Rightarrow$ finalize; ESCALATE_HITL $\Rightarrow$ `interrupt()`.

**Termination.** $r$ is a non-negative integer that strictly increases on every cycle, and the auditor escalates as soon as $r\ge R_{\max}$. Therefore at most
$R_{\max}+1$ drafts are produced and the graph always halts (tested for $R_{\max}\in\{0,1,2,3\}$).

**Monotone directives ⇒ convergence.** Since $\mathcal D_r$ only grows, a violation that was remediated deterministically (floor, perk, SKU exclusion) cannot recur; only stochastic LLM tone faults can persist,
which is exactly what the escalation bound protects against.

**Pass rate** after $r$ rounds, with per-round pass probabilities $\rho_j$: $P(\text{pass by }r)=1-\prod_{j\le r}(1-\rho_j)$; the empirical first-pass and final-pass rates are reported in the dashboard.

---

## L9 — Offline policy evaluation (`evaluation/offline_replay.py`)

Logged tuples $(x_i,a_i,r_i,\pi_0(a_i\mid x_i))$; target $\pi$ (deterministic ⇒ $\pi(a\mid x)=\mathbb 1[a=\pi(x)]$). Importance weights $w_i=\pi(a_i\mid x_i)/\pi_0(a_i\mid x_i)$.

$$\hat V_{IPS}=\frac1N\sum_i\min(w_i,M)\,r_i,\qquad
\hat V_{SNIPS}=\frac{\sum_i\min(w_i,M)\,r_i}{\sum_i\min(w_i,M)}$$

$$\hat V_{DR}=\frac1N\sum_i\Big[\hat q^\pi(x_i)+\min(w_i,M)\big(r_i-\hat q(x_i,a_i)\big)\Big],\qquad \hat q^\pi(x)=\sum_a\pi(a\mid x)\hat q(x,a).$$

* **Unbiasedness.** $E[w\,r]=E_\pi[r]$ when $\pi_0>0$ wherever $\pi>0$ (IPS); DR is unbiased if *either* $\pi_0$ or $\hat q$ is correct, and its variance is reduced when $\hat q$ is accurate (both verified numerically on the simulator).
* **Clipping at $M=50$** trades a small downward bias for variance — visible as IPS < truth in the results.
* **Effective sample size** $\mathrm{ESS}=\big(\sum_iw_i\big)^2/\sum_iw_i^2$; bootstrap 95 % CIs on $\hat V_{DR}$.
* **Ground truth** $V(\pi)=\frac1N\sum_iE[r\mid x_i,\pi(x_i)]$ and **regret** $\mathrm{Reg}(\pi)=\frac1N\sum_i\big(\max_aE[r\mid x_i,a]-E[r\mid x_i,\pi(x_i)]\big)$.

**Qini / AUUC.** Sort guests by predicted uplift; with $n_t(\phi),n_c(\phi)$ the treated/control counts and $Y_t(\phi),Y_c(\phi)$ the conversions in the top-$\phi$ fraction:

$$Q(\phi)=\frac{1}{N}\Big[Y_t(\phi)-Y_c(\phi)\,\frac{n_t(\phi)}{n_c(\phi)}\Big],\qquad
\mathrm{AUUC}=\int_0^1Q(\phi)\,d\phi,\qquad
\text{Qini}=\int_0^1\big(Q(\phi)-\phi\,Q(1)\big)\,d\phi.$$

The Qini coefficient is the area between the model curve and the random-targeting diagonal.

**Guardrail-filtered policy.** $\pi_{ML+G}(x)=\arg\max_{a\in\mathcal A_{ok}(x)}\hat q(x,a)$ where
$\mathcal A_{ok}(x)=\{a:\ p_a\ge\text{floor},\ \mathrm{CM}/p\ge\theta_m,\ \neg\text{entitled},\ \text{eligible}\}\cup\{0\}$ (vectorised replay of contracts 2–3; a test asserts it equals the Auditor).

---

## L10 — GPU telemetry (`models/utils.py`)

$$\text{latency}=(t_1-t_0)\ \text{ms after } \texttt{cuda.synchronize()},\qquad \text{throughput}=N/\text{latency},\qquad \text{VRAM}=\texttt{max\_memory\_allocated}/2^{20}\ \text{MiB}.$$

Logged per training/inference call to `outputs/logs/gpu_telemetry.jsonl` (`timed()` context manager).
