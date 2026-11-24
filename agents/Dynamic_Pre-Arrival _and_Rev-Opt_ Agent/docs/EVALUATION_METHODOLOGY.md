# Evaluation methodology

## 1. XGBoost NBA model (`src/evaluation/model_eval.py`)
Held-out 25% of `offer_history`. Reported: ROC-AUC, PR-AUC (vs base rate), **KS** = max(TPR - FPR), **Brier** (vs a constant-base-rate predictor), decile lift, 10-bin quantile calibration curve, gain importances. Training device/`tree_method` are recorded in `evaluation/training_info.json` so GPU use is verifiable, not asserted.

## 2. Offline replay (`src/evaluation/offline_replay.py`)
N upcoming reservations (Platinum guests oversampled so the VIP path is exercised). For each: run the full graph, auto-resolve any HITL pause as "approve" (simulated concierge), then simulate the guest:
* accept with probability from the hidden behaviour model -> inbound "Yes" reply -> live inventory check -> folio posting;
* small share send a complaint; Platinum guests sometimes send a custom request (both must escalate).
Adversarial injection: ~10% of journeys poison the first draft (`once`), ~4% every draft (`persistent`) with a fabricated `$49`, `40%` and `SPA-FAKE9`.

## 3. Hallucination metric (target 0.0%)
`judge.deterministic_check` re-audits the **final delivered text** of every sent message: each `$` amount, `%` and `AAA-XXXX` SKU must belong to the offer payload. Rate = flagged messages / judged messages. Separately, `adversarial_sent_to_guest` counts poisoned drafts that reached a guest (must be 0) and `adversarial_catch_rate_pct` counts poisoned journeys that were retried, escalated or hard-stopped.

## 4. LLM-as-a-Judge (brand tone)
Qwen3-VL scores each sent message 1-5 against a rubric at temperature 0 (`tone_source: llm`); when the LLM is unavailable a deterministic heuristic is used and labelled `heuristic`. The tone judge is advisory; it never overrides the deterministic hallucination gate.

## 5. Revenue methodology
* **Baseline:** static `SPA-RELAX` at list price to every opted-in guest.
* **Expected ancillary revenue per room-night** = sum over sent messages of P(accept) x final price / room-nights, with P from the hidden behaviour model (variance-free, primary metric). Loyalty offers contribute avoided OTA commission (18% of room revenue) instead of ancillary revenue and are reported separately.
* **Realized** = Bernoulli draws of the same probabilities (noisy, secondary).
* **Margin** = P x (final price - cost).
Caveats: synthetic data; the agent deliberately suppresses messages (opt-out, caps, quiet hours, sold-out, low propensity), so per-room-night revenue can be below a blast-everyone baseline while margin per message and guest experience improve. The README reports the number as measured, whichever direction it points.

## 6. Reproducibility
Seeded generator (`--seed`), seeded replay sampling, deterministic temperature-0 judging; LLM drafts at temperature 0.4 vary run to run, which is why the gate evaluates the final text rather than trusting the prompt.
