# Architecture

## Design principle: probabilistic reasoning is fenced in by deterministic logic
| Concern | Owner | Why |
|---|---|---|
| Who should get which offer, at what price | XGBoost propensity (GPU) + RMS rules (`nba_rms_engine.py`) | auditable, testable, no free-text numbers |
| Property facts and brand voice | Hybrid RAG (Chroma dense + BM25, RRF, cross-encoder rerank) | grounded in `data/knowledge_base` |
| Copywriting, reply intent/sentiment | Qwen3-VL (`qwen3-vl-30b-a3b-instruct`, OpenAI-compatible, LM Studio at `:1234`) | language is what an LLM is for |
| Is the message safe to send | Deterministic gate (`compliance_tools.py`) | regex token audit: every `$`, `%`, SKU in the text must come from the offer payload |

The LLM never writes a price. It writes copy with placeholders (`{{OFFER_PRICE}}`, `{{OFFER_DISCOUNT}}`, ...). `compliance_check` renders them from the RMS offer, then audits the rendered text. A second audit runs inside `dispatch_and_log`, so human-edited and fallback drafts are covered too.

## Graph (6 nodes, every node wrapped by `instrument()` in `src/graph/nodes.py`)
`instrument` runs `pre_validate -> node -> post_validate -> telemetry` using `src/validation/step_validators.py`; a `ValidationFailure` becomes `status=FAIL_VALIDATION` and the graph ends, never sends.

| # | Node | Validators (pre / post) |
|---|---|---|
| 1 | `enrich_context` | ids present / guest+reservation exist and match, `check_out > check_in`, party >= 1, `precip_prob in [0,1]` |
| 2 | `retrieve_knowledge_and_offers` | context present / price = list x (1-disc), discount <= cap, margin >= floor, inventory > 0, propensity in [0,1], KB snippets present |
| 3 | `draft_message` | selected offer / non-empty subject + template |
| 4 | `compliance_check` | template / valid status |
| 5 | `hitl_gate` | guardrail result / decision recorded when paused |
| 6 | `dispatch_and_log` | PASS or human approval / terminal status |

### Routing (`src/graph/edges.py`)
* `PASS` -> `hitl_gate` -> `dispatch_and_log`
* `FAIL_POLICY` (hallucinated token, banned phrase, tone, price mismatch) -> back to `draft_message` with the violations as feedback, up to `MAX_RETRIES`; then `ESCALATE_HITL` with a safe fallback draft.
* `FAIL_VALIDATION` (opt-out, 7-day frequency cap, quiet hours, no inventory, malformed state) -> END, nothing sent.
* `hitl_gate` triggers: repeated policy failure, negative sentiment (< -0.3), Platinum guest with a custom request. It calls `langgraph.types.interrupt()`; state is persisted by `SqliteSaver` (`data/processed/checkpoints.sqlite`), so approvals survive a process restart. Resume with `Command(resume={decision, reviewer, edited_body?})`.

### Two-way replies
`trigger_type=inbound_reply` re-enters the same graph. `draft_message` classifies the reply (Qwen JSON + deterministic keyword safety net for complaints and custom requests), then chooses `confirm` / `ack_decline` / `answer` / `holding`. `confirm` re-checks live inventory; `dispatch_and_log` calls `book_ancillary_item`, which is a single `BEGIN IMMEDIATE` transaction (check capacity, decrement, post to folio, idempotent per reservation+SKU).

## GPU usage (`src/utils/device.py`)
* XGBoost: `tree_method="hist"`, `device="cuda"` for train and predict (CPU only if CUDA is absent).
* Embeddings: `bge-large-en-v1.5` in FP16 on CUDA; reranker `bge-reranker-large` FP16 on CUDA. If free VRAM < `MIN_FREE_VRAM_GB` the retriever drops to `bge-small`.
* Budget on a 24 GB card: Qwen3-VL-30B-A3B in LM Studio (~18-19 GB) + embedder + reranker (~2 GB) + XGBoost.

## Persistence
SQLite `pms.sqlite` (guests, reservations, inventory, folio, outbox, hitl_queue, audit), `checkpoints.sqlite` (LangGraph), Chroma persistent store in `data/chroma`, structured logs in `logs/events.jsonl` and `logs/telemetry.jsonl`.
