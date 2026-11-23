# Hospitality Marketing RAG: Walkthrough

A guided tour of the system with real screenshots, all captured from live runs against a local
Qwen3-Coder-30B-A3B model (LM Studio) and the GPU embedder and reranker. Screenshots live in
[`outputs/screenshots/`](../outputs/screenshots/).

- [1. Start the app](#1-start-the-app)
- [2. Using the UI](#2-using-the-ui)
- [3. The API](#3-the-api)
- [4. Under the hood](#4-under-the-hood)
- [5. Reproduce everything](#5-reproduce-everything)
- [6. Known limitations](#6-known-limitations)

---

## 1. Start the app

```powershell
conda activate rag-agent-gpu
cd hosiptality_marketing_rag
python scripts/run_end_to_end.py     # one time: generates data, builds the index, writes metrics
python scripts/run_ui.py             # API on :8000, UI on :8501 (add --mock to skip the Qwen server)
```

Open <http://localhost:8501>. The sidebar badge shows the API state, the number of indexed chunks, the LLM mode
(live or mock) and whether the GPU is in use.

![Home screen](../outputs/screenshots/ui_01_home_health.png)

The sidebar has three things:
- **Filters** (Property, Period, Channel): explicit overrides that win over anything extracted from the question.
- **Evidence chunks (top-k)**: how many reranked chunks reach the answer generator.
- **Examples**: one-click sample questions.

---

## 2. Using the UI

### 2.1 Ask a grounded KPI question

Click **Miami Direct ADR (grounded KPI)** and press **Ask**. The question is
*"What was the ADR for the Direct channel at Miami Beach Resort in Q3 2025?"*

![Grounded answer](../outputs/screenshots/ui_02_ask_grounded_answer.png)

What to look at:
- **Answer card**: each fact is followed by a citation chip `source · page · period`.
- **Claim table** (right): every number in the answer (`$182`, `$167`, `$134.68`) and the chunk that supports it.
- **Filters applied**: `period`, `property` and `channel` were extracted from the question automatically.
- **Faithfulness gauge**: green at 0.95 or above, amber at 0.75 or above, red below that.

> This capture scored **0.83**. All four numbers (`$182`, `$182`, `$167`, `$134.68`) are verified, but the LLM
> judge flagged one sentence ("...the highest RevPAR of $134.68...") as a comparison it could not match to a
> single chunk. That is 1 flagged check out of 6 (4 numbers, 1 non-numeric sentence, 1 flagged sentence), shown
> as an amber "Judge flagged" box. Live answers vary between runs: the same question sometimes returns without
> that sentence and scores **1.00**.

### 2.2 Inspect the evidence

The **Evidence** tab shows how the question was rewritten and what was retrieved.

![Evidence tab](../outputs/screenshots/ui_03_evidence_tab.png)

- **Rewritten search queries**: at most 3, standalone, with acronyms expanded (`ADR` becomes *average daily rate*).
- **Top chunks after reranking**: each with its cross-encoder score and metadata chips. KPI table chunks open to the
  **whole source table** (tables are never split) and the row-by-row sentences that were indexed.
- Scores fall off quickly (0.999, 0.999, 0.987, 0.780, 0.011), so the reranker is clearly separating the relevant
  tables from filler text.

### 2.3 See where the time goes

![Pipeline tab](../outputs/screenshots/ui_04_pipeline_latency.png)

Retrieval and reranking take about 0.2 s together. The three LLM calls (rewrite, answer, verify) are about 99% of
the total, so latency is a function of the model, not the search.

### 2.4 Combine sidebar filters with the question

Set **Property = Orlando Grand Hotel**, click **RevPAR by channel, only Q3 2025**, press **Ask**.

![Filter isolation](../outputs/screenshots/ui_05_filter_isolation.png)

Every retrieved chunk is Orlando and Q3 2025. The question text contributes the period; the sidebar contributes
the property. Filters apply at three layers (Chroma `where`, BM25 masking, and a post-retrieval guard), and the test
suite asserts zero chunks from other quarters.

### 2.5 Out-of-scope questions are refused

Click **Tokyo ADR (out of scope, refused)**.

![Out of scope](../outputs/screenshots/ui_06_out_of_scope_refusal.png)

No Tokyo property exists in the corpus. A rule-based scope guard refuses before the answer LLM is called (a few
seconds instead of about 30), so nothing is invented.

### 2.6 Narrative (non-numeric) questions

Click **CRM and LTV strategy (narrative)**.

![Narrative answer](../outputs/screenshots/ui_09_narrative_crm_ltv.png)

The answer cites text chunks instead of tables, so there are no numbers to verify; the score comes from the
judge's sentence-level check. This capture scored **0.75**: the judge flagged one of four bullets ("The measurable
lift in repeat stays demonstrates the effectiveness of this LTV-based targeting approach"), a conclusion that goes
beyond what the cited chunk states.

> **Scoring fix.** An earlier version of the score only counted items that were flagged, so this same situation,
> one judge-flagged sentence among four bullets, showed **0.00**. Every non-numeric sentence now counts as a
> check, so one flag among four bullets costs 0.25 (score 0.75). This is covered by unit tests in
> `tests/test_04_verification.py`.

### 2.7 Try the verifier yourself

On the **Ask** tab, open **Try the verifier: inject a fabricated number**. The first dollar amount in the answer is
replaced with `$299`. Press **Verify against the retrieved evidence**.

![Verifier demo](../outputs/screenshots/ui_08_verifier_demo.png)

`$299` is **BLOCKED** (it appears in none of the retrieved chunks) while `74%` stays verified, and the score
drops to **0.50**. You can edit the text freely to test any claim.

### 2.8 Metrics dashboard

![Metrics dashboard](../outputs/screenshots/ui_07_metrics_dashboard.png)

Retrieval quality across four systems on 12 golden questions (BM25 only, dense only, RRF hybrid, RRF plus
cross-encoder rerank), mean faithfulness, test results, and latency per benchmark query. The numbers come from
`outputs/metrics/` via `GET /meta/metrics`.

---

## 3. The API

The UI is a thin client over FastAPI. Interactive docs: <http://localhost:8000/docs>.

![Swagger overview](../outputs/screenshots/api_05_swagger_all_routes.png)

| Endpoint | Purpose |
|---|---|
| `GET /health` | API state, chunk count, LLM mode, CUDA |
| `POST /ingest` | Rebuild the index |
| `POST /query` | Full pipeline: answer, citations, evidence, verification, latency |
| `POST /verify` | Verify any answer text against given chunk ids |
| `GET /meta/filters` | Filter options for dropdowns |
| `GET /meta/examples` | Curated sample questions |
| `GET /meta/metrics` | Latest retrieval, latency and test metrics |
| `GET /chunks/{chunk_id}` | Full text and source table of one chunk |

### `GET /meta/filters`: filter vocabulary

![meta filters](../outputs/screenshots/api_06_meta_filters.png)

```bash
curl http://localhost:8000/meta/filters
```

### `GET /meta/examples`: sample questions

![meta examples](../outputs/screenshots/api_07_meta_examples.png)

```bash
curl http://localhost:8000/meta/examples
```

### `GET /meta/metrics`: dashboard data

![meta metrics](../outputs/screenshots/api_08_meta_metrics.png)

```bash
curl http://localhost:8000/meta/metrics
```

### `POST /query` with filter overrides

`filters` (keys: `property`, `period`, `channel`, `segment`, `doc_type`) win over filters extracted from the
question. `top_k` (1 to 10) limits the evidence chunks.

![query with filters](../outputs/screenshots/api_09_query_with_filter_override.png)

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{
  "question": "Show RevPAR by channel",
  "filters": {"period": "Q3 2025", "property": "Orlando Grand Hotel"},
  "top_k": 3
}'
```

Unknown filter keys or `top_k` outside 1 to 10 return `422`.

### `GET /chunks/{chunk_id}`: view a source chunk

![chunk by id](../outputs/screenshots/api_10_chunk_by_id.png)

```bash
curl "http://localhost:8000/chunks/case_study_miami_beach_resort.md::p1::t31"
```

Unknown ids return `404`.

---

## 4. Under the hood

The nine pipeline stages (ingest, chunk, embed, dual index, rewrite, hybrid retrieve, rerank, answer, verify)
each write an artifact to `outputs/stage_artifacts/`. These screenshots are generated by
`scripts/run_end_to_end.py`.

**Table linearization.** A KPI table is kept whole and each row also becomes a natural-language sentence, so both
dense and sparse retrievers can match a specific metric.

![Table linearization](../outputs/screenshots/02_table_linearization_inspect.png)

**Query rewrite audit.** At most 3 queries, acronym expansion, extracted filters.

![Query rewrite](../outputs/screenshots/03_query_rewrite_audit.png)

**Hallucination guard.** The same answer with a fabricated `$299` is blocked and the score drops from 1.00 to 0.50.

![Verification card](../outputs/screenshots/04_verification_hallucination_guard.png)

**Test suite.**

![pytest](../outputs/screenshots/01_pytest_suite_terminal.png)

**`/query` response payload.**

![FastAPI response](../outputs/screenshots/05_fastapi_endpoint_response.png)

**Charts** (`outputs/visualizations/`):

| Retrieval comparison | Latency waterfall |
|---|---|
| ![retrieval](../outputs/visualizations/retrieval_comparison.png) | ![latency](../outputs/visualizations/latency_waterfall.png) |

![Embedding projection](../outputs/visualizations/embedding_umap.png)

---

## 5. Reproduce everything

```powershell
python scripts/run_end_to_end.py            # metrics, charts, pytest, the five pipeline screenshots
python scripts/capture_ui_screenshots.py    # drives the real UI and Swagger with Chrome (ui_*.png, api_*.png)
pytest                                      # full suite (mock LLM); add --live-llm for the live tests
```

`capture_ui_screenshots.py` uses Playwright with your installed Chrome and needs internet access for the Swagger
page assets. Use `--only ui`, `--only api` or `--only metrics` to recapture part of the set.

---

## 6. Known limitations

- **Faithfulness formula.** `1 - bad / checks`, where checks are every numeric claim, every non-numeric answer
  sentence and every citation error (plus one unit for a judge flag on a sentence already counted through its
  numbers). Numbers are checked strictly by regex against the retrieved chunks; sentences rely on the LLM judge.
- **LLM judge noise.** The judge sometimes flags valid comparisons ("highest RevPAR"). A lexical-overlap
  cross-check removes some, not all, of these. In the latest live run every score below 1.00 (0.83 and 0.75) was
  a single judge flag with no unsupported number and no citation error.
- **Latency.** A live answer takes about 17 to 31 s because three LLM calls dominate. The first query after
  starting the API also loads the GPU models; `run_ui.py` warms them up for you.
- **Small synthetic corpus.** 58 chunks, 12 golden questions. Hybrid retrieval already scores 1.00 on the golden
  set, so reranking cannot show a gain over it here.
- **Model-computed numbers are rejected.** Differences or totals the LLM calculates on its own are not in the
  context, so they are flagged as unsupported by design.
