# Hospitality Marketing RAG

Hybrid (dense + BM25 + RRF + cross-encoder) RAG over synthetic hospitality case studies, KPI sheets, campaign
reports and brand guidelines, grounded by a local **Qwen3-Coder-30B-A3B-Instruct** (OpenAI-compatible endpoint,
e.g. LM Studio at `http://localhost:1234/v1`).

## Pipeline (9 stages, `src/pipeline.py`)
| # | Stage | Module |
|---|-------|--------|
| 1 | Ingest PDF / MD / CSV / XLSX (pymupdf + pdfplumber + pandas) | `src/ingest/parser.py` |
| 2 | Chunk (512/64 tokens), **tables never split**, row linearization, 8-field Pydantic metadata | `src/ingest/chunker.py` |
| 3 | Embed on GPU (`BAAI/bge-large-en-v1.5`, FP16) | `src/indexing/embedder.py` |
| 4 | Dual index: persistent ChromaDB + pickled BM25Okapi, both honouring metadata filters | `src/indexing/vector_sparse_store.py` |
| 5 | Query rewrite: <=3 queries, acronym expansion, filter extraction | `src/retrieval/query_rewriter.py` |
| 6 | Hybrid retrieve + Reciprocal Rank Fusion | `src/retrieval/hybrid_retriever.py` |
| 7 | Rerank top-25 -> top-5 (`BAAI/bge-reranker-large`, FP16) | `src/retrieval/reranker.py` |
| 8 | Grounded answer with `[source \| page \| period]` citations | `src/generation/generator.py` |
| 9 | Verify: numeric claim matcher + citation check + LLM judge -> faithfulness 0..1 | `src/generation/verifier.py` |

## Setup (Windows, Python 3.11, CUDA GPU)
```powershell
conda activate rag-agent-gpu            # created by ../setup.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Load `qwen3-coder-30b-a3b-instruct` in LM Studio and start its server (port 1234). Endpoint/model are in
`configs/config.yaml` (`llm.*`). Set `llm.mode: mock` (or `RAG_LLM_MODE=mock`) to run without the LLM.

## Run
```powershell
python scripts/run_end_to_end.py          # data -> index -> queries -> metrics/charts -> pytest -> screenshots
python scripts/run_end_to_end.py --mock   # same, deterministic mock LLM
pytest                                    # fast suite (mock LLM; GPU models are real)
pytest --live-llm                         # also exercise the real Qwen endpoint
uvicorn api.main:app --port 8000          # /health /ingest /query /verify  (docs at /docs)
```

## Streamlit UI
```powershell
python scripts/run_ui.py                      # starts FastAPI (:8000) + Streamlit (:8501), warms the GPU models
python scripts/run_ui.py --mock               # without the Qwen server
python scripts/capture_ui_screenshots.py      # drives the real UI/Swagger with Chrome -> outputs/screenshots/ui_*.png, api_*.png
```
The UI is a thin client over the API (`ui/api_client.py`); tabs: Ask, Evidence, Pipeline, Metrics, History.
Sidebar filters are sent as explicit overrides that win over filters extracted from the question.

New endpoints: `GET /meta/filters`, `GET /meta/examples`, `GET /meta/metrics`, `GET /chunks/{chunk_id}`;
`POST /query` now accepts optional `filters` and `top_k`.

## Design notes
- **Tables**: the chunk text is the full markdown table plus appended one-sentence-per-row summaries
  (`Miami Beach Resort: Q3 2025, Direct channel, ADR (Average Daily Rate) $182, Occupancy 74%, ...`). The dense
  embedder sees section + row sentences only, so long tables are never truncated mid-table for retrieval.
- **Filters**: `period`/`property`/`doc_type` are exact matches (`only Q3 2025` -> 0 chunks from other quarters,
  enforced in Chroma `where`, BM25 masking and a post-hoc guard). `channel`/`segment` also match chunks tagged
  `all` (multi-channel tables).
- **Refusals**: unknown place names (e.g. Tokyo), nonexistent periods, or weak rerank evidence short-circuit to a
  refusal without calling the LLM; the LLM prompt also instructs refusal.
- **Verifier**: numbers ($, %, x) must appear in the retrieved chunks; citations must reference a retrieved
  chunk; the LLM judge's flags are cross-checked by lexical overlap so paraphrases are not penalised. Derived
  numbers (differences/sums) are *not* accepted - the generator prompt forbids computing them.
  `faithfulness = 1 - (unsupported numbers + judge flags + citation errors) / (all of those checked)`.
- **VRAM**: embedder (~0.7 GB) + reranker (~1.1 GB) share the GPU with Qwen in LM Studio; models load lazily.

## Outputs (`outputs/`, git-ignored)
`logs/`, `stage_artifacts/` (JSON per stage), `metrics/` (`retrieval_metrics.json`, `latency_breakdown.csv`,
`test_report.json`), `visualizations/` (latency waterfall, retrieval comparison, embedding UMAP),
`screenshots/` (5 PNG audit cards).
