const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("C:/Users/RamaKrishna/.claude/skills/synced/c4087e75-6e8c-4493-8472-1adcd2a2603e_2c02675d-fb05-4126-b416-349c05b8005e/pptx/scripts/apply_theme.js");

const ROOT = "D:/Ramakrishna_D/Python/GitProjects/RAG_Agent/hosiptality_marketing_rag";
const OUT = ROOT + "/presentation/Hospitality_RAG_Presentation.pptx";
const metrics = JSON.parse(fs.readFileSync(ROOT + "/outputs/metrics/retrieval_metrics.json", "utf8"));
const sample = JSON.parse(fs.readFileSync(ROOT + "/presentation/sample_run.json", "utf8"));
const lat = fs.readFileSync(ROOT + "/outputs/metrics/latency_breakdown.csv", "utf8").trim().split(/\r?\n/).slice(1)
  .map(l => { const m = l.match(/^(?:"([^"]*)"|([^,]*)),(.*)$/); const n = m[3].split(",").map(Number); return { q: m[1] || m[2], total: n[5] }; });

const THEME = {
  name: "Hospitality RAG", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2933", lt1: "FFFFFF", dk2: "0F3D3E", lt2: "EEF3F2", accent1: "0F766E", accent2: "C8793B",
    accent3: "5B8E7D", accent4: "8A6F3D", accent5: "B42318", accent6: "5B6B7A", hlink: "0F766E", folHlink: "5B6B7A" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Hospitality Marketing RAG";
pres.author = "RAG Agent project";
const C = pres.SchemeColor;
const MONO = "Courier New";

pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 8.8, h: 0.9, fontSize: 36, bold: true, color: C.background1, valign: "middle" }, text: "" } }],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.background2 },
});
pres.defineSlideMaster({
  title: "LIGHT", background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.7, fontSize: 26, bold: true, color: C.text2, valign: "middle" }, text: "" } },
    { text: { text: "Hospitality Marketing RAG", options: { x: 0.5, y: 5.2, w: 4, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6 },
});

const shadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 2, angle: 90, opacity: 0.12 });
let current = "Overview";
function addSlide(master, section, notes) {
  if (section !== current) { pres.addSection({ title: section }); current = section; }
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (notes) s.addNotes(notes);
  return s;
}
pres.addSection({ title: "Overview" });

function card(s, x, y, w, h, name, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { color: fill || C.background2, width: 0 }, shadow: shadow(), objectName: name });
}
function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true }, o));
}
function bullets(s, items, x, y, w, h, size = 14, color) {
  const arr = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  s.addText(arr, { x, y, w, h, fontSize: size, color: color || C.text1, margin: 0, valign: "top", isTextBox: true });
}
function badge(s, n, x, y, d, fill, name) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { color: fill || C.accent1, width: 0 }, objectName: name || `Badge ${n}` });
  s.addText(String(n), { x, y, w: d, h: d, fontSize: Math.round(d * 30), bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, big, label, name, fill) {
  card(s, x, y, w, 1.65, name, fill);
  txt(s, big, x + 0.15, y + 0.15, w - 0.3, 0.7, { fontSize: 30, bold: true, color: C.accent1, fontFace: "Cambria", valign: "middle" });
  txt(s, label, x + 0.15, y + 0.9, w - 0.3, 0.7, { fontSize: 12, color: C.accent6 });
}
const chartBase = (title) => ({
  showTitle: true, title, titleFontSize: 14, titleColor: "0F3D3E", titleFontFace: "+mn-lt",
  catAxisLabelColor: "475569", valAxisLabelColor: "475569", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: "1F2933",
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: "475569",
  valGridLine: { color: "D5DEDC", size: 0.5 }, catGridLine: { style: "none" }, showValue: true,
});

// ------------------------------------------------------------------ 1 title
{
  const s = addSlide("DARK", "Overview", "Opening: a hybrid RAG system for hospitality marketing questions. Everything shown comes from the real run in this repository.");
  txt(s, "Hospitality Marketing RAG", 0.6, 1.3, 5.8, 1.6, { fontSize: 44, bold: true, color: C.background1, fontFace: "Cambria", valign: "top" });
  txt(s, "Hybrid retrieval, cross-encoder reranking and verified, cited answers", 0.6, 3.05, 5.2, 0.9, { fontSize: 18, color: C.background2 });
  txt(s, "9-stage pipeline  |  local Qwen3-Coder-30B-A3B  |  GPU embeddings and reranker", 0.6, 4.35, 5.6, 0.6, { fontSize: 14, color: C.accent3 });
  let n = 1;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    badge(s, n, 6.9 + c * 0.85, 1.25 + r * 0.85, 0.65, (n % 2 ? C.accent1 : C.accent2), `Stage motif ${n}`); n++;
  }
}

// ------------------------------------------------------------------ 2 exec summary
{
  const s = addSlide("LIGHT", "Overview", "Headline numbers. All metrics come from the live run with Qwen3-Coder-30B and the GPU models.");
  s.addText("What we built and what it achieved", { placeholder: "title" });
  const m = metrics.retrieval;
  stat(s, 0.5, 1.25, 2.1, "58", "chunks, 24 intact KPI tables", "Stat chunks");
  stat(s, 2.75, 1.25, 2.1, m["rrf+rerank"]["mrr@5"].toFixed(2), `MRR@5 with rerank (BM25: ${m.bm25["mrr@5"].toFixed(2)})`, "Stat mrr");
  stat(s, 5.0, 1.25, 2.1, metrics.mean_faithfulness.toFixed(2), "mean faithfulness, 6 live queries", "Stat faithfulness");
  stat(s, 7.25, 1.25, 2.25, "57 / 57", "tests passing, 75% coverage", "Stat tests");
  card(s, 0.5, 3.15, 9, 1.8, "Summary card");
  bullets(s, [
    "Every claim carries a [source | page | period] citation",
    "Numbers are checked against retrieved chunks; fabricated metrics are blocked",
    "Unanswerable questions (for example Tokyo) are refused, not guessed",
    "Filters like \"only Q3 2025\" are exact, with zero leakage from other quarters",
  ], 0.75, 3.3, 8.5, 1.5, 14);
}

// ------------------------------------------------------------------ 3 flow
{
  const s = addSlide("LIGHT", "Overview", "Offline stages run once per corpus. Online stages run per question.");
  s.addText("Overall flow: nine stages, two phases", { placeholder: "title" });
  const box = (n, name, sub, x, y, w, fill) => {
    card(s, x, y, w, 1.15, `Stage ${n}`, fill || C.background2);
    badge(s, n, x + 0.1, y + 0.1, 0.34, n <= 4 ? C.accent1 : C.accent2, `Flow badge ${n}`);
    txt(s, name, x + 0.5, y + 0.08, w - 0.58, 0.48, { fontSize: 12, bold: true, color: C.text2, valign: "middle" });
    txt(s, sub, x + 0.12, y + 0.62, w - 0.24, 0.5, { fontSize: 11, color: C.accent6 });
  };
  const arrow = (x1, y, x2) => s.addShape(pres.ShapeType.line, { x: x1, y, w: x2 - x1, h: 0, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "Flow arrow" });
  txt(s, "OFFLINE  |  build the index once", 0.5, 1.1, 5, 0.3, { fontSize: 12, bold: true, color: C.accent1 });
  const r1 = [["Ingest", "PDF, MD, CSV, XLSX"], ["Chunk + metadata", "tables kept whole"], ["Embed (GPU)", "bge-large, FP16"], ["Dual index", "Chroma + BM25"]];
  r1.forEach((b, i) => { box(i + 1, b[0], b[1], 0.5 + i * 2.3, 1.45, 2.0); if (i < 3) arrow(2.5 + i * 2.3, 2.02, 2.8 + i * 2.3); });
  txt(s, "ONLINE  |  every question", 0.5, 2.85, 5, 0.3, { fontSize: 12, bold: true, color: C.accent2 });
  const r2 = [["Query rewrite", "<=3 queries, filters"], ["Hybrid retrieve", "dense + BM25 + RRF"], ["Rerank (GPU)", "top-25 to top-5"], ["Grounded answer", "Qwen3-Coder, cited"], ["Verify", "claims + faithfulness"]];
  r2.forEach((b, i) => { box(i + 5, b[0], b[1], 0.5 + i * 1.85, 3.2, 1.65); if (i < 4) arrow(2.15 + i * 1.85, 3.77, 2.35 + i * 1.85); });
  txt(s, "Stage 4 index feeds stage 6; stage 9 returns the answer, citations and faithfulness score.", 0.5, 4.55, 9, 0.4, { fontSize: 12, color: C.accent6 });
}

// ------------------------------------------------------------------ 4 stages 1-2
{
  const s = addSlide("LIGHT", "Pipeline stages", "Tables are never split. Each row also becomes a sentence so dense and sparse retrievers can match specific metrics.");
  s.addText("Stages 1-2: ingest and chunking", { placeholder: "title" });
  bullets(s, [
    "pymupdf, pdfplumber and pandas parse PDF, Markdown, CSV and XLSX",
    "Section headings and page numbers are preserved",
    "Text: 512-token chunks, 64-token overlap",
    "KPI tables stay whole, plus one sentence per row",
    "8 Pydantic-validated metadata fields per chunk",
  ], 0.5, 1.2, 4.2, 2.6, 14);
  [["58", "chunks"], ["24", "tables"], ["10", "source files"]].forEach((st, i) => {
    card(s, 0.5 + i * 1.45, 3.95, 1.3, 1.0, `Mini stat ${i}`);
    txt(s, st[0], 0.5 + i * 1.45, 4.0, 1.3, 0.5, { fontSize: 24, bold: true, color: C.accent1, fontFace: "Cambria", align: "center", valign: "middle" });
    txt(s, st[1], 0.5 + i * 1.45, 4.5, 1.3, 0.35, { fontSize: 12, color: C.accent6, align: "center" });
  });
  card(s, 4.95, 1.2, 4.55, 3.75, "Code card", C.text2);
  const code = (t, o = {}) => ({ text: t, options: Object.assign({ fontFace: MONO, fontSize: 11, color: C.background2, breakLine: true }, o) });
  s.addText([
    code("RAW TABLE ROW", { color: C.accent3, bold: true }),
    code("| Q3 2025 | Direct | $182 | 74% | $134.68 |", { paraSpaceAfter: 8 }),
    code("LINEARIZED SENTENCE", { color: C.accent3, bold: true }),
    code("Miami Beach Resort: Q3 2025, Direct channel, ADR (Average Daily Rate) $182, Occupancy 74%, RevPAR (Revenue Per Available Room) $134.68.", { paraSpaceAfter: 8 }),
    code("METADATA", { color: C.accent3, bold: true }),
    code("source   case_study_miami_beach_resort.md"),
    code("section  Q3 2025 Channel Performance"),
    code("page 1  property Miami Beach Resort"),
    code("period Q3 2025  channel all  segment all"),
    code("doc_type case_study", { breakLine: false }),
  ], { x: 5.1, y: 1.3, w: 4.25, h: 3.55, margin: 0, valign: "top", isTextBox: true });
}

// ------------------------------------------------------------------ 5 stages 3-4
{
  const s = addSlide("LIGHT", "Pipeline stages", "Dense and sparse indexes are built from the same chunk list and checked for sync on load.");
  s.addText("Stages 3-4: embeddings and dual index", { placeholder: "title" });
  const cols = [
    ["Embed", "BAAI/bge-large-en-v1.5", ["CUDA, FP16, batches of 32", "Normalized 1024-d vectors", "Tables embed section + row sentences, so nothing is truncated mid-table"]],
    ["Dense index", "ChromaDB, persistent", ["Cosine space", "Metadata stored per chunk", "Native where filters, for example period = Q3 2025"]],
    ["Sparse index", "BM25Okapi, pickled", ["Same chunk ids and order", "Filters applied by masking scores", "Sync check against chunks.jsonl on load"]],
  ];
  cols.forEach((c, i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.2, 2.9, 3.05, `Column ${c[0]}`);
    badge(s, [3, 4, 4][i], x + 0.15, 1.35, 0.4, i === 0 ? C.accent1 : C.accent2, `Col badge ${i}`);
    txt(s, c[0], x + 0.65, 1.35, 2.1, 0.4, { fontSize: 16, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria" });
    txt(s, c[1], x + 0.15, 1.85, 2.6, 0.3, { fontSize: 12, bold: true, color: C.accent1 });
    bullets(s, c[2], x + 0.15, 2.25, 2.6, 1.95, 13);
  });
  card(s, 0.5, 4.4, 9, 0.6, "VRAM note", C.text2);
  txt(s, "VRAM: embedder 0.64 GB, plus reranker about 1.1 GB, beside Qwen3-Coder in LM Studio on one 24 GB RTX 3090", 0.7, 4.4, 8.6, 0.6, { fontSize: 13, color: C.background1, valign: "middle" });
}

// ------------------------------------------------------------------ 6 stage 5
{
  const s = addSlide("LIGHT", "Pipeline stages", "The LLM proposes; deterministic rules guarantee the cap, the acronym expansion and the filters.");
  s.addText("Stage 5: query rewrite and filters", { placeholder: "title" });
  const rows = [["Acronym", "Expanded to"], ["ADR", "average daily rate"], ["RevPAR", "revenue per available room"], ["OTA", "online travel agency"], ["CRM", "customer relationship management"], ["LTV", "lifetime value"]];
  s.addTable(rows.map((r, i) => r.map(t => ({ text: t, options: { bold: i === 0, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.text2 : (i % 2 ? C.background1 : C.background2) }, fontSize: 12, fontFace: i === 0 ? "Calibri" : "Calibri" } }))),
    { x: 0.5, y: 1.25, w: 4.5, colW: [1.1, 3.4], rowH: 0.42, border: { type: "solid", pt: 0.5, color: "D5DEDC" }, margin: [0, 0.1, 0, 0.1], valign: "middle", objectName: "Acronym table" });
  txt(s, "Mandatory expansion, kept beside the original acronym", 0.5, 3.9, 4.5, 0.5, { fontSize: 12, color: C.accent6 });
  bullets(s, [
    "Qwen returns JSON: at most 3 standalone queries plus filters",
    "Hard cap of 3 queries, even if the model sends more",
    "Rule-based filters win: \"only Q3 2025\" becomes period = Q3 2025",
    "Property and channel phrases map to metadata filters",
    "Filter values are checked against the index vocabulary",
  ], 5.3, 1.25, 4.2, 3.6, 14);
}

// ------------------------------------------------------------------ 7 stages 6-7
{
  const s = addSlide("LIGHT", "Pipeline stages", "RRF needs no score calibration, which is why it fuses cosine similarity and BM25 safely.");
  s.addText("Stages 6-7: hybrid retrieval and rerank", { placeholder: "title" });
  const steps = [["Rewritten queries", "up to 3"], ["Dense + BM25", "top-25 each, filtered"], ["RRF fusion", "k = 60, top-25"], ["bge-reranker-large", "cross-encoder, GPU"], ["Top-5 chunks", "to the generator"]];
  steps.forEach((b, i) => {
    const x = 0.5 + i * 1.85;
    card(s, x, 1.25, 1.65, 1.2, `Retrieval step ${i + 1}`, i === 3 ? C.accent1 : C.background2);
    txt(s, b[0], x + 0.1, 1.35, 1.45, 0.55, { fontSize: 13, bold: true, color: i === 3 ? C.background1 : C.text2, valign: "middle", align: "center" });
    txt(s, b[1], x + 0.1, 1.9, 1.45, 0.5, { fontSize: 11, color: i === 3 ? C.background2 : C.accent6, align: "center" });
    if (i < 4) s.addShape(pres.ShapeType.line, { x: x + 1.65, y: 1.85, w: 0.2, h: 0, line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: "Step arrow" });
  });
  card(s, 0.5, 2.75, 4.4, 2.2, "RRF card");
  txt(s, "Reciprocal Rank Fusion", 0.7, 2.85, 4, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  txt(s, "score(d) = sum w / (60 + rank)", 0.7, 3.3, 4, 0.4, { fontSize: 13, fontFace: MONO, color: C.accent1 });
  txt(s, "A chunk ranked well by both dense and BM25, across several rewritten queries, rises to the top.", 0.7, 3.8, 4, 1.0, { fontSize: 13, color: C.text1 });
  card(s, 5.1, 2.75, 4.4, 2.2, "Filter card");
  txt(s, "Zero filter leakage", 5.3, 2.85, 4, 0.35, { fontSize: 14, bold: true, color: C.text2 });
  bullets(s, ["Chroma where clause", "BM25 score masking", "Post-hoc guard on fused results"], 5.3, 3.3, 4, 1.0, 13);
  txt(s, "Tested: \"only Q3 2025\" returns 0 chunks from Q1, Q2 or Q4", 5.3, 4.3, 4, 0.55, { fontSize: 12, color: C.accent6 });
}

// ------------------------------------------------------------------ 8 stages 8-9
{
  const s = addSlide("LIGHT", "Pipeline stages", "Verification is layered: strict regex for numbers, plus an LLM judge cross-checked by lexical overlap.");
  s.addText("Stages 8-9: grounded answer and verify", { placeholder: "title" });
  card(s, 0.5, 1.2, 4.4, 3.75, "Generate card");
  badge(s, 8, 0.65, 1.35, 0.4, C.accent2, "Gen badge");
  txt(s, "Grounded answer", 1.15, 1.35, 3.6, 0.4, { fontSize: 16, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria" });
  bullets(s, ["Only the top-5 chunks are given to Qwen3-Coder", "Inline citation after each fact: [source | page | period]", "No outside knowledge and no model arithmetic", "Refuses when evidence is weak or the entity is unknown (rule-based scope guard)"], 0.7, 1.9, 4.0, 2.95, 13);
  card(s, 5.1, 1.2, 4.4, 3.75, "Verify card");
  badge(s, 9, 5.25, 1.35, 0.4, C.accent2, "Ver badge");
  txt(s, "Verification", 5.75, 1.35, 3.6, 0.4, { fontSize: 16, bold: true, color: C.text2, valign: "middle", fontFace: "Cambria" });
  bullets(s, ["Numbers ($, %, x) must appear in the retrieved chunks", "Every citation must match a retrieved chunk", "LLM judge flags unsupported sentences"], 5.3, 1.9, 4.0, 1.7, 13);
  card(s, 5.3, 3.7, 4.0, 1.1, "Formula card", C.text2);
  txt(s, "faithfulness = 1 - (unsupported numbers + flagged sentences + citation errors) / (numbers + sentences checked)", 5.45, 3.75, 3.7, 1.0, { fontSize: 12, color: C.background1, valign: "middle" });
}

// ------------------------------------------------------------------ 9 example overview
pres.addSection({ title: "Worked example" }); current = "Worked example";
{
  const s = addSlide("LIGHT", "Worked example", "One real run against live Qwen3-Coder. Latencies are steady-state, measured after a warm-up query.");
  s.addText("Example: one question end to end", { placeholder: "title" });
  card(s, 0.5, 1.2, 9, 0.95, "Question card", C.accent2);
  txt(s, "\"" + sample.q + "\"", 0.7, 1.2, 8.6, 0.95, { fontSize: 18, bold: true, color: C.background1, valign: "middle", fontFace: "Cambria" });
  const L = sample.lat, fmt = ms => ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
  const st = [["5", "Rewrite", L.rewrite], ["6", "Retrieve", L.retrieve], ["7", "Rerank", L.rerank], ["8", "Answer", L.generate], ["9", "Verify", L.verify]];
  st.forEach((b, i) => {
    const x = 0.5 + i * 1.85;
    card(s, x, 2.5, 1.65, 1.6, `Example step ${b[0]}`);
    badge(s, b[0], x + 0.62, 2.62, 0.4, C.accent1, `Example badge ${b[0]}`);
    txt(s, b[1], x, 3.1, 1.65, 0.35, { fontSize: 14, bold: true, color: C.text2, align: "center" });
    txt(s, fmt(b[2]), x, 3.5, 1.65, 0.45, { fontSize: 20, bold: true, color: C.accent1, align: "center", fontFace: "Cambria" });
  });
  txt(s, `Total ${fmt(L.total)}. Stages 1-4 were done offline when the corpus was indexed.`, 0.5, 4.4, 9, 0.4, { fontSize: 13, color: C.accent6 });
}

// ------------------------------------------------------------------ 10 example rewrite
{
  const s = addSlide("LIGHT", "Worked example", "The filter on property + period + channel narrows 58 chunks to 5 candidates before any scoring.");
  s.addText("Example, stage 5: queries and filters", { placeholder: "title" });
  sample.rewrite.search_queries.forEach((q, i) => {
    const y = 1.2 + i * 1.0;
    card(s, 0.5, y, 5.3, 0.85, `Query ${i + 1}`);
    badge(s, i + 1, 0.62, y + 0.22, 0.4, C.accent1, `Query badge ${i + 1}`);
    txt(s, q, 1.2, y + 0.05, 4.5, 0.75, { fontSize: 12, valign: "middle" });
  });
  card(s, 6.0, 1.2, 3.5, 2.85, "Filters card", C.text2);
  txt(s, "Extracted filters", 6.2, 1.3, 3.1, 0.35, { fontSize: 14, bold: true, color: C.background1 });
  const f = sample.rewrite.filters;
  txt(s, [
    { text: `period    ${f.period}`, options: { breakLine: true } },
    { text: `property  ${f.property}`, options: { breakLine: true } },
    { text: `channel   ${f.channel}`, options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "ADR -> average daily rate" },
  ], 6.2, 1.75, 3.2, 2.2, { fontSize: 12, fontFace: MONO, color: C.background2 });
  txt(s, `After filtering: ${sample.n_cands} candidates (2 KPI tables and 3 text chunks), all Miami Beach Resort, Q3 2025.`, 0.5, 4.3, 9, 0.6, { fontSize: 14, color: C.text1 });
}

// ------------------------------------------------------------------ 11 example rerank
{
  const s = addSlide("LIGHT", "Worked example", "Scores are sigmoid outputs of the cross-encoder. The two KPI tables that contain $182 score highest.");
  s.addText("Example, stages 6-7: reranked evidence", { placeholder: "title" });
  const kind = id => /::t\d+$/.test(id) ? "KPI table" : "text";
  const hdr = ["#", "Chunk", "Type", "Rerank score"].map(t => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12 } }));
  const rows = [hdr].concat(sample.top5.map((r, i) => [String(i + 1), r[0].split("::")[0], kind(r[0]), r[3].toFixed(3)].map(t => ({
    text: t, options: { fontSize: 12, color: C.text1, fill: { color: i % 2 ? C.background2 : C.background1 } } }))));
  s.addTable(rows, { x: 0.5, y: 1.25, w: 5.9, colW: [0.4, 3.2, 1.1, 1.2], rowH: 0.45, border: { type: "solid", pt: 0.5, color: "D5DEDC" }, margin: [0, 0.08, 0, 0.08], valign: "middle", objectName: "Rerank table" });
  card(s, 6.6, 1.25, 2.9, 2.7, "Rerank insight");
  txt(s, "0.999 vs 0.011", 6.75, 1.4, 2.6, 0.6, { fontSize: 22, bold: true, color: C.accent1, fontFace: "Cambria" });
  txt(s, "The cross-encoder cleanly separates the table that answers the question from filler text, so the generator sees the best evidence first.", 6.75, 2.1, 2.6, 1.8, { fontSize: 13 });
  txt(s, "Only the top-5 go to the LLM. Citations will point at these chunks.", 0.5, 4.2, 9, 0.5, { fontSize: 13, color: C.accent6 });
}

// ------------------------------------------------------------------ 12 example answer
{
  const s = addSlide("LIGHT", "Worked example", "Both numbers trace to the intact KPI table chunk. The judge raised no flags.");
  s.addText("Example, stages 8-9: answer and verify", { placeholder: "title" });
  card(s, 0.5, 1.15, 5.3, 3.85, "Answer card");
  txt(s, "Generated answer (Qwen3-Coder)", 0.7, 1.22, 4.9, 0.3, { fontSize: 12, bold: true, color: C.accent2 });
  const lines = sample.answer.split("\n").filter(l => l.trim()).map(l => l.replace(/\*\*/g, "").replace(/^\u2022\s*/, ""));
  const arr = lines.map((l, i) => ({ text: l, options: i === 0 ? { bold: true, breakLine: true, paraSpaceAfter: 5 } : { bullet: true, breakLine: i < lines.length - 1, paraSpaceAfter: 4 } }));
  s.addText(arr, { x: 0.7, y: 1.55, w: 4.9, h: 3.35, fontSize: 11, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  card(s, 6.0, 1.15, 3.5, 3.85, "Verification card", C.text2);
  txt(s, "Verifier output", 6.2, 1.25, 3.1, 0.3, { fontSize: 14, bold: true, color: C.background1 });
  const v = sample.ver;
  const row = (t, ok, y) => { txt(s, t, 6.2, y, 2.35, 0.5, { fontSize: 12, color: C.background2, valign: "middle" }); txt(s, ok, 8.5, y, 0.85, 0.5, { fontSize: 12, bold: true, color: ok === "OK" ? "7FD6B0" : "F4A261", align: "right", valign: "middle" }); };
  row(`${v.claims[0].value} supported by ${v.claims[0].evidence_chunk.split("::")[0].replace("kpi_miami_beach_resort", "kpi_miami")}`, "OK", 1.65);
  row(`${v.claims[1].value} supported`, "OK", 2.2);
  row("Citations match retrieved chunks (2 of 2)", "OK", 2.75);
  row("LLM judge unsupported sentences: 0", "OK", 3.3);
  txt(s, "Faithfulness", 6.2, 3.95, 1.8, 0.4, { fontSize: 13, color: C.background2, valign: "middle" });
  txt(s, v.faithfulness_score.toFixed(2), 7.9, 3.85, 1.45, 0.6, { fontSize: 30, bold: true, color: "7FD6B0", align: "right", fontFace: "Cambria", valign: "middle" });
}

// ------------------------------------------------------------------ 13 hallucination guard
{
  const s = addSlide("LIGHT", "Worked example", "Same answer with $182 replaced by a fabricated $299. The verifier blocks it; the faithfulness score drops from 1.00 to 0.50.");
  s.addText("Hallucination guard: verified vs blocked", { placeholder: "title" });
  s.addImage({ path: ROOT + "/outputs/screenshots/04_verification_hallucination_guard.png", x: 0.8, y: 1.15, w: 8.4, h: 8.4 * 462 / 1160, altText: "Side-by-side verification card: verified answer scores 1.00, injected $299 hallucination scores 0.50", objectName: "Verification screenshot" });
  txt(s, "Left: grounded answer, 4 of 4 numeric claims supported. Right: fabricated $299 is blocked as not in the retrieved context.", 0.8, 4.6, 8.4, 0.5, { fontSize: 13, color: C.accent6 });
}

// ------------------------------------------------------------------ 14 retrieval metrics
pres.addSection({ title: "Metrics" }); current = "Metrics";
{
  const s = addSlide("LIGHT", "Metrics", "12 golden questions generated from the corpus, asked without filters, so ranking quality is what is measured. RRF already saturates this small set.");
  s.addText("Retrieval quality across four systems", { placeholder: "title" });
  const m = metrics.retrieval, cats = ["Hit@5", "MRR@5", "NDCG@5"];
  const names = [["bm25", "BM25 only"], ["dense", "Dense only"], ["rrf", "RRF hybrid"], ["rrf+rerank", "RRF + rerank"]];
  const data = names.map(([k, n]) => ({ name: n, labels: cats, values: [m[k]["hit@5"], m[k]["mrr@5"], m[k]["ndcg@5"]].map(v => +v.toFixed(3)) }));
  s.addChart(pres.charts.BAR, data, Object.assign(chartBase("Higher is better"), { x: 0.4, y: 1.1, w: 6.1, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: ["94A3B8", "5B8E7D", "0F766E", "C8793B"], valAxisMaxVal: 1.1, valAxisMinVal: 0, valAxisMajorUnit: 0.25, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", showLegend: true, legendPos: "b" }));
  card(s, 6.7, 1.2, 2.8, 1.15, "Callout rerank");
  txt(s, "+" + Math.round((m["rrf+rerank"]["mrr@5"] / m.bm25["mrr@5"] - 1) * 100) + "%", 6.85, 1.25, 2.5, 0.55, { fontSize: 26, bold: true, color: C.accent1, fontFace: "Cambria", valign: "middle" });
  txt(s, "MRR@5 vs BM25 alone", 6.85, 1.8, 2.5, 0.5, { fontSize: 12, color: C.accent6 });
  card(s, 6.7, 2.5, 2.8, 1.15, "Callout dense");
  txt(s, "+" + ((m["rrf+rerank"]["mrr@5"] / m.dense["mrr@5"] - 1) * 100).toFixed(1) + "%", 6.85, 2.55, 2.5, 0.55, { fontSize: 26, bold: true, color: C.accent1, fontFace: "Cambria", valign: "middle" });
  txt(s, "MRR@5 vs dense alone", 6.85, 3.1, 2.5, 0.5, { fontSize: 12, color: C.accent6 });
  txt(s, "Reranking matches RRF here (both 1.00); a larger, harder golden set is needed to separate them.", 6.7, 3.85, 2.8, 1.1, { fontSize: 12, color: C.accent6 });
}

// ------------------------------------------------------------------ 15 performance
{
  const s = addSlide("LIGHT", "Metrics", "The three LLM calls dominate latency; retrieval and reranking together take about 0.2 seconds.");
  s.addText("Performance: where the time goes", { placeholder: "title" });
  const L = sample.lat, labels = ["Rewrite (LLM)", "Retrieve", "Rerank (GPU)", "Answer (LLM)", "Verify (LLM)"];
  const vals = [L.rewrite, L.retrieve, L.rerank, L.generate, L.verify].map(v => +(v / 1000).toFixed(2));
  s.addChart(pres.charts.BAR, [{ name: "seconds", labels, values: vals }], Object.assign(chartBase("Per-stage latency, sample query (seconds)"), { x: 0.4, y: 1.1, w: 5.4, h: 3.5, barDir: "bar", chartColors: ["0F766E"], showLegend: false, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", valAxisMaxVal: 20, catAxisOrientation: "maxMin", valAxisLabelFormatCode: "0" }));
  const llmShare = Math.round((L.rewrite + L.generate + L.verify) / L.total * 100);
  txt(s, `LLM calls = ${llmShare}% of ${(L.total / 1000).toFixed(1)} s total; retrieve + rerank = ${Math.round(L.retrieve + L.rerank)} ms.`, 0.5, 4.65, 5.3, 0.5, { fontSize: 12, color: C.accent6 });
  const hdr = ["Query", "Total"].map(t => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 11 } }));
  const short = ["Miami Direct ADR", "Orlando RevPAR", "Fort Laud. OTA vs Direct", "Miami campaign ROI", "CRM and LTV", "Tokyo (refused)"];
  const rows = [hdr].concat(lat.map((r, i) => [short[i], (r.total / 1000).toFixed(1) + " s"].map(t => ({ text: t, options: { fontSize: 11, color: C.text1, fill: { color: i % 2 ? C.background2 : C.background1 } } }))));
  s.addTable(rows, { x: 6.0, y: 1.15, w: 3.5, colW: [2.5, 1.0], rowH: 0.36, border: { type: "solid", pt: 0.5, color: "D5DEDC" }, margin: [0, 0.08, 0, 0.08], valign: "middle", objectName: "Latency table" });
  txt(s, `Indexing 58 chunks: ${metrics.ingest.seconds.ingest_chunk.toFixed(1)} s parse + chunk, ${metrics.ingest.seconds.embed_index.toFixed(1)} s embed + index. Refusals skip generation (5.4 s).`, 6.0, 3.85, 3.5, 1.1, { fontSize: 12, color: C.accent6 });
}

// ------------------------------------------------------------------ 16 quality + tests
{
  const s = addSlide("LIGHT", "Metrics", "Faithfulness is from the live run. Both scores below 1.00 are single LLM-judge sentence flags, with no numeric error.");
  s.addText("Quality and test results", { placeholder: "title" });
  const labels = ["Miami ADR", "Orlando RevPAR", "FtL OTA vs Direct", "Campaign ROI", "CRM and LTV", "Tokyo"];
  const vals = metrics.answers.map(a => +a.faithfulness.toFixed(2));
  s.addChart(pres.charts.BAR, [{ name: "faithfulness", labels, values: vals }], Object.assign(chartBase("Faithfulness score per live query"), { x: 0.4, y: 1.1, w: 5.4, h: 3.8, barDir: "col", chartColors: ["0F766E"], showLegend: false, valAxisMaxVal: 1.1, valAxisMinVal: 0, valAxisMajorUnit: 0.25, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", catAxisLabelFontSize: 10 }));
  const rows = [["Check", "Result"], ["Full suite (live Qwen)", "57 / 57 passed"], ["Code coverage (src)", "75%"], ["\"only Q3 2025\" leakage", "0 chunks"], ["Grounded $182 answer", "faithfulness 1.00"], ["Injected $299 answer", "flagged, 0.50"], ["Tokyo question", "refused"]];
  s.addTable(rows.map((r, i) => r.map(t => ({ text: t, options: { bold: i === 0, fontSize: 12, color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.text2 : (i % 2 ? C.background1 : C.background2) } } }))),
    { x: 6.0, y: 1.2, w: 3.5, colW: [1.9, 1.6], rowH: 0.45, border: { type: "solid", pt: 0.5, color: "D5DEDC" }, margin: [0, 0.08, 0, 0.08], valign: "middle", objectName: "Test results table" });
  txt(s, `Mean faithfulness ${metrics.mean_faithfulness.toFixed(2)} over ${metrics.answers.length} queries.`, 6.0, 4.5, 3.5, 0.4, { fontSize: 12, color: C.accent6 });
}

// ------------------------------------------------------------------ 17 limits / next
pres.addSection({ title: "Wrap-up" }); current = "Wrap-up";
{
  const s = addSlide("LIGHT", "Wrap-up", "Be upfront about limits: small synthetic corpus, judge strictness, and LLM latency.");
  s.addText("Limitations and next steps", { placeholder: "title" });
  card(s, 0.5, 1.2, 4.4, 3.75, "Limits card");
  txt(s, "Known limits", 0.7, 1.3, 4, 0.4, { fontSize: 16, bold: true, color: C.accent2, fontFace: "Cambria" });
  bullets(s, ["Synthetic corpus of 58 chunks; golden set of 12 questions is saturated", "LLM judge sometimes flags valid comparisons such as \"highest RevPAR\"", "Model-computed numbers (differences) are rejected by design", "LLM calls take 17-28 s per answerable question"], 0.7, 1.8, 4.0, 3.1, 13);
  card(s, 5.1, 1.2, 4.4, 3.75, "Next card");
  txt(s, "Next steps", 5.3, 1.3, 4, 0.4, { fontSize: 16, bold: true, color: C.accent1, fontFace: "Cambria" });
  bullets(s, ["Index real campaign and brand documents; grow the golden set", "Skip the LLM rewrite for simple queries to save about 5-9 s", "Stream the answer and run the judge in parallel", "Allow verified derived numbers (sum, difference) with shown formulas"], 5.3, 1.8, 4.0, 3.1, 13);
}

// ------------------------------------------------------------------ 18 closing
{
  const s = addSlide("DARK", "Wrap-up", "Everything is reproducible with one command; artifacts land in outputs/.");
  s.addText("Run it yourself", { placeholder: "title" });
  card(s, 0.6, 1.5, 8.8, 2.0, "Commands card", C.accent1);
  const code = (t, o = {}) => ({ text: t, options: Object.assign({ fontFace: MONO, fontSize: 13, color: C.background1, breakLine: true, paraSpaceAfter: 6 }, o) });
  s.addText([
    code("python scripts/run_end_to_end.py"),
    code("pytest --live-llm"),
    code("uvicorn api.main:app --port 8000", { breakLine: false }),
  ], { x: 0.9, y: 1.7, w: 8.2, h: 1.6, margin: 0, valign: "middle", isTextBox: true });
  txt(s, "Generates data, indexes, queries, scores retrieval, runs tests and renders charts and screenshots. API: /health /ingest /query /verify.", 0.6, 3.8, 8.8, 0.9, { fontSize: 14, color: C.background2 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
