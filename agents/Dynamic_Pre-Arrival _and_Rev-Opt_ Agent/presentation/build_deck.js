// Builds Revenue_Agent_Deck.pptx from the REAL files the pipeline produced (metrics/, evaluation/, validation/, outputs/).
// Run:  cd presentation && npm install && node build_deck.js      (re-run after `python run_all.py` to refresh every number)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const J = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const M = J("metrics/metrics.json");
const MM = J("evaluation/model_metrics.json");
const TI = J("evaluation/training_info.json");
const VR = J("validation/data_validation_report.json");
const REV = M.revenue;
const NL = M.node_latency;
const F = M.funnel;

// ------------------------------------------------------------------ theme
const THEME = {
  name: "Azure Bay Agent",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: { dk1: "1F2D3A", lt1: "FFFFFF", dk2: "0B2B40", lt2: "EEF4F7", accent1: "0E7C86", accent2: "F2994A", accent3: "2D6CDF",
            accent4: "7A5CC7", accent5: "5B6B78", accent6: "1F7A52", hlink: "0E7C86", folHlink: "5B6B78" },
};
const H = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GPU-Accelerated Dynamic Pre-Arrival & Revenue Optimization Agent";
pres.author = "Revenue Agent Team";
const C = pres.SchemeColor;
const S = pres.ShapeType;

pres.defineSlideMaster({ title: "TITLE_DARK", background: { color: H.dk2 } });
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { text: { text: "Dynamic Pre-Arrival & Revenue Optimization Agent", options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0, isTextBox: true } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, margin: 0, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5 },
});

// ------------------------------------------------------------------ helpers
const fmt = (n, d = 0) => Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const sign = (n) => (n > 0 ? "+" : "") + n;
function slide(title, notes, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function T(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true }, o));
}
function bullets(items, o = {}) {
  return items.map((t, i) => ({ text: t, options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 }, o) }));
}
function tile(s, x, y, w, h, num, label, color = C.accent1, numSize = 32) {
  if (color === C.accent2) color = "B35A00"; // darker orange keeps text contrast on the light tile
  s.addShape(S.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1 });
  T(s, num, x + 0.2, y + 0.12, w - 0.4, h * 0.46, { fontSize: numSize, bold: true, color, valign: "middle" });
  T(s, label, x + 0.2, y + h * 0.58, w - 0.4, h * 0.38, { fontSize: 12, color: C.text1 });
}
function pngSize(p) { const b = fs.readFileSync(p); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function pic(s, rel, x, y, maxW, maxH, alt) {
  const p = path.join(ROOT, rel);
  const [w, h] = pngSize(p);
  const k = Math.min(maxW / w, maxH / h);
  const W = w * k, Hh = h * k, ox = x + (maxW - W) / 2;
  s.addShape(S.rect, { x: ox - 0.04, y: y - 0.04, w: W + 0.08, h: Hh + 0.08, fill: { color: "FFFFFF" }, line: { color: "C9D4DB", width: 0.75 } });
  s.addImage({ path: p, x: ox, y, w: W, h: Hh, altText: alt });
  return { W, H: Hh, x: ox };
}
function arrow(s, x, y, w, h, o = {}) {
  s.addShape(S.line, Object.assign({ x, y, w, h, line: { color: H.accent5, width: 2, endArrowType: "triangle" } }, o));
}
function seg(s, x, y, w, h, o = {}) { // line without arrowhead
  s.addShape(S.line, Object.assign({ x, y, w, h, line: { color: H.accent5, width: 2 } }, o));
}
function chartBase() {
  return { catAxisLabelColor: H.accent5, valAxisLabelColor: H.accent5, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
           catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: H.dk1,
           legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: H.dk1, titleFontFace: "+mn-lt", titleColor: H.dk1, titleFontSize: 14,
           valGridLine: { color: "DDE5EA", size: 0.75 }, catGridLine: { style: "none" } };
}

// ================================================================== 1 TITLE
pres.addSection({ title: "Overview" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
  T(s, "GPU-ACCELERATED  ·  LANGGRAPH  ·  LOCAL QWEN3-VL", 0.8, 1.5, 11, 0.4, { fontSize: 14, color: C.accent2, bold: true, charSpacing: 3 });
  T(s, "Dynamic Pre-Arrival &\nRevenue Optimization Agent", 0.8, 2.0, 11.5, 2.2, { fontSize: 50, bold: true, color: C.background1, fontFace: "Cambria", valign: "top" });
  T(s, "A bounded state-machine agent that grows ancillary revenue and direct bookings\nwhile guaranteeing zero price, discount or SKU hallucination", 0.8, 4.45, 11.5, 1.0, { fontSize: 20, color: "C9D9E2" });
  T(s, `Azure Bay Resort & Spa (synthetic)  ·  ${M.n_journeys} replayed guest journeys  ·  ${M.gpu.name}`, 0.8, 6.55, 11.5, 0.4, { fontSize: 14, color: "8FB0C2" });
  s.addNotes("End-to-end walkthrough: architecture, every stage of the agent flow with its own metrics, overall and performance metrics, and the Streamlit UI flow with five sample use cases. All numbers are read from the files the pipeline produced.");
}

// ================================================================== 2 EXEC SUMMARY
{
  const s = slide("Executive summary", "Headline safety results first, then the honest revenue picture.", "Overview");
  const w = 2.85, g = 0.233, x0 = 0.6;
  tile(s, x0, 1.55, w, 1.7, `${M.hallucination_rate_pct.toFixed(1)}%`, `price / discount / SKU hallucination across ${M.messages_judged} sent messages`, C.accent6, 40);
  tile(s, x0 + (w + g), 1.55, w, 1.7, `${M.adversarial_caught}/${M.adversarial_injected}`, "poisoned drafts blocked, 0 reached a guest", C.accent6, 40);
  tile(s, x0 + 2 * (w + g), 1.55, w, 1.7, `${M.inbound_booked}/${M.inbound_accepts}`, "guest YES replies booked to the room folio", C.accent1, 40);
  tile(s, x0 + 3 * (w + g), 1.55, w, 1.7, `${VR.n_passed}/${VR.n_checks}`, "data validation checks passed", C.accent1, 40);
  T(s, bullets([
    "LLM writes copy only. Prices, discounts and SKUs come from XGBoost (GPU) plus deterministic RMS rules and are re-verified before sending.",
    "Six validated LangGraph nodes with a retry loop and human-in-the-loop interrupts; 11 risky interactions escalated and resolved.",
    `GPU-first: XGBoost on ${TI.device} (hist), bge embeddings and reranker in FP16; avg pre-arrival journey ${M.avg_journey_s} s on local Qwen3-VL.`,
    `Revenue on synthetic data: ${sign(REV.anc_revpar_lift_pct)}% ancillary revenue per room-night vs a blast-everyone baseline, offset by ~$${fmt(REV.est_ota_commission_avoided_usd)} estimated OTA commission avoided.`,
  ]), 0.6, 3.6, 12.1, 3.3, { fontSize: 18 });
}

// ================================================================== 3 OBJECTIVES
{
  const s = slide("Problem and objectives", "Four objectives from the project brief and how each is measured.", "Overview");
  const rows = [
    ["Incremental revenue and channel shift", "Context-aware pre-arrival upsells; convert OTA guests to direct loyalty members to cut 15-25% commission leakage.", "Ancillary revenue per room-night, loyalty offers sent"],
    ["Zero-hallucination grounding", "LLM reasoning separated from RMS/XGBoost logic; HybridRAG grounds facts; automated gates before any send.", "Hallucination rate, adversarial catch rate"],
    ["Human-in-the-loop governance", "Two-way replies check live inventory; VIP custom requests, negative sentiment and repeated failures pause for a concierge.", "Escalations by reason, folio bookings"],
    ["GPU-first, fully observable", "CUDA for embeddings, reranking and XGBoost with CPU fallback; JSONL telemetry, validation reports, evaluation benchmarks.", "Node latency, device, data checks"],
  ];
  const y0 = 1.5, rh = 1.28;
  rows.forEach((r, i) => {
    const y = y0 + i * (rh + 0.1);
    s.addShape(S.roundRect, { x: 0.6, y, w: 12.1, h: rh, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08 });
    s.addShape(S.ellipse, { x: 0.85, y: y + 0.36, w: 0.56, h: 0.56, fill: { color: [C.accent1, C.accent3, C.accent4, C.accent6][i] }, line: { color: C.background1, width: 0 } });
    T(s, String(i + 1), 0.85, y + 0.36, 0.56, 0.56, { fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, r[0], 1.7, y + 0.14, 5.0, 0.4, { fontSize: 18, bold: true, color: C.text2 });
    T(s, r[1], 1.7, y + 0.55, 6.6, 0.7, { fontSize: 13 });
    T(s, "Measured by", 8.9, y + 0.2, 3.6, 0.3, { fontSize: 11, color: C.accent5, bold: true });
    T(s, r[2], 8.9, y + 0.5, 3.6, 0.7, { fontSize: 14, color: C.text1 });
  });
}

// ================================================================== 4 DESIGN PRINCIPLE
{
  const s = slide("Design principle: LLM for language, rules for facts", "Who owns each concern. The LLM never writes a number a guest will see.", "Overview");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14 } });
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 13, color: C.text1 }, o) });
  s.addTable([
    [hdr("Concern"), hdr("Owner"), hdr("Why")],
    [cell("Which offer, at what price", { bold: true }), cell("XGBoost propensity (GPU) + RMS rules"), cell("Auditable and testable; no free-text numbers")],
    [cell("Property facts and brand voice", { bold: true }), cell("Hybrid RAG: Chroma dense + BM25, RRF, cross-encoder rerank"), cell("Grounded in the knowledge base")],
    [cell("Copywriting, reply intent and sentiment", { bold: true }), cell("Qwen3-VL (qwen3-vl-30b-a3b-instruct, LM Studio)"), cell("Language is what an LLM is for")],
    [cell("Is the message safe to send", { bold: true }), cell("Deterministic compliance gate + final re-scan"), cell("Every $, % and SKU in the text must come from the offer payload")],
  ], { x: 0.6, y: 1.5, w: 12.1, colW: [3.3, 4.6, 4.2], rowH: 0.62, border: { type: "solid", color: "D5DEE4", pt: 0.75 }, valign: "middle", margin: [0.05, 0.12, 0.05, 0.12] });
  s.addShape(S.roundRect, { x: 0.6, y: 4.9, w: 12.1, h: 1.7, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1 });
  T(s, "How a price reaches a guest", 0.9, 5.05, 6, 0.35, { fontSize: 16, bold: true, color: C.text2 });
  T(s, [
    { text: "1. RMS offer carries list price, discount and final price.   ", options: { breakLine: true } },
    { text: "2. Qwen writes copy with placeholders such as {{OFFER_PRICE}}.   ", options: { breakLine: true } },
    { text: "3. Compliance node substitutes the real values and audits the rendered text; dispatch re-scans once more.", options: {} },
  ], 0.9, 5.5, 11.5, 1.0, { fontSize: 14 });
}

// ================================================================== 5 ARCHITECTURE
pres.addSection({ title: "Agent flow" });
{
  const s = slide("System architecture", "Layered view: sources, deterministic engines, LLM, orchestration, interfaces.", "Agent flow");
  const layers = [
    ["Interfaces", ["Streamlit Control Center", "Flask REST API  /api/v1", "Playwright capture"], H.accent3],
    ["Orchestration", ["LangGraph StateGraph (6 nodes)", "Pre / post step validators", "SQLite checkpointer + HITL interrupt()"], H.accent4],
    ["Reasoning and grounding", ["Qwen3-VL 30B-A3B (local, OpenAI API)", "Chroma + BM25 + RRF + reranker (GPU fp16)", "Intent / sentiment / tone judge"], H.accent1],
    ["Deterministic logic", ["XGBoost NBA (tree_method=hist, device=cuda)", "RMS rules: margin, discount, inventory", "Compliance gate + caps + audit"], H.accent2],
    ["Data", ["PMS reservations, CRM Guest360", "RMS inventory and occupancy, weather, events", "Knowledge base: spa, dining, rooms, policies, brand"], H.accent5],
  ];
  layers.forEach((L, i) => {
    const y = 1.45 + i * 1.08;
    s.addShape(S.roundRect, { x: 0.6, y, w: 2.6, h: 0.95, fill: { color: L[2] }, line: { color: L[2] }, rectRadius: 0.08 });
    T(s, L[0], 0.75, y, 2.3, 0.95, { fontSize: 15, bold: true, color: i === 3 ? C.text1 : C.background1, valign: "middle" });
    L[1].forEach((t, j) => {
      const x = 3.35 + j * 3.12;
      s.addShape(S.roundRect, { x, y, w: 3.0, h: 0.95, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08 });
      T(s, t, x + 0.15, y, 2.7, 0.95, { fontSize: 13, valign: "middle" });
    });
  });
}

// ================================================================== 6 FLOW DIAGRAM
{
  const s = slide("End-to-end agent flow: six validated stages", "Every node runs pre-condition validation, the work, post-condition validation and telemetry. FAIL_POLICY loops back to drafting; repeated failure escalates to a human.", "Agent flow");
  T(s, "Trigger (pre-arrival campaign or inbound guest reply)  →  six validated nodes  →  SENT / BOOKED, escalated, or safely stopped", 0.6, 1.3, 12.1, 0.35, { fontSize: 14, color: C.accent5 });
  const nodes = [
    ["1  Enrich context", "PMS + CRM + weather, events, occupancy", C.accent1, C.background1],
    ["2  Retrieve and offers", "Hybrid RAG + XGBoost + RMS rules", C.accent1, C.background1],
    ["3  Draft message", "Qwen3-VL copy with placeholders", C.accent3, C.background1],
    ["4  Compliance check", "$ / % / SKU audit, tone, caps", C.accent2, C.text1],
    ["5  HITL gate", "VIP, negative sentiment, failures", C.accent4, C.background1],
    ["6  Dispatch and log", "Re-scan, send, folio, audit", C.accent6, C.background1],
  ];
  const nw = 1.75, gap = 0.32, y = 1.85, nh = 1.45;
  nodes.forEach((n, i) => {
    const x = 0.6 + i * (nw + gap);
    s.addShape(S.roundRect, { x, y, w: nw, h: nh, fill: { color: n[2] }, line: { color: n[2] }, rectRadius: 0.1 });
    T(s, n[0], x + 0.1, y + 0.1, nw - 0.2, 0.5, { fontSize: 14, bold: true, color: n[3] });
    T(s, n[1], x + 0.1, y + 0.62, nw - 0.2, 0.8, { fontSize: 12, color: n[3] });
    if (i < 5) arrow(s, x + nw + 0.02, y + nh / 2, gap - 0.04, 0);
  });
  const cx = (i) => 0.6 + i * (nw + gap) + nw / 2;
  // retry loop: node4 bottom -> down -> left -> up into node3 (arrow)
  seg(s, cx(3), y + nh, 0, 0.5);
  seg(s, cx(2), y + nh + 0.5, cx(3) - cx(2), 0);
  arrow(s, cx(2), y + nh, 0, 0.5, { flipV: true });
  T(s, "FAIL_POLICY: redraft with violations as feedback (retries ≤ 2)", cx(2) - 0.6, y + nh + 0.58, 4.4, 0.35, { fontSize: 12, color: C.accent2, bold: true });
  // HITL queue
  arrow(s, cx(4), y + nh, 0, 1.15);
  T(s, "interrupt()", cx(4) + 0.1, y + nh + 0.35, 1.3, 0.3, { fontSize: 12, color: C.accent4, bold: true });
  s.addShape(S.roundRect, { x: cx(4) - 1.0, y: y + nh + 1.2, w: 2.5, h: 0.85, fill: { color: "EFE9FA" }, line: { color: H.accent4, width: 1.25 }, rectRadius: 0.1 });
  T(s, "Human concierge queue\napprove / edit / reject", cx(4) - 0.9, y + nh + 1.2, 2.3, 0.85, { fontSize: 12, color: C.text1, valign: "middle", align: "center", bold: true });
  seg(s, cx(4) + 1.5, y + nh + 1.62, cx(5) - cx(4) - 1.5, 0);
  arrow(s, cx(5), y + nh, 0, 1.62, { flipV: true });
  T(s, "approve", cx(5) + 0.1, y + nh + 1.2, 1.0, 0.3, { fontSize: 12, color: C.accent4, bold: true });
  // exits
  T(s, "Safe exits (nothing is sent)", 0.6, 5.55, 5, 0.3, { fontSize: 13, bold: true, color: C.text2 });
  const exits = [["Validation failure", "bad state at any node"], ["No eligible offer", `${F.no_offer} of ${M.n_journeys} journeys`], ["Hard stop", "opt-out, frequency cap, quiet hours, no inventory"], ["Rejected by human", "concierge declines"]];
  exits.forEach((e, i) => {
    const x = 0.6 + i * 3.05;
    s.addShape(S.roundRect, { x, y: 5.95, w: 2.9, h: 0.9, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08 });
    T(s, e[0], x + 0.12, 6.0, 2.66, 0.3, { fontSize: 13, bold: true });
    T(s, e[1], x + 0.12, 6.32, 2.66, 0.5, { fontSize: 11, color: C.accent5 });
  });
}

// ================================================================== 7 VALIDATION CONTRACTS
{
  const s = slide("Step validation contracts at every node boundary", "src/validation/step_validators.py. A violation turns into FAIL_VALIDATION and the graph stops; it never sends.", "Agent flow");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
  const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 12, color: C.text1 }, o) });
  s.addTable([
    [hdr("Node"), hdr("Pre-condition"), hdr("Post-condition")],
    [c("1 enrich_context", { bold: true }), c("guest_id and reservation_id present; valid trigger; reply text non-empty"), c("Guest and reservation exist and match; check_out > check_in; party size ≥ 1; precip ∈ [0,1]; occupancy ∈ [0,100]")],
    [c("2 retrieve + offers", { bold: true }), c("Guest, stay and environment present"), c("final = list × (1 − discount); discount ≤ cap; margin ≥ floor; inventory > 0; propensity ∈ [0,1]; KB snippets found")],
    [c("3 draft_message", { bold: true }), c("Selected offer and guest present"), c("Subject and template non-empty; length bounded")],
    [c("4 compliance_check", { bold: true }), c("Draft template present; offer present"), c("Guardrail status is PASS, FAIL_POLICY or FAIL_VALIDATION")],
    [c("5 hitl_gate", { bold: true }), c("Guardrail result present"), c("Human decision recorded (approve / reject) when paused")],
    [c("6 dispatch_and_log", { bold: true }), c("Guardrail PASS or human approval"), c("Terminal status SENT, BOOKED, SOLD_OUT or BLOCKED")],
  ], { x: 0.6, y: 1.45, w: 12.1, colW: [2.5, 4.4, 5.2], rowH: [0.45, 0.75, 0.75, 0.55, 0.55, 0.55, 0.55], border: { type: "solid", color: "D5DEE4", pt: 0.75 }, valign: "middle", margin: [0.04, 0.1, 0.04, 0.1] });
}

// ================================================================== STAGE SLIDES
pres.addSection({ title: "Stage metrics" });
function stageSlide(n, name, color, does, checks, tiles, notes) {
  const s = slide(`Stage ${n}: ${name}`, notes, "Stage metrics");
  s.addShape(S.roundRect, { x: 0.6, y: 1.5, w: 0.9, h: 0.5, fill: { color }, line: { color }, rectRadius: 0.08 });
  T(s, `Node ${n}`, 0.6, 1.5, 0.9, 0.5, { fontSize: 14, bold: true, color: n === 4 ? C.text1 : C.background1, align: "center", valign: "middle" });
  T(s, "What it does", 1.7, 1.55, 4.5, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  T(s, bullets(does), 0.6, 2.2, 6.2, 2.5, { fontSize: 14 });
  T(s, "Validation and checks", 0.6, 4.55, 6, 0.35, { fontSize: 16, bold: true, color: C.text2 });
  T(s, bullets(checks, { paraSpaceAfter: 5 }), 0.6, 4.95, 6.2, 1.95, { fontSize: 13 });
  const tw = 2.65, th = 1.6;
  T(s, "Metrics from the replay", 7.2, 1.55, 5.5, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  tiles.forEach((t, i) => tile(s, 7.2 + (i % 2) * (tw + 0.2), 2.2 + Math.floor(i / 2) * (th + 0.2), tw, th, t[0], t[1], t[2] || color, 30));
  return s;
}
stageSlide(1, "Enrich context", C.accent1,
  ["Calls get_guest_360() for the CRM profile and the PMS reservation", "Calls get_environmental_signals(): forecast, local events, occupancy", "Builds the typed AgentState used by every later node"],
  ["Guest and reservation exist and belong together", "check_out after check_in, party size ≥ 1", "Weather probability within [0, 1]"],
  [[`${M.n_journeys}/${M.n_journeys}`, "journeys passed pre and post validation"], [`${NL.enrich_context.mean_ms} ms`, `mean latency (p95 ${NL.enrich_context.p95_ms} ms)`], ["0", "enrichment validation failures"], ["2", "tools called per journey (CRM/PMS, environment)"]],
  "Cheapest stage; all 120 replayed reservations were valid.");
stageSlide(2, "Retrieve knowledge and offers", C.accent1,
  ["Hybrid RAG: ChromaDB dense (bge-large, FP16 on CUDA) + BM25, fused by RRF, reranked on GPU", "XGBoost (tree_method=hist, device=cuda) scores acceptance propensity per eligible SKU", "RMS rules set discount and margin, then rank by expected value (top 3)"],
  ["Eligibility: inventory, room type, party size, occupancy", "Margin ≥ 35%, discount ≤ 20%, price = list × (1 − discount)", "Retrieved snippets must exist for grounding"],
  [[`${M.n_journeys - F.no_offer}/${M.n_journeys}`, `journeys with ≥ 1 eligible offer (${F.no_offer} NO_OFFER)`], [`${NL.retrieve_knowledge_and_offers.p50_ms} ms`, `median latency; mean ${NL.retrieve_knowledge_and_offers.mean_ms} ms includes one-time GPU model load`], [`${MM.roc_auc}`, `XGBoost ROC-AUC on ${fmt(MM.n_test)} held-out offers`], [`${M.loyalty_offers_sent}`, "loyalty (OTA to direct) offers sent"]],
  "Offer mix by SKU is in metrics.json offers_by_sku. The mean latency is inflated by the first call loading embedder and reranker.");
stageSlide(3, "Draft message (Qwen3-VL)", C.accent3,
  ["Local qwen3-vl-30b-a3b-instruct via OpenAI-compatible LM Studio endpoint", "Writes subject and body with placeholders only: {{OFFER_PRICE}}, {{OFFER_DISCOUNT}}, ...", "On a retry it receives the gate's violations as feedback"],
  ["Subject and template non-empty, bounded length", "Brand voice and property facts injected from retrieved snippets", "Safe template fallback if the LLM fails"],
  [[`${fmt(NL.draft_message.mean_ms)} ms`, `mean draft latency (p95 ${fmt(NL.draft_message.p95_ms)} ms)`], [`${NL.draft_message.n}`, "draft calls incl. retries and reply drafts"], [`${M.llm_sources["qwen3-vl"]} / ${M.llm_sources.safe_fallback}`, "Qwen drafts vs safe fallback messages sent"], [`${M.avg_tone_score} / 5`, `brand-tone score, LLM judge on ${M.messages_judged} messages`]],
  "Draft generation is the dominant cost of a journey.");
stageSlide(4, "Compliance check", C.accent2,
  ["Substitutes real values into the placeholders, then audits the rendered text", "FAIL_POLICY loops back to drafting (retry ≤ 2); exhausted retries escalate", "FAIL_VALIDATION is a hard stop: opt-out, 7-day frequency cap, quiet hours, no inventory"],
  ["Every $, % and SKU token must belong to the offer payload", "Banned phrases, ALL CAPS, length, unresolved placeholders", "Discount cap, margin floor, price consistency, live inventory"],
  [[`${F.first_pass}`, "journeys passed on the first draft"], [`${F.retried_then_passed}`, "blocked, redrafted and passed"], [`${F.hard_stop_guest_not_opted_in + F.hard_stop_frequency_cap_exceeded}`, `hard stops (${F.hard_stop_guest_not_opted_in} not opted in, ${F.hard_stop_frequency_cap_exceeded} frequency cap)`], [`${M.adversarial_caught}/${M.adversarial_injected}`, "injected fake-price drafts caught"]],
  `Gate latency is ${NL.compliance_check.mean_ms} ms mean. Adversarial drafts claim a fake SKU, $49 and 40% off.`);
stageSlide(5, "Human-in-the-loop gate", C.accent4,
  ["Pauses the graph with interrupt(); state persists in the SQLite checkpointer", "Concierge approves, edits or rejects; approval resumes the same thread", "Escalation triggers: repeated policy failure, negative sentiment, Platinum custom request"],
  ["Decision must be approve or reject before dispatch", "Escalated drafts use a safe fallback, never the failed text", "Every decision is written to the audit table"],
  [[`${Object.values(M.hitl_reasons).reduce((a, b) => a + b, 0)}`, "escalations in the replay, all resolved"], [`${M.hitl_reasons.repeated_policy_failure} / ${M.hitl_reasons.negative_sentiment} / ${M.hitl_reasons.vip_platinum_custom_request}`, "policy failure / negative sentiment / VIP request"], [`${NL.hitl_gate.mean_ms} ms`, `mean gate overhead (p50 ${NL.hitl_gate.p50_ms} ms on the no-risk path)`], [`${M.inbound_replies}`, "inbound guest replies processed"]],
  "In the replay a simulated concierge approves every escalation.");
stageSlide(6, "Dispatch and log", C.accent6,
  ["Final re-scan of the exact text being delivered (covers human edits)", "On a confirmed reply: atomic inventory check, decrement and folio posting", "Writes outbox message and audit record"],
  ["Requires guardrail PASS or human approval", "Idempotent booking per reservation and SKU", "Terminal status SENT, BOOKED, SOLD_OUT or BLOCKED"],
  [[`${M.messages_sent}`, "messages dispatched"], [`${M.inbound_booked}/${M.inbound_accepts}`, `accepted replies booked to the folio (${M.inbound_sold_out} sold out)`], [`${NL.dispatch_and_log.mean_ms} ms`, `mean latency (p95 ${NL.dispatch_and_log.p95_ms} ms), SQLite writes`], [`${M.adversarial_sent_to_guest}`, "poisoned drafts that reached a guest"]],
  "Booking uses BEGIN IMMEDIATE so two guests cannot take the last unit.");

// ================================================================== OVERALL
pres.addSection({ title: "Overall and performance" });
{
  const s = slide("Overall metrics at a glance", `Source: metrics/metrics.json, evaluation/model_metrics.json, validation/data_validation_report.json. Seed ${M.seed}.`, "Overall and performance");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14 } });
  const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 13, color: C.text1 }, o) });
  const rows = [
    ["Guest journeys replayed", `${M.n_journeys} (${M.n_crashed} crashed)`, `${fmt(M.wall_seconds)} s wall time`],
    ["Messages sent / judged", `${M.messages_sent} / ${M.messages_judged}`, "every sent message re-audited"],
    ["Hallucination rate (price, discount, SKU)", `${M.hallucination_rate_pct.toFixed(1)}%`, `${M.hallucinated_messages} of ${M.messages_judged}; target 0.0%`],
    ["Adversarial drafts injected / caught / delivered", `${M.adversarial_injected} / ${M.adversarial_caught} / ${M.adversarial_sent_to_guest}`, "catch rate " + M.adversarial_catch_rate_pct + "%"],
    ["Inbound replies / accepted / booked", `${M.inbound_replies} / ${M.inbound_accepts} / ${M.inbound_booked}`, `${M.inbound_sold_out} sold out`],
    ["Brand-tone score (LLM judge)", `${M.avg_tone_score} / 5`, `n = ${M.messages_judged}`],
    ["XGBoost ROC-AUC / PR-AUC / KS / Brier", `${MM.roc_auc} / ${MM.pr_auc} / ${MM.ks} / ${MM.brier}`, `Brier baseline ${MM.brier_baseline}`],
    ["Data validation", `${VR.n_passed} / ${VR.n_checks} checks`, `PSI drift max ${Math.max(...Object.values(VR.psi))}`],
    ["Unit and integration tests", "27 passed", "pytest, mock LLM; live test optional"],
  ];
  s.addTable([[hdr("Metric"), hdr("Value"), hdr("Note")]].concat(rows.map((r) => [c(r[0], { bold: true }), c(r[1], { color: C.accent1, bold: true }), c(r[2], { color: C.accent5 })])),
    { x: 0.6, y: 1.45, w: 12.1, colW: [5.2, 3.4, 3.5], rowH: 0.52, border: { type: "solid", color: "D5DEE4", pt: 0.75 }, valign: "middle", margin: [0.03, 0.12, 0.03, 0.12] });
}
{
  const s = slide("Guardrail funnel and escalations", "Counts per journey outcome from the replay; the funnel sums to the 120 journeys.", "Overall and performance");
  const lab = { first_pass: "Passed first draft", retried_then_passed: "Blocked, retried, passed", no_offer: "No eligible offer", hard_stop_guest_not_opted_in: "Hard stop: not opted in", escalated_hitl: "Escalated to human", hard_stop_frequency_cap_exceeded: "Hard stop: frequency cap" };
  const ent = Object.entries(F).sort((a, b) => b[1] - a[1]);
  s.addChart(pres.charts.BAR, [{ name: "Journeys", labels: ent.map((e) => lab[e[0]] || e[0]), values: ent.map((e) => e[1]) }],
    Object.assign(chartBase(), { x: 0.6, y: 1.45, w: 7.6, h: 5.3, barDir: "bar", chartColors: [H.accent1], catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "outEnd", showLegend: false, valGridLine: { style: "none" }, valAxisHidden: true, barGapWidthPct: 45 }));
  T(s, "Why journeys escalate", 8.7, 1.5, 4, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  tile(s, 8.7, 2.05, 4.0, 1.25, `${M.hitl_reasons.repeated_policy_failure}`, "repeated policy failure (retries exhausted)", C.accent4, 30);
  tile(s, 8.7, 3.4, 4.0, 1.25, `${M.hitl_reasons.negative_sentiment}`, "negative-sentiment replies", C.accent4, 30);
  tile(s, 8.7, 4.75, 4.0, 1.25, `${M.hitl_reasons.vip_platinum_custom_request}`, "Platinum custom requests", C.accent4, 30);
  T(s, `Total ${Object.values(F).reduce((a, b) => a + b, 0)} journeys`, 8.7, 6.2, 4, 0.4, { fontSize: 12, color: C.accent5 });
}
{
  const s = slide("Data and model metrics", "Synthetic data with a hidden behaviour model; XGBoost trained on GPU.", "Overall and performance");
  T(s, "Datasets (validated)", 0.6, 1.5, 5, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  const rc = VR.row_counts;
  tile(s, 0.6, 2.0, 2.0, 1.2, fmt(rc.guests), "guests (CRM)", C.accent1, 26);
  tile(s, 2.75, 2.0, 2.0, 1.2, fmt(rc.reservations), "reservations (PMS)", C.accent1, 26);
  tile(s, 0.6, 3.35, 2.0, 1.2, fmt(rc.offer_history), "offer outcomes", C.accent1, 26);
  tile(s, 2.75, 3.35, 2.0, 1.2, fmt(rc.inventory), "inventory rows", C.accent1, 26);
  tile(s, 0.6, 4.7, 4.15, 1.2, `${VR.n_passed}/${VR.n_checks} checks`, "schema, nulls, ranges, FK, consistency, PSI drift", C.accent6, 26);
  T(s, "XGBoost NBA (GPU)", 5.1, 1.5, 4, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  [[`${MM.roc_auc}`, "ROC-AUC"], [`${MM.pr_auc}`, `PR-AUC (base ${MM.base_rate})`], [`${MM.ks}`, "KS statistic"], [`${MM.brier}`, `Brier (const ${MM.brier_baseline})`], [`${MM.top_decile_lift}x`, "top-decile lift"], [`${TI.device} / ${TI.tree_method}`, `${TI.best_iteration} trees, early stopped`]].forEach((t, i) => tile(s, 5.1 + (i % 2) * 1.95, 2.0 + Math.floor(i / 2) * 1.35, 1.85, 1.2, t[0], t[1], C.accent3, 22));
  const fi = MM.feature_importance.slice(0, 7);
  s.addChart(pres.charts.BAR, [{ name: "Gain", labels: fi.map((f) => f.feature), values: fi.map((f) => f.gain) }],
    Object.assign(chartBase(), { x: 9.1, y: 1.45, w: 3.7, h: 5.2, barDir: "bar", chartColors: [H.accent3], catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, showLegend: false, valGridLine: { style: "none" }, valAxisHidden: true, showTitle: true, title: "Top features (gain)", barGapWidthPct: 40 }));
}
{
  const s = slide("Model quality: ROC and calibration", `AUC ${MM.roc_auc}, KS ${MM.ks}. Moderate by design: labels come from a noisy behaviour model with a per-guest effect the model cannot see.`, "Overall and performance");
  pic(s, "outputs/charts/05_roc_curve.png", 0.6, 1.5, 5.9, 4.6, "ROC curve");
  pic(s, "outputs/charts/07_calibration.png", 6.8, 1.5, 5.9, 4.6, "Calibration plot");
  T(s, `ROC-AUC ${MM.roc_auc} and KS ${MM.ks} indicate useful ranking (top-decile lift ${MM.top_decile_lift}x); Brier ${MM.brier} beats the constant-rate baseline ${MM.brier_baseline}. The model is deliberately not over-fit: it never sees the hidden guest effect.`, 0.6, 6.25, 12.1, 0.7, { fontSize: 13, color: C.accent5 });
}
{
  const s = slide("Performance: where the time goes", "Node latency from telemetry of the replay. Drafting with the local 30B LLM dominates.", "Overall and performance");
  const nodes = ["enrich_context", "retrieve_knowledge_and_offers", "draft_message", "compliance_check", "hitl_gate", "dispatch_and_log"];
  const tot = nodes.reduce((a, n) => a + NL[n].mean_ms, 0);
  s.addChart(pres.charts.BAR, [{ name: "Mean (ms)", labels: nodes, values: nodes.map((n) => NL[n].mean_ms) }, { name: "p95 (ms)", labels: nodes, values: nodes.map((n) => NL[n].p95_ms) }],
    Object.assign(chartBase(), { x: 0.6, y: 1.45, w: 7.8, h: 5.3, barDir: "bar", chartColors: [H.accent1, H.accent2], catAxisOrientation: "maxMin", showValue: true, dataLabelPosition: "outEnd", showLegend: true, legendPos: "b", valGridLine: { style: "none" }, valAxisHidden: true, barGapWidthPct: 40, dataLabelFormatCode: "#,##0" }));
  tile(s, 8.8, 1.5, 3.9, 1.25, `${M.avg_journey_s} s`, "average pre-arrival journey (end to end)", C.accent1, 28);
  tile(s, 8.8, 2.9, 3.9, 1.25, `${Math.round((100 * NL.draft_message.mean_ms) / tot)}%`, "of node time is Qwen3-VL drafting", C.accent3, 28);
  tile(s, 8.8, 4.3, 3.9, 1.25, `${fmt(M.wall_seconds)} s`, `whole replay: ${M.n_journeys} journeys, ${M.inbound_replies} replies, ${M.messages_judged} LLM judge calls`, C.accent5, 28);
  T(s, `GPU: ${M.gpu.name}, ${M.gpu.total_vram_gb} GB. XGBoost on ${TI.device}; embeddings and reranker FP16. Retrieval mean includes one-time model load (median ${NL.retrieve_knowledge_and_offers.p50_ms} ms).`, 8.8, 5.7, 3.9, 1.1, { fontSize: 12, color: C.accent5 });
}
{
  const s = slide("Revenue impact: measured honestly", "Expected value = P(accept) x final price using the hidden behaviour model; baseline = static SPA-RELAX at list price to every opted-in guest. Synthetic data, so a mechanics demo not a real uplift claim.", "Overall and performance");
  const rn = REV.room_nights;
  s.addChart(pres.charts.BAR, [
    { name: "Static baseline", labels: ["Revenue / room-night ($)", "Margin / room-night ($)"], values: [REV.baseline_anc_revpar, +(REV.baseline_exp_margin_usd / rn).toFixed(3)] },
    { name: "Agent", labels: ["Revenue / room-night ($)", "Margin / room-night ($)"], values: [REV.agent_anc_revpar, +(REV.agent_exp_margin_usd / rn).toFixed(3)] }],
    Object.assign(chartBase(), { x: 0.6, y: 1.45, w: 6.6, h: 5.2, barDir: "col", chartColors: [H.accent5, H.accent1], showValue: true, dataLabelPosition: "outEnd", showLegend: true, legendPos: "b", valGridLine: { style: "none" }, valAxisHidden: true, barGapWidthPct: 60, dataLabelFormatCode: "0.00", showTitle: true, title: `Expected value per room-night (${rn} room-nights)` }));
  tile(s, 7.6, 1.5, 2.45, 1.4, `${sign(REV.anc_revpar_lift_pct)}%`, "ancillary revenue vs baseline", C.accent2, 28);
  tile(s, 10.25, 1.5, 2.45, 1.4, `${sign(REV.margin_lift_pct)}%`, "contribution margin vs baseline", C.accent2, 28);
  tile(s, 7.6, 3.05, 2.45, 1.4, `$${fmt(REV.est_ota_commission_avoided_usd)}`, `est. OTA commission avoided (${M.loyalty_offers_sent} loyalty offers)`, C.accent6, 28);
  tile(s, 10.25, 3.05, 2.45, 1.4, `${sign(REV.realized_lift_pct)}%`, "realized (sampled) revenue vs baseline", C.accent5, 28);
  T(s, bullets([
    "The agent withholds messages (opt-out, caps, no offer), so raw volume is lower than a blast-everyone baseline.",
    "It also discounts when demand is soft, trading revenue for conversion and loyalty value.",
    "Commission avoided is reported separately and is not in the revenue bars.",
  ], { paraSpaceAfter: 4 }), 7.6, 4.65, 5.1, 2.2, { fontSize: 13 });
}

// ================================================================== UI AS AGENT
pres.addSection({ title: "UI as agent" });
{
  const s = slide("End-to-end UI flow: operating the agent from Streamlit", "The Control Center is a thin client over the Flask REST API; every click is a real workflow run.", "UI as agent");
  const tabs = [["Live Trigger", "Pick a reservation, run the workflow, see offer + message + guardrail table", C.accent1], ["Use Cases", "Five one-click end-to-end sample flows", C.accent3], ["State Trace", "Node-by-node status and latency; retry loop visible", C.accent4], ["HITL Queue", "Review, edit, approve or reject escalations", C.accent2], ["Data Lineage", "Lineage graph, table explorer, validation results", C.accent5], ["Metrics & Logs", "Live metrics from /metrics and JSONL logs", C.accent6]];
  const w = 1.85, g = 0.2;
  tabs.forEach((t, i) => {
    const x = 0.6 + i * (w + g);
    s.addShape(S.roundRect, { x, y: 1.55, w, h: 1.0, fill: { color: t[2] }, line: { color: t[2] }, rectRadius: 0.1 });
    T(s, `${i + 1}  ${t[0]}`, x + 0.1, 1.55, w - 0.2, 1.0, { fontSize: 14, bold: true, color: i === 3 ? C.text1 : C.background1, align: "center", valign: "middle" });
    T(s, t[1], x, 2.7, w, 1.3, { fontSize: 12, color: C.text1 });
    if (i < 5) arrow(s, x + w + 0.01, 2.05, g - 0.02, 0);
  });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
  const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 12, color: C.text1 }, o) });
  s.addTable([[hdr("UI action"), hdr("REST call"), hdr("Agent behaviour")],
    [c("Run pre-arrival workflow"), c("POST /api/v1/trigger"), c("Six-node graph; offer, draft, gate, send")],
    [c("Send inbound reply"), c("POST /api/v1/reply"), c("Classify intent, check inventory, book to folio or escalate")],
    [c("Approve / reject in queue"), c("POST /api/v1/hitl/approve"), c("Resume paused graph from its checkpoint")],
    [c("Run use case end-to-end"), c("POST /api/v1/scenarios/{id}/run"), c("Auto-pick reservation, run multi-step flow")],
    [c("Metrics tab"), c("GET /api/v1/metrics, /health"), c("Latest metrics, LLM / GPU / PMS status")]],
    { x: 0.6, y: 4.1, w: 12.1, colW: [3.6, 4.1, 4.4], rowH: 0.48, border: { type: "solid", color: "D5DEE4", pt: 0.75 }, valign: "middle", margin: [0.03, 0.1, 0.03, 0.1] });
}
function twoShots(title, a, b, notes) {
  const s = slide(title, notes, "UI as agent");
  [a, b].forEach((it, i) => {
    const x = 0.6 + i * 6.15;
    const r = pic(s, it.file, x, 1.45, 5.9, 4.85, it.alt);
    T(s, it.cap, x, 6.4, 5.9, 0.55, { fontSize: 12, color: C.accent5, align: "center" });
  });
  return s;
}
twoShots("UI flow: trigger a journey, then reply as the guest",
  { file: "outputs/ui_screens/02_live_trigger_result.png", alt: "Live trigger result", cap: "1. Run the workflow: RMS offer, Qwen message, guardrail check table" },
  { file: "outputs/ui_screens/03_two_way_reply_booking.png", alt: "Two-way reply booking", cap: "2. Guest replies YES: live inventory check, folio posting, status BOOKED" },
  "Screens are captured by Playwright from the live Streamlit app.");
twoShots("UI flow: inspect the trace and review escalations",
  { file: "outputs/ui_screens/04_state_trace.png", alt: "State trace", cap: "3. State Trace: node statuses and latency; repeated draft/compliance = retry loop" },
  { file: "outputs/ui_screens/05_hitl_queue.png", alt: "HITL queue", cap: "4. HITL Queue: edit, approve or reject; approval resumes the paused graph" },
  "The queue is backed by the hitl_queue table and the SQLite LangGraph checkpointer.");
twoShots("UI flow: lineage and live metrics",
  { file: "outputs/ui_screens/06_data_lineage.png", alt: "Data lineage", cap: "5. Data Lineage: source-to-message graph, table explorer, validation results" },
  { file: "presentation/assets/07_metrics_top.png", alt: "Metrics and logs", cap: "6. Metrics & Logs: live KPIs, funnel, latency, structured logs" },
  "The metrics tab reads /api/v1/metrics.");

// ================================================================== USE CASES
pres.addSection({ title: "Use cases" });
const UC = [
  { id: "uc1_upsell_accept", title: "Pre-arrival upsell, guest accepts", persona: "Any opted-in guest with a high-propensity ancillary", flow: ["Trigger pre-arrival offer", "Guest replies YES", "Live inventory check, folio posting"], out: "SENT, then BOOKED on the folio", why: "Shows the full revenue path: GPU scoring, RMS price, Qwen copy, verified send, two-way booking." },
  { id: "uc2_ota_to_direct", title: "OTA guest converted to direct loyalty member", persona: "OTA booking, no loyalty status; loyalty offer ranks first", flow: ["Channel-shift offer (free Azure Circle membership)", "Guest accepts", "Membership posted to the folio"], out: "SENT loyalty offer, membership booked", why: "Targets the 15-25% OTA commission leakage with a fixed benefit from the RMS payload." },
  { id: "uc3_guardrail_retry", title: "Hallucinated price blocked, redraft passes", persona: "Any opted-in guest; first draft deliberately poisoned", flow: ["Draft claims fake SKU, $49 and 40% off", "Gate flags violations (FAIL_POLICY)", "Redraft passes, message sent (retries = 1)"], out: "Blocked, retried, PASS, SENT", why: "The poisoned text never reaches the guest; the violations are shown in the UI." },
  { id: "uc4_policy_escalation", title: "Repeated failure escalates to a concierge", persona: "Any opted-in guest; every draft poisoned", flow: ["Retries exhausted, graph pauses at hitl_gate", "Safe fallback draft queued", "Concierge approves, graph resumes"], out: "ESCALATED, then SENT with the safe fallback", why: "Proves HITL interrupt and resume from the SQLite checkpoint." },
  { id: "uc5_vip_custom_request", title: "Platinum VIP custom request to concierge", persona: "Platinum loyalty guest", flow: ["Offer sent", "Guest asks for a private beach dinner", "Agent holds, concierge approves reply"], out: "ESCALATED (vip_platinum_custom_request)", why: "The agent never improvises bespoke promises; it routes them to a human." },
];
{
  const s = slide("Five sample use cases, end to end", "Defined in src/scenarios.py; run from the Use Cases tab or POST /api/v1/scenarios/{id}/run.", "Use cases");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
  const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 12, color: C.text1 }, o) });
  s.addTable([[hdr("#"), hdr("Use case"), hdr("Persona"), hdr("Expected outcome")]].concat(UC.map((u, i) => [c(String(i + 1), { bold: true }), c(u.title, { bold: true }), c(u.persona), c(u.out, { color: C.accent1, bold: true })])),
    { x: 0.6, y: 1.5, w: 12.1, colW: [0.5, 4.0, 4.0, 3.6], rowH: [0.45, 0.8, 0.8, 0.8, 0.8, 0.8], border: { type: "solid", color: "D5DEE4", pt: 0.75 }, valign: "middle", margin: [0.04, 0.1, 0.04, 0.1] });
  T(s, "Each run auto-selects an unmessaged upcoming reservation pre-screened with the real offer engine, then drives the real LangGraph workflow.", 0.6, 6.3, 12.1, 0.6, { fontSize: 13, color: C.accent5 });
}
UC.forEach((u, i) => {
  const s = slide(`Use case ${i + 1}: ${u.title}`, u.why, "Use cases");
  T(s, "Persona", 0.6, 1.5, 3, 0.3, { fontSize: 12, bold: true, color: C.accent5 });
  T(s, u.persona, 0.6, 1.8, 4.6, 0.65, { fontSize: 15 });
  T(s, "Flow", 0.6, 2.6, 3, 0.3, { fontSize: 12, bold: true, color: C.accent5 });
  u.flow.forEach((f, j) => {
    const y = 2.95 + j * 0.88;
    s.addShape(S.ellipse, { x: 0.6, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1 } });
    T(s, String(j + 1), 0.6, y + 0.05, 0.5, 0.5, { fontSize: 14, bold: true, color: C.background1, align: "center", valign: "middle" });
    T(s, f, 1.3, y, 3.9, 0.7, { fontSize: 14, valign: "middle" });
  });
  s.addShape(S.roundRect, { x: 0.6, y: 5.7, w: 4.6, h: 1.15, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.1 });
  T(s, "Outcome", 0.8, 5.78, 3, 0.3, { fontSize: 12, bold: true, color: C.accent5 });
  T(s, u.out, 0.8, 6.08, 4.2, 0.7, { fontSize: 15, bold: true, color: C.accent1 });
  pic(s, `presentation/assets/${u.id}_crop.png`, 5.55, 1.5, 7.15, 5.4, `Streamlit screenshot of use case ${i + 1}`);
});

// ================================================================== WRAP-UP
pres.addSection({ title: "Wrap-up" });
{
  const s = slide("Quality, reproducibility and how to run", "One command regenerates data, validation, training, replay, charts, screenshots and README.", "Wrap-up");
  T(s, "Quality gates", 0.6, 1.5, 5.5, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  T(s, bullets(["27 pytest tests: device fallback, data validator (incl. corruption), step validators, RMS rules, hallucination gate, retry loop, HITL resume and reject, inbound booking, Flask endpoints", "Seeded generator and replay; temperature-0 judging", "README and this deck are generated from the real metrics files"]), 0.6, 2.0, 5.8, 4.5, { fontSize: 16 });
  T(s, "Run it", 6.9, 1.5, 5.5, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  s.addShape(S.roundRect, { x: 6.9, y: 2.0, w: 5.8, h: 2.9, fill: { color: H.dk2 }, line: { color: H.dk2 }, rectRadius: 0.1 });
  T(s, "pip install -r requirements.txt\nplaywright install chromium\ncp .env.example .env\npython run_all.py\npython run_all.py --mock-llm\nstreamlit run src/ui/streamlit_app.py\npytest -q", 7.15, 2.15, 5.4, 2.6, { fontSize: 14, color: "E6EEF3", fontFace: "Courier New" });
  T(s, "Local LLM: LM Studio at http://localhost:1234/v1, model qwen3-vl-30b-a3b-instruct.", 6.9, 5.1, 5.8, 0.7, { fontSize: 13, color: C.accent5 });
}
{
  const s = slide("Limitations and next steps", "Be explicit about what the simulation can and cannot claim.", "Wrap-up");
  T(s, "Known limitations", 0.6, 1.5, 5.8, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  T(s, bullets([
    `Synthetic data and a hidden behaviour model: revenue results are mechanics, not forecasts.`,
    `XGBoost AUC ${MM.roc_auc} is moderate; discounts and category dominate the signal.`,
    `Agent is ${sign(REV.anc_revpar_lift_pct)}% on ancillary revenue per room-night vs a blast baseline; the benefit is margin discipline, safety and commission avoided.`,
    "Single-property, single-LLM, one-process SQLite; HITL reviewer simulated in the replay.",
  ]), 0.6, 2.0, 5.9, 4.8, { fontSize: 14 });
  T(s, "Next steps", 6.9, 1.5, 5.8, 0.4, { fontSize: 18, bold: true, color: C.text2 });
  T(s, bullets([
    "Uplift modeling and an A/B hold-out to tune the discount ladder against revenue, not just margin.",
    "Real PMS / CRM / RMS adapters and a production vector store.",
    "Use Qwen3-VL vision for guest-photo triage and property imagery.",
    "Drift and hallucination monitoring dashboards from the JSONL telemetry.",
    "Multi-property rollout with per-brand voice packs.",
  ]), 6.9, 2.0, 5.8, 4.8, { fontSize: 14 });
}

// ------------------------------------------------------------------ write
const out = path.join(__dirname, "Revenue_Agent_Deck.pptx");
pres.writeFile({ fileName: out }).then(() => applyTheme(out, THEME)).then(() => console.log("wrote", out));
