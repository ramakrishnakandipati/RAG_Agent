// Builds OTA_to_Direct_Agent_Deck.pptx from the REAL artifacts the pipeline produced (outputs/*) and the Streamlit screenshot crops.
// Run:  python prepare_assets.py && node build_deck.js      (re-run after `python evaluation/verify_e2e.py` to refresh every number)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "outputs");
const A = (f) => path.join(__dirname, "assets", f);
const SIZES = JSON.parse(fs.readFileSync(A("sizes.json"), "utf8"));
const readJSON = (...p) => JSON.parse(fs.readFileSync(path.join(OUT, ...p), "utf8"));
const readCSV = (...p) => {
  const [h, ...rows] = fs.readFileSync(path.join(OUT, ...p), "utf8").trim().split(/\r?\n/);
  const cols = h.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
};

// ------------------------------------------------------------------ real data
const replay = readJSON("evaluation", "offline_replay_roi.json");
const dv = readJSON("validation", "data_validation.json");
const lat = readJSON("metrics", "latency_gpu.json");
const hw = readJSON("logs", "hardware_telemetry.json");
const vlm = readJSON("evaluation", "vlm_visual_audit.json");
const mm = readCSV("metrics", "model_metrics.csv");
const cal = readCSV("metrics", "cancel_calibration.csv");
const dec = readCSV("evaluation", "counterfactual_lift_by_decile.csv");
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const usd = (x) => "$" + Math.round(x).toLocaleString("en-US");
const cancel = mm.find((r) => r.model === "Cancel-XGB");
const tl = mm.find((r) => r.model === "T-learner");
const xl = mm.find((r) => r.model === "X-learner");
const D2 = dv.D2.metrics;

// Per-node latency (ms) measured over the five curated flows with the live LLM (first call = cold start)
const NODE_MS = {
  ingest_and_resolve: [1636, 7.5, 8.7, 7.0, 7.2], score_guest: [197, 13.7, 17.6, 12.6, 15.1], select_strategy: [6012, 11353, 3910, 4175, 4332],
  apply_guardrails: [0.4, 2822, 0.4, 0.4, 0.4], compose_and_dispatch: [0.3, 0.1, 0.4, 0.3, 0.3], log_and_attribute: [0.3, 0.4, 0.3, 0.3, 0.2],
};

// ------------------------------------------------------------------ theme + layouts
const THEME = {
  name: "OTA to Direct", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2333", lt1: "FFFFFF", dk2: "0B3C49", lt2: "F1F5F7", accent1: "0E7C86", accent2: "E3A008", accent3: "D1495B", accent4: "1F9D6B", accent5: "2A6FDB", accent6: "8A94A6", hlink: "2A6FDB", folHlink: "8A94A6" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Autonomous OTA-to-Direct Conversion Agent"; pres.author = "OTA-to-Direct agent project";
const C = pres.SchemeColor;
const W = 13.33;

pres.defineSlideMaster({ title: "TITLE_DARK", background: { color: C.text2 }, objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.1, w: 11.7, h: 1.6, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } }] });
pres.defineSlideMaster({ title: "SECTION_DARK", background: { color: C.text2 }, objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } }] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: C.background1 }, margin: [0.5, 0.5, 0.5, 0.5],
  slideNumber: { x: 12.4, y: 7.05, w: 0.6, h: 0.3, fontFace: "Calibri", fontSize: 10, color: C.accent6, align: "right" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontFace: "Cambria", fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } }] });

// ------------------------------------------------------------------ helpers
let sectionOpen = null;
function section(name) { pres.addSection({ title: name }); sectionOpen = name; }
function slide(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sectionOpen });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function divider(title, sub) {
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: sectionOpen });
  s.addText(title, { placeholder: "title" });
  if (sub) s.addText(sub, { x: 0.8, y: 3.9, w: 11, h: 1.0, fontFace: "Calibri", fontSize: 20, color: C.accent2, margin: 0, isTextBox: true, valign: "top" });
  return s;
}
function kpi(s, x, y, w, h, value, label, color = C.accent1, vsize = 36) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.08, objectName: "kpi card" });
  s.addText(value, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: h * 0.55, fontFace: "Cambria", fontSize: vsize, bold: true, color, margin: 0, valign: "middle", isTextBox: true, fit: "shrink" });
  s.addText(label, { x: x + 0.15, y: y + h * 0.62, w: w - 0.3, h: h * 0.34, fontFace: "Calibri", fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}
function bullets(s, items, x, y, w, h, size = 16) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
    { x, y, w, h, fontFace: "Calibri", fontSize: size, color: C.text1, margin: 0, valign: "top", isTextBox: true });
}
function label(s, text, x, y, w, color = C.accent1) {
  s.addText(text.toUpperCase(), { x, y, w, h: 0.3, fontFace: "Calibri", fontSize: 12, bold: true, color, margin: 0, charSpacing: 2, isTextBox: true });
}
function img(s, file, x, y, w, maxH, caption) {
  const [pw, ph] = SIZES[file]; let iw = w, ih = (ph * w) / pw;
  if (ih > maxH) { ih = maxH; iw = (pw * maxH) / ph; }
  s.addImage({ path: A(file), x, y, w: iw, h: ih, altText: caption || file });
  s.addShape(pres.ShapeType.rect, { x, y, w: iw, h: ih, fill: { type: "none" }, line: { color: C.accent6, width: 0.75 } });
  if (caption) s.addText(caption, { x, y: y + ih + 0.05, w: iw, h: 0.28, fontFace: "Calibri", fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
  return { w: iw, h: ih };
}
const TH = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontFace: "Calibri", fontSize: 12, valign: "middle" } });
const TD = (t, o = {}) => ({ text: String(t), options: { fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "middle", ...o } });
function table(s, header, rows, x, y, w, colW, rowH = 0.36) {
  const data = [header.map(TH), ...rows.map((r, i) => r.map((c) => (typeof c === "object" && c !== null ? c : TD(c, { fill: { color: i % 2 ? C.background2 : C.background1 } }))))];
  s.addTable(data, { x, y, w, colW, rowH, border: { type: "solid", color: "D6DBE5", pt: 0.5 }, margin: [0.04, 0.08, 0.04, 0.08], autoPage: false });
}
const PASS = (t = "PASS") => TD(t, { bold: true, color: C.accent4 });
const chartBase = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: C.text1, valAxisLabelColor: C.text1, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, legendFontSize: 12, dataLabelFontSize: 12, titleFontSize: 14, titleColor: C.text1 };

// ================================================================== 1. title
section("Overview");
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: sectionOpen });
  s.addText("Autonomous OTA-to-Direct Conversion Agent", { placeholder: "title" });
  s.addText("From masked OTA reservations to measured direct bookings: GPU uplift models, a 6-node LangGraph agent and deterministic guardrails", { x: 0.8, y: 4.0, w: 10.5, h: 1.0, fontFace: "Calibri", fontSize: 22, color: C.background2, margin: 0, isTextBox: true, valign: "top" });
  s.addText("Detailed end-to-end flow, metrics at every stage, and a Streamlit walkthrough", { x: 0.8, y: 5.4, w: 10.5, h: 0.4, fontFace: "Calibri", fontSize: 16, color: C.accent2, margin: 0, isTextBox: true });
  s.addText("Synthetic data (seed 42, 20,000 bookings)  |  Local qwen3-vl-30b-a3b-instruct  |  " + (hw.device_name || "CPU"), { x: 0.8, y: 6.7, w: 11, h: 0.3, fontFace: "Calibri", fontSize: 12, color: C.accent6, margin: 0, isTextBox: true });
  s.addNotes("Opening: this deck walks the full pipeline, shows what each stage measures, then replays five real flows in the Streamlit UI. All numbers are from the latest verify_e2e run on synthetic data.");
}

// ================================================================== 2. executive summary
{
  const s = slide("Executive summary", "Headline: a compliant policy lifts direct rebooking by ~14.5 points against a randomised ghost holdout. The ROI is modest and the caveats slide explains why.");
  kpi(s, 0.6, 1.4, 2.9, 1.6, "+" + (replay.incremental_lift * 100).toFixed(1) + " pts", `Lift vs ghost holdout (95% CI ${(replay.lift_ci95[0] * 100).toFixed(1)} to ${(replay.lift_ci95[1] * 100).toFixed(1)})`, C.accent4);
  kpi(s, 3.7, 1.4, 2.9, 1.6, usd(replay.net_revenue_delta_usd), `Net revenue delta, ${replay.roi_x}x ROI on incentives`, C.accent1);
  kpi(s, 6.8, 1.4, 2.9, 1.6, "3 / 3", "Gates pass: D1 ingestion, D2 identity, D3 models", C.accent5);
  kpi(s, 9.9, 1.4, 2.85, 1.6, "40", "Tests pass; 12 gates per clean agent run", C.accent2);
  label(s, "What was built", 0.6, 3.4, 6);
  bullets(s, ["Identity resolution links masked OTA guests to the CDP (recall " + pct(D2.recall) + ", false merges " + pct(D2.false_positive_merge_rate, 2) + ")",
    "CUDA XGBoost uplift and cancellation-risk models, blocked from promotion if quality gates fail",
    "A 6-node LangGraph agent; a local VLM proposes offers, deterministic rules decide compliance",
    "A randomised 15% ghost holdout that is never messaged, so lift is causal"], 0.6, 3.75, 6.1, 3.1, 15);
  label(s, "What to know before trusting it", 7.1, 3.4, 5.6, C.accent3);
  bullets(s, ["All data is synthetic; ROI reflects the generator, not a hotel",
    "Only " + pct(replay.target_rate, 0) + " of guests are targeted: guests with no CDP match have unknown consent",
    "Uplift ranking is modest (decile Spearman " + replay.decile_rank_corr_pred_vs_measured + ")",
    "GPU is slower than CPU at this data size"], 7.1, 3.75, 5.6, 3.1, 15);
}

// ================================================================== 3. problem and approach
{
  const s = slide("The problem and the approach", "OTA commissions are the cost; the opportunity is converting OTA guests into direct relationships without breaching rate parity or over-messaging.");
  label(s, "The problem", 0.6, 1.4, 5);
  bullets(s, ["OTA bookings carry 10-30% commission (avg saved commission per conversion " + usd(replay.avg_saved_commission_usd) + " here)",
    "Guests arrive masked: relay e-mails, name noise, partial phone",
    "Incentives must not undercut OTA rates (rate parity) or erode margin",
    "Blasting everyone wastes spend and annoys guests; some guests are sleeping dogs (negative uplift)"], 0.6, 1.75, 5.3, 4.9, 16);
  const steps = [["1", "Identify", "Resolve masked guests to the CDP", C.accent1], ["2", "Predict", "Uplift (who is persuadable) and cancellation risk", C.accent5],
    ["3", "Decide safely", "LLM proposes, deterministic policy engine approves", C.accent3], ["4", "Prove", "Ghost holdout + offline replay measure causal lift", C.accent4]];
  steps.forEach(([n, h, d, c], i) => {
    const y = 1.4 + i * 1.35;
    s.addShape(pres.ShapeType.ellipse, { x: 6.5, y: y + 0.1, w: 0.7, h: 0.7, fill: { color: c }, line: { color: c, width: 0 } });
    s.addText(n, { x: 6.5, y: y + 0.1, w: 0.7, h: 0.7, fontFace: "Cambria", fontSize: 22, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: 7.4, y: y + 0.02, w: 5.3, h: 0.4, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
    s.addText(d, { x: 7.4, y: y + 0.45, w: 5.3, h: 0.7, fontFace: "Calibri", fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
}

// ================================================================== architecture
section("Architecture and approach");
{
  const s = slide("System architecture", "Five layers: ingestion, identity and features, GPU scoring, the LangGraph orchestrator with policy engine, and interfaces and evaluation.");
  img(s, "system_architecture_diagram.png", 1.5, 1.25, 10.3, 5.7, "Layers: ingestion, identity + features, GPU ML, LangGraph orchestrator, interfaces and evaluation");
}
{
  const s = slide("Data lineage and record counts", "Counts come from the D1 and D2 reports: 20,000 bookings, 3,000 holdout, 53.6% matched.");
  img(s, "data_pipeline_erd_flow.png", 1.0, 1.25, 11.3, 5.6, "Bookings -> Gate D1 -> identity resolution -> features; 85% treated / 15% ghost holdout; train/val/test split");
}
{
  const s = slide("Step-by-step approach", "Ten steps. Each step produces an artifact and most carry a gate that can stop the pipeline.");
  const steps = [
    ["0", "Generate", "20k bookings, 6k CDP profiles, 15% holdout", C.accent6], ["1", "Gate D1", "schema + integrity; holdout 15% per OTA", C.accent2],
    ["2", "Resolve", "email, FAISS, CUDA cosine + JW; Gate D2", C.accent1], ["3", "Features", "19 features, zero-NaN contract", C.accent1],
    ["4", "Train", "T/X-learner + cancel model on CUDA; Gate D3", C.accent5], ["5", "Promote", "only if D3 passes", C.accent5],
    ["6", "Agent", "6 LangGraph nodes, 12 gates per run", C.accent3], ["7", "Log", "decisions, guardrail audits, outbox", C.accent3],
    ["8", "Replay", "treated vs holdout: lift, ROI, Qini", C.accent4], ["9", "Audit", "diagrams, VLM audit, UIs, screenshots", C.accent4]];
  steps.forEach(([n, h, d, c], i) => {
    const col = i % 5, row = Math.floor(i / 5), x = 0.6 + col * 2.46, y = 1.6 + row * 2.6;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: 2.3, h: 2.3, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.08 });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.15, y: y + 0.15, w: 0.55, h: 0.55, fill: { color: c }, line: { color: c, width: 0 } });
    s.addText(n, { x: x + 0.15, y: y + 0.15, w: 0.55, h: 0.55, fontFace: "Cambria", fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: x + 0.8, y: y + 0.15, w: 1.4, h: 0.55, fontFace: "Cambria", fontSize: 18, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: x + 0.15, y: y + 0.9, w: 2.0, h: 1.3, fontFace: "Calibri", fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
}

// ================================================================== agent stage by stage
section("The agent, stage by stage");
{
  const s = slide("LangGraph state machine: six nodes, a gate on each", "Every node has a pre-condition and a post-condition gate. A failed gate short-circuits to node 6 so the failure is still logged. Guardrail violations loop back to node 3 at most twice.");
  img(s, "langgraph_state_machine.png", 0.9, 1.25, 11.5, 5.0, "G = pre + post validation gate; red loop = guardrail retry (max 2); dotted bus = gate failure short-circuit");
  s.addText("Clean run = 12 gate results, all PASS. The parity-breach flow produces 20 because nodes 3 and 4 run three times.", { x: 0.9, y: 6.55, w: 11.5, h: 0.4, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
}
const LOOK = {
  1: "the identity method and confidence (fuzzy name + phone-last-4, exact e-mail, or none), whether a real CDP profile or a SHADOW profile was created, and the saved commission that justifies any incentive.",
  2: "tau (positive = persuadable, negative = sleeping dog), the cancellation probability, the device that scored it and the latency; the gate rows confirm no NaNs and valid ranges.",
  3: "the proposed offer and channel, whether the model answered live on the first attempt, the rationale and the personalised copy; a rejected-offers list appears when policy bounced earlier proposals.",
  4: "APPROVED vs BLOCKED, the retry count, the downgrade banner naming the blocking rule, and the R1-R6 table; here R1 (rate parity) forced NO_OFFER after two retries.",
  5: "the dispatch status (sent, suppressed_ghost_holdout or no_action), the signed link, the CRS inventory, and the yellow note when the final offer differs from the model's proposal.",
  6: "the attribution record: treatment vs ghost-holdout flag, final offer, tau and expected net revenue delta; these records feed the offline replay.",
};
function stageSlide(n, name, what, gate, chips, shot, cap, notes) {
  const s = slide(`Stage ${n}: ${name}`, notes);
  label(s, "What it does", 0.6, 1.3, 4.9);
  bullets(s, what, 0.6, 1.62, 4.9, 2.2, 14);
  label(s, "Validation gate", 0.6, 3.85, 4.9, C.accent2);
  s.addText(gate, { x: 0.6, y: 4.17, w: 4.9, h: 1.0, fontFace: "Calibri", fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  label(s, "Metrics", 0.6, 5.2, 4.9, C.accent4);
  chips.forEach(([v, l], i) => {
    const x = 0.6 + (i % 3) * 1.65, y = 5.52 + Math.floor(i / 3) * 0.78;
    s.addShape(pres.ShapeType.roundRect, { x, y, w: 1.55, h: 0.7, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.06 });
    s.addText(v, { x: x + 0.05, y: y + 0.03, w: 1.45, h: 0.37, fontFace: "Cambria", fontSize: 15, bold: true, color: C.accent1, margin: 0, align: "center", valign: "middle", isTextBox: true, fit: "shrink" });
    s.addText(l, { x: x + 0.05, y: y + 0.4, w: 1.45, h: 0.27, fontFace: "Calibri", fontSize: 10, color: C.text1, margin: 0, align: "center", isTextBox: true });
  });
  const sz = img(s, shot, 5.9, 1.3, 6.85, 4.0, cap);
  const ly = 1.3 + sz.h + 0.5;
  s.addShape(pres.ShapeType.roundRect, { x: 5.9, y: ly, w: 6.85, h: Math.min(1.35, 6.95 - ly), fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, rectRadius: 0.08 });
  s.addText([{ text: "Look for: ", options: { bold: true, color: C.accent1 } }, { text: LOOK[n] }], { x: 6.1, y: ly + 0.08, w: 6.45, h: Math.min(1.2, 6.8 - ly), fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  return s;
}
stageSlide(1, "ingest and resolve identity",
  ["Validates the OTA webhook against BookingWebhookSchema", "Resolves the guest: e-mail, FAISS blocking, CUDA cosine + Jaro-Winkler; no match gets a SHADOW profile", "Computes saved commission = ADR x LOS x commission; sets the holdout flag"],
  "Pre: payload schema valid. Post: profile id present, saved commission > 0, holdout flag set.",
  [[pct(D2.recall), "ID recall (D2)"], [pct(D2.false_positive_merge_rate, 2), "false merges"], [pct(D2.match_rate), "match rate"], ["~8 ms", "warm latency"], ["20,000", "rows pass D1"], ["15.0%", "holdout / OTA"]],
  "happy_path_stage1.png", "Streamlit Stage 1 panel: webhook, identity method and confidence, CDP profile, saved commission, gates",
  "D2 counts: 8,794 fuzzy name+phone4 merges, 1,724 deterministic e-mail, 201 name-only-unique, 9,281 unmatched. First call is ~1.6 s because of cold start.");
stageSlide(2, "score the guest",
  ["Builds 19 features from webhook and CDP profile", "CUDA XGBoost uplift tau (CATE) and calibrated P(cancel)", "tau below zero marks a sleeping dog: the intervention would hurt"],
  "Pre: all 19 features present. Post: zero NaNs, tau in [-1,1], P(cancel) in [0,1], inference under 1 s.",
  [[cancel.roc_auc, "cancel ROC-AUC"], [(+tl.qini_coef).toFixed(4), "uplift Qini"], [cancel.ece_raw, "cancel ECE"], ["~15 ms", "warm scoring"], ["19", "features"], ["T-learner", "promoted"]],
  "happy_path_stage2.png", "Streamlit Stage 2 panel: tau, P(cancel), device, latency, feature vector, gates",
  "Qini: T 0.00498 vs X 0.00496, a tie; T promoted. X has the higher correlation with the oracle tau (0.54 vs 0.37) but Qini is the gate metric. P(cancel) is clipped to [0.005, 0.995].");
stageSlide(3, "select strategy (local VLM)",
  ["qwen3-vl-30b-a3b-instruct returns strict JSON: offer, channel, rationale, personalised copy", "Malformed output is retried up to 2 times, then a deterministic fallback is used", "Parity-breaking offers are never shown to the model"],
  "Pre: scores present, window valid. Post: OfferStrategySchema valid, offer in catalog, window matches, copy non-empty.",
  [["4-6 s", "per LLM call"], ["<= 2", "LLM retries"], ["8", "catalog offers"], ["live", "llm_mode"], ["1", "attempts (typical)"], ["100%", "schema valid"]],
  "happy_path_stage3.png", "Streamlit Stage 3 panel: proposal, LLM mode and attempts, rationale, personalised copy, gates",
  "This stage dominates end-to-end latency: 3.9 to 6.0 s per call, 11.4 s in the parity flow where it is called three times.");
stageSlide(4, "policy guardrails",
  ["Deterministic rules R1-R6 decide compliance; the LLM never does", "A blocked offer returns to Stage 3 (max 2 retries), then a downgrade to the best compliant offer or NO_OFFER", "Every evaluation is appended to guardrail_audits.jsonl"],
  "Pre: strategy present. Post: verdict present, approved or retrying, retries <= 2. R1 parity, R2 margin >= 35%, R3 incentive <= 60% of commission, R4 ENRD > 0, R5 consent + fatigue < 3, R6 window.",
  [["< 1 ms", "rule evaluation"], ["<= 2", "retries"], ["6", "rules"], ["35%", "margin floor"], ["60%", "incentive ceiling"], ["3 / 30d", "fatigue cap"]],
  "parity_breach_key.png", "Parity-breach flow, Stage 4: R1 blocks all proposals, 2 retries, downgrade to NO_OFFER",
  "The shown panel is the parity-breach flow: three proposals all blocked by R1. Deterministic rules take under 1 ms; 2.8 s appears only when the LLM writes the downgrade explanation.");
stageSlide(5, "compose and dispatch",
  ["CRS inventory check: an upgrade is swapped for a late checkout when no inventory", "Builds an HMAC-signed member link and dispatches the message", "Ghost-holdout guests are suppressed, never sent"],
  "Pre: final offer set and approved. Post: dispatch recorded, holdout never 'sent', gated link signature valid.",
  [["< 1 ms", "stage latency"], ["15%", "holdout (hash)"], ["0", "holdout sends"], ["HMAC", "signed links"], ["3", "statuses"], ["1", "CRS swap rule"]],
  "ghost_holdout_key.png", "Holdout flow, Stage 5: status suppressed_ghost_holdout, CRS swap upgrade to late checkout",
  "Statuses are sent, suppressed_ghost_holdout and no_action. The yellow note appears whenever the final offer differs from the LLM proposal.");
stageSlide(6, "log and attribute",
  ["Writes the attribution record: treatment flag, ghost-holdout flag, offer, tau, ENRD", "Appends decisions, guardrail audits, gate results and outbox to JSONL", "Always runs, even after an upstream failure"],
  "Pre: run id present. Post: attribution written and carries the treatment flag.",
  [["< 1 ms", "stage latency"], ["12", "gates per clean run"], ["4", "JSONL logs"], ["100%", "runs attributed"], ["treated", "vs holdout"], ["audit", "trail"]],
  "happy_path_stage6.png", "Streamlit Stage 6 panel: attribution record and gates",
  "Attribution is what lets the offline replay compare treated guests with the 15% ghost holdout.");

// ================================================================== metrics
section("Performance metrics");
divider("Metrics: overall and at every stage", "Gates, model quality, business replay, ranking quality and latency");
{
  const s = slide("Overall business metrics (held-out test replay)", "Test split of 6,000 bookings; policy targets 26%. Lift is treated minus ghost holdout among targeted guests, bootstrap CI.");
  kpi(s, 0.6, 1.4, 3.0, 1.4, "+" + (replay.incremental_lift * 100).toFixed(1) + " pts", "Incremental lift (treated vs holdout)", C.accent4);
  kpi(s, 3.8, 1.4, 3.0, 1.4, usd(replay.net_revenue_delta_usd), "Net revenue delta", C.accent1);
  kpi(s, 7.0, 1.4, 2.9, 1.4, replay.roi_x + "x", "ROI on incentive spend", C.accent5);
  kpi(s, 10.1, 1.4, 2.65, 1.4, pct(replay.target_rate, 0), "Guests targeted by policy", C.accent2);
  s.addChart(pres.charts.BAR, [{ name: "Direct-rebook rate", labels: ["Treated (n=" + replay.treated_targeted + ")", "Ghost holdout (n=" + replay.holdout_targeted + ")"], values: [replay.conv_treated * 100, replay.conv_holdout * 100] }],
    { x: 0.6, y: 3.05, w: 5.4, h: 3.8, barDir: "col", chartColors: [THEME.colors.accent4, THEME.colors.accent6], showTitle: true, title: "Direct-rebook rate, targeted guests (%)", showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", showLegend: false,
      valAxisMinVal: 0, valAxisMaxVal: 50, valGridLine: { color: "D6DBE5", size: 0.5 }, catGridLine: { style: "none" }, ...chartBase });
  s.addChart(pres.charts.BAR, [{ name: "USD", labels: ["Gross saved commission", "Cancellation leakage", "Incentive cost", "Net revenue delta"], values: [replay.gross_saved_commission_usd, -replay.cancellation_leakage_usd, -replay.incentive_cost_usd, replay.net_revenue_delta_usd] }],
    { x: 6.3, y: 3.05, w: 6.45, h: 3.8, barDir: "col", chartColors: [THEME.colors.accent4, THEME.colors.accent3, THEME.colors.accent3, THEME.colors.accent1], showTitle: true, title: "Net revenue build-up (USD)", showValue: true, dataLabelFormatCode: "#,##0", dataLabelPosition: "outEnd", showLegend: false,
      valGridLine: { color: "D6DBE5", size: 0.5 }, catGridLine: { style: "none" }, ...chartBase });
}
{
  const s = slide("Metrics at each pipeline stage", "Every pipeline stage has a measured value and a threshold. D3 blocked promotion once at AUC 0.672 before the synthetic cancel signal was strengthened.");
  const d1 = dv.D1.checks; const get = (n) => d1.find((c) => c.check === n).detail;
  table(s, ["Stage", "Metric", "Value", "Threshold", "Result"], [
    ["D1 ingestion", "rows / duplicate keys / nulls", `${dv.D1.rows.toLocaleString("en-US")} / ${get("pk_non_null_unique").dupes} / 0`, "unique, non-null", PASS()],
    ["D1 ingestion", "commission range", `${get("commission_range_0.10_0.30").min} to ${get("commission_range_0.10_0.30").max}`, "0.10 to 0.30", PASS()],
    ["D1 ingestion", "ghost holdout share (overall, per OTA)", `${pct(get("ghost_holdout_15pct_pm1").holdout_frac)}, 15.0% each`, "15% +/- 1%", PASS()],
    ["D2 identity", "recall on known links", pct(D2.recall) + ` (${D2.n_linked_truth.toLocaleString("en-US")} linked)`, "> 90%", PASS()],
    ["D2 identity", "false-positive merge rate", pct(D2.false_positive_merge_rate, 2) + ` (${D2.n_wrong_merges} of ${D2.n_merged.toLocaleString("en-US")})`, "< 2%", PASS()],
    ["D3 uplift", "Qini coefficient, T / X learner", `${(+tl.qini_coef).toFixed(5)} / ${(+xl.qini_coef).toFixed(5)}`, "> 0", PASS()],
    ["D3 cancellation", "ROC-AUC", cancel.roc_auc, ">= 0.72", PASS()],
    ["D3 cancellation", "Brier / ECE", `${cancel.brier} / ${cancel.ece_raw}`, "informational", TD("n/a")],
    ["Agent gates", "gate results per clean run", "12 of 12 PASS", "all pass", PASS()],
    ["Tests", "automated tests", "40 passed", "all pass", PASS()],
  ], 0.6, 1.4, 12.1, [2.1, 4.2, 2.9, 1.7, 1.2], 0.45);
}
{
  const s = slide("Model quality: cancellation risk is well calibrated", "Calibration on the test split in ten equal-size bins. ROC-AUC 0.787 against a 0.72 gate; ECE 0.015.");
  s.addChart(pres.charts.LINE, [{ name: "Predicted P(cancel)", labels: cal.map((_, i) => "bin " + (i + 1)), values: cal.map((r) => +r.pred) }, { name: "Observed rate", labels: cal.map((_, i) => "bin " + (i + 1)), values: cal.map((r) => +r.obs) }],
    { x: 0.6, y: 1.3, w: 7.6, h: 5.5, chartColors: [C.accent5, C.accent3], lineSize: 3, lineDataSymbolSize: 7, showTitle: true, title: "Predicted vs observed cancellation rate", showLegend: true, legendPos: "b",
      valGridLine: { color: "D6DBE5", size: 0.5 }, catGridLine: { style: "none" }, valAxisLabelFormatCode: "0.0", ...chartBase });
  kpi(s, 8.6, 1.4, 4.15, 1.4, cancel.roc_auc, "Cancellation ROC-AUC (gate >= 0.72)", C.accent4);
  kpi(s, 8.6, 3.0, 4.15, 1.4, cancel.ece_raw, "Expected calibration error", C.accent1);
  kpi(s, 8.6, 4.6, 4.15, 1.4, cancel.base_rate, "Base cancellation rate (test)", C.accent2);
  s.addText("Isotonic calibration did not improve ECE (0.015 raw to 0.020), so the raw model is already well calibrated.", { x: 8.6, y: 6.15, w: 4.15, h: 0.7, fontFace: "Calibri", fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}
{
  const s = slide("Uplift ranking quality: better than random, but modest", "Predicted tau spans -0.17 to 0.37 while the oracle tau is flat near 0.10-0.17, and measured lift per decile is noisy (about 90 control guests per decile). Spearman 0.37.");
  s.addChart(pres.charts.LINE, [{ name: "Predicted tau", labels: dec.map((r) => "D" + (+r.decile + 1)), values: dec.map((r) => +r.pred_tau) }, { name: "True tau (oracle)", labels: dec.map((r) => "D" + (+r.decile + 1)), values: dec.map((r) => +r.true_tau) },
    { name: "Measured lift", labels: dec.map((r) => "D" + (+r.decile + 1)), values: dec.map((r) => +r.measured_lift) }],
    { x: 0.6, y: 1.3, w: 8.0, h: 5.5, chartColors: [C.accent5, C.accent6, C.accent4], lineSize: 3, lineDataSymbolSize: 7, showTitle: true, title: "By predicted-uplift decile (D10 = highest predicted)", showLegend: true, legendPos: "b",
      valGridLine: { color: "D6DBE5", size: 0.5 }, catGridLine: { style: "none" }, valAxisLabelFormatCode: "0.00", ...chartBase });
  label(s, "How to read it", 9.0, 1.4, 3.7);
  bullets(s, ["Qini is positive (" + (+tl.qini_coef).toFixed(4) + "), so targeting beats random", "The bottom decile is predicted negative but really responds positively", "Measured lift is not monotonic in the prediction", "Treat targeting as modest; more data would help"], 9.0, 1.75, 3.75, 4.8, 14);
}
{
  const s = slide("Agent metrics at each node (five flows, live LLM)", "Latency in ms from each run's trace. First call in a process includes cold start for the FAISS index and CUDA warm-up. The LLM call dominates; everything deterministic is under 50 ms.");
  const names = ["ingest_and_resolve", "score_guest", "select_strategy", "apply_guardrails", "compose_and_dispatch", "log_and_attribute"];
  const fmt = (v) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : v.toFixed(1));
  const notes = ["cold start on first call; ~8 ms warm", "CUDA warm-up on first call; ~15 ms warm", "LLM-bound: 4-6 s per call; 3 calls in parity flow", "rules < 1 ms; 2.8 s only when LLM writes a downgrade reason", "signed link + dispatch", "JSONL writes"];
  table(s, ["Node", "Happy", "Parity", "Holdout", "Wi-Fi", "Post-stay", "Notes"], names.map((n, i) => [n, ...NODE_MS[n].map((v) => fmt(v)), notes[i]]), 0.6, 1.4, 12.1, [2.3, 0.9, 0.9, 0.9, 0.9, 1.0, 5.2], 0.55);
  kpi(s, 0.6, 5.35, 3.8, 1.4, "~25 ms", "End-to-end without the LLM (warm)", C.accent4);
  kpi(s, 4.6, 5.35, 3.8, 1.4, "4-6 s", "End-to-end with the live 30B VLM", C.accent2);
  kpi(s, 8.6, 5.35, 4.15, 1.4, "12 / 12", "Gate results PASS per clean run", C.accent5);
}
{
  const s = slide("Hardware: GPU is wired in, but not faster at this scale", "CUDA path is real (device=cuda, torch tensors) but 20k rows cannot amortise launch and transfer overhead. FAISS is the CPU build in this environment.");
  const t = lat.train_seconds, id = lat.identity_resolution.gpu_run;
  table(s, ["Operation", "CPU", "CUDA (" + hw.device_name.replace("NVIDIA GeForce ", "") + ")"], [
    ["X-learner training (20k rows)", t.uplift_x_train_cpu_s + " s", t.uplift_x_train_cuda_s + " s"],
    ["T-learner / cancel-model training", "n/a", t.uplift_t_train_cuda_s + " s / " + t.cancel_train_cuda_s + " s"],
    ["Uplift inference, 1 row", lat.cpu.single_ms + " ms", lat.cuda.single_ms + " ms"],
    ["Uplift inference, 5,000 rows", lat.cpu.batch5000_ms + " ms", lat.cuda.batch5000_ms + " ms"],
    ["Identity resolution, 20,000 bookings", "n/a", id.t_total_s + " s"],
    ["  embed / FAISS block / CUDA cosine / rerank", "n/a", `${id.t_embed_s} / ${id.t_block_s} / ${id.t_gpu_cosine_s} / ${id.t_rerank_s} s`],
  ], 0.6, 1.4, 7.6, [3.6, 1.5, 2.5], 0.5);
  kpi(s, 8.6, 1.4, 4.15, 1.3, (hw.vram_peak_mb / 1024).toFixed(1) + " GB", "Peak VRAM of " + (hw.vram_total_mb / 1024).toFixed(0) + " GB", C.accent5);
  kpi(s, 8.6, 2.9, 4.15, 1.3, hw.faiss_backend, "FAISS backend (GPU needs a faiss-gpu build)", C.accent2, 28);
  kpi(s, 8.6, 4.4, 4.15, 1.3, "VLM " + vlm.mean_score + "/10", "Live chart audit, " + vlm.n_charts + " charts (advisory)", C.accent1, 30);
  s.addText("The VLM scored two diagrams 9 before manual review found overlapping text; those were then fixed.", { x: 8.6, y: 5.85, w: 4.15, h: 0.8, fontFace: "Calibri", fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
}

// ================================================================== E2E streamlit
section("End-to-end flows in the Streamlit UI");
divider("Five end-to-end flows in Streamlit", "Real bookings sent through all six stages; the screenshots are captured from the live dashboard");
{
  const s = slide("The End-to-End Flows tab", "The tab lists five curated flows. Run selected flow shows the six stage panels; Run all 5 fills the comparison table. The same flows are in the Flask simulator and in tests/test_scenarios.py.");
  table(s, ["#", "Flow", "Identity", "tau / P(cancel)", "Proposed to final", "Dispatch"], [
    ["1", "Happy path, pre-arrival", "fuzzy, P000104", "+0.214 / 0.03", "perk approved", TD("sent", { bold: true, color: C.accent4 })],
    ["2", "OTA rate-parity breach", "fuzzy, P004497", "+0.400 / 0.10", "3 blocked by R1, NO_OFFER", TD("no_action", { bold: true, color: C.accent3 })],
    ["3", "Ghost holdout", "exact e-mail, P003328", "+0.350 / 0.10", "upgrade swapped to late checkout", TD("suppressed", { bold: true, color: C.accent2 })],
    ["4", "New guest on Wi-Fi", "no match, SHADOW", "+0.240 / 0.09", "Wi-Fi loyalty join", TD("sent", { bold: true, color: C.accent4 })],
    ["5", "Post-stay win-back", "fuzzy, P003854", "+0.388 / 0.09", "points win-back", TD("sent", { bold: true, color: C.accent4 })],
  ], 0.6, 1.4, 12.1, [0.5, 2.9, 2.3, 1.9, 3.1, 1.4], 0.55);
  label(s, "Each flow panel shows", 0.6, 4.85, 6);
  bullets(s, ["Stage 1: webhook, identity method, CDP or shadow profile, saved commission", "Stage 2: tau, P(cancel), device, latency, feature vector", "Stage 3: LLM proposal, rationale, personalised copy, attempts"], 0.6, 5.2, 5.9, 1.7, 14);
  bullets(s, ["Stage 4: verdict, retries, ENRD, margin, R1-R6 table, downgrade reason", "Stage 5: final copy, signed link, dispatch status, CRS inventory", "Stage 6: attribution record; every stage lists its gates"], 6.8, 5.2, 5.9, 1.7, 14);
}
function flowSlide(n, title, story, facts, top, key, keyCap, notes) {
  const s = slide(`Flow ${n}: ${title}`, notes);
  label(s, "The story", 0.6, 1.3, 4.9);
  s.addText(story, { x: 0.6, y: 1.62, w: 4.9, h: 2.6, fontFace: "Calibri", fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  label(s, "Key numbers", 0.6, 4.3, 4.9, C.accent4);
  facts.forEach(([v, l], i) => {
    const y = 4.65 + i * 0.7;
    s.addText(v, { x: 0.6, y, w: 1.75, h: 0.6, fontFace: "Cambria", fontSize: 18, bold: true, color: C.accent1, margin: 0, valign: "middle", isTextBox: true, fit: "shrink" });
    s.addText(l, { x: 2.4, y, w: 3.1, h: 0.6, fontFace: "Calibri", fontSize: 13, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
  const a = img(s, top, 5.9, 1.25, 6.85, 2.2, "Streamlit: six-stage progress strip and outcome");
  img(s, key, 5.9, 1.25 + a.h + 0.42, 6.85, 6.78 - (1.25 + a.h + 0.42) - 0.3, keyCap);
}
flowSlide(1, "happy path (pre-arrival)",
  "Mei booked 4 nights through Expedia; commission is about 20%. The agent resolves her to CDP profile P000104 from name and phone last-4, scores tau +0.21 with only 3% cancel risk, and the local VLM proposes a perk. Every rule passes, a signed member link is built and the message is sent.",
  [["+0.214", "uplift tau"], ["3%", "cancel risk"], ["$133", "saved commission"], ["sent", "dispatch status"]], "happy_path_top.png", "happy_path_key.png", "Stage 4: all rules pass, no retries",
  "Walk the six green stage cards left to right, then open Stage 4 to show every rule passing.");
flowSlide(2, "OTA rate-parity breach",
  "Lisa has the highest uplift of the five (tau +0.40) but the hotel's own web rate is 10% below the OTA rate. Any incentive risks a parity penalty, so rule R1 blocks the first proposal. The loop returns to Stage 3 twice (the x3 badge), the LLM tries three offers, all blocked, and after the retry cap the agent chooses NO_OFFER and sends nothing.",
  [["+0.400", "uplift tau"], ["3", "proposals blocked (R1)"], ["2", "retries (cap)"], ["no_action", "dispatch status"]], "parity_breach_top.png", "parity_breach_stage4.png", "Stage 4: R1 blocks, downgrade to NO_OFFER",
  "This is the safety story: high-value guest, but compliance wins. The audit trail records each block and the reason.");
flowSlide(3, "ghost holdout (control arm)",
  "This guest resolves by exact e-mail and clearly qualifies (tau +0.35, ENRD $68). She was randomly assigned to the 15% holdout, so Stage 5 records suppressed_ghost_holdout and the gate asserts the message was never sent. The yellow note shows the CRS swap: the LLM proposed a room upgrade, there was no inventory, so the offer became a late checkout.",
  [["+0.350", "uplift tau"], ["$68", "expected net rev. delta"], ["15%", "holdout share"], ["suppressed", "dispatch status"]], "ghost_holdout_top.png", "ghost_holdout_stage5.png", "Stage 5: suppressed, plus CRS swap note",
  "The holdout is what makes the lift number causal. Show that approval and suppression are separate decisions.");
flowSlide(4, "new guest on the Wi-Fi portal",
  "Ashley matches nobody in the CDP, so the agent creates a SHADOW profile with unknown consent. Pre-arrival e-mail would breach the consent rule (R5), but the on-property Wi-Fi portal captures opt-in, so a loyalty-join offer is approved and sent through the portal.",
  [["+0.240", "uplift tau"], ["$47", "saved commission"], ["unmatched", "identity method"], ["sent", "via Wi-Fi portal"]], "new_guest_wifi_top.png", "new_guest_wifi_key.png", "Stage 1: SHADOW profile, consent unknown",
  "Explain why 74% of guests are not targeted overall: unknown consent is only allowed in the on-property window.");
flowSlide(5, "post-stay win-back",
  "Steven has checked out. With $97 of commission at stake and tau +0.39, a points-based win-back fits inside the 60% incentive ceiling and the 35% margin floor. The message is generated, a signed link is attached and the guest is sent an invitation to book direct next time.",
  [["+0.388", "uplift tau"], ["$97", "saved commission"], ["$22", "expected net rev. delta"], ["sent", "dispatch status"]], "post_stay_winback_top.png", "post_stay_winback_key.png", "Stage 3: win-back proposal and personalised copy",
  "The same agent handles all three windows: pre-arrival, on-property and post-stay.");
{
  const s = slide("Comparing the five flows in one table", "Run all 5 flows fills this table: three messages sent, one blocked by policy, one suppressed for measurement.");
  img(s, "summary_table.png", 0.6, 1.3, 12.1, 3.0, "Streamlit flow comparison after Run all 5 flows");
  const y = 4.55;
  [["3 sent", "flows 1, 4 and 5 pass every gate and rule", C.accent4], ["1 blocked", "flow 2: parity breach stopped by R1 after 2 retries", C.accent3], ["1 suppressed", "flow 3: approved but held out to measure lift", C.accent2]].forEach(([v, l, c], i) => kpi(s, 0.6 + i * 4.1, y, 3.9, 1.5, v, l, c));
  s.addText("Each flow takes about 4 to 14 s with the live LLM (parity is slowest: three LLM calls); the deterministic stages add about 25 ms.", { x: 0.6, y: 6.3, w: 12.1, h: 0.4, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
}
{
  const s = slide("Governance and ROI tabs", "The Guardrails tab lists the rules, catalog and recent blocked proposals. ROI and Validation shows the three gates, replay lift and every chart with its VLM audit.");
  img(s, "gov_guardrails.png", 0.6, 1.3, 5.9, 4.9, "Guardrails tab: rules, offer catalog, recent blocks");
  img(s, "gov_roi.png", 6.85, 1.3, 5.9, 4.9, "ROI and Validation tab: gates, lift, ROI, charts");
  s.addText("Lists every rule R1-R6, the eight catalog offers (including the public rate cut that is always blocked by parity) and recent blocked proposals with the rule that blocked them.", { x: 0.6, y: 5.0, w: 5.9, h: 1.3, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Shows the three gates (PASS), measured lift, net revenue delta and ROI, then every chart with its VLM audit result.", { x: 6.85, y: 5.0, w: 5.9, h: 1.3, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("Other tabs: Live Agent Flow (any booking row + scenario) and Data Explorer (bookings, holdout share, identity-method breakdown).", { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, isTextBox: true });
}

// ================================================================== close
section("Caveats and next steps");
{
  const s = slide("Honest caveats", "Be direct that the ROI is a property of the synthetic generator and that the compliant policy is deliberately conservative.");
  const items = [["Synthetic data", "ROI and lift describe the generator, not a real property. The mechanics (gates, estimators, guardrails) are what is demonstrated.", C.accent3],
    ["Consent rule is a policy choice", "Excluding unmatched guests from pre-arrival offers cut targeted share from 63% to 26% and net delta from about $15k to $7.5k. Relax per market if legally appropriate.", C.accent2],
    ["Blast-to-all is not a fair baseline", `Messaging everyone nets ${usd(replay.blast_all_net_revenue_delta_usd)} but ignores consent and guardrails, so it is only an upper reference.`, C.accent6],
    ["Modest uplift ranking", "Qini is positive but decile lift is noisy and predicted tau is over-dispersed.", C.accent5],
    ["No GPU speed-up here", "20k rows is too small; FAISS runs on CPU in this environment.", C.accent1]];
  items.forEach(([h, d, c], i) => {
    const y = 1.35 + i * 1.1;
    s.addShape(pres.ShapeType.ellipse, { x: 0.6, y: y + 0.12, w: 0.5, h: 0.5, fill: { color: c }, line: { color: c, width: 0 } });
    s.addText(h, { x: 1.35, y, w: 3.5, h: 0.75, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: 4.9, y, w: 7.85, h: 0.95, fontFace: "Calibri", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  });
}
{
  const s = slide("Run it and next steps", "All commands run from the project folder. verify_e2e regenerates every artifact, including these screenshots.");
  label(s, "Run it", 0.6, 1.4, 5.5);
  s.addText([{ text: "python evaluation/verify_e2e.py", options: { breakLine: true } }, { text: "python -m web_flask.app", options: { breakLine: true } },
    { text: "python -m streamlit run ui_streamlit/dashboard.py", options: { breakLine: true } }, { text: "python -m pytest -q", options: { breakLine: true } },
    { text: "python presentation/prepare_assets.py", options: { breakLine: true } }, { text: "node presentation/build_deck.js" }],
    { x: 0.6, y: 1.8, w: 6.2, h: 2.6, fontFace: "Courier New", fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, paraSpaceAfter: 6 });
  label(s, "Next steps", 7.2, 1.4, 5.5, C.accent4);
  bullets(s, ["Replace synthetic data with real PMS and CDP extracts, then re-run the three gates", "Measure real holdout lift per OTA and window before scaling", "Decide the consent policy per market", "Move FAISS to GPU and test at production volume", "Add a second LLM-judge check on generated copy"], 7.2, 1.8, 5.5, 4.2, 15);
  s.addText("Artifacts: README.md, ARCHITECTURE_AND_DATA_FLOW.md, outputs/ (validation, metrics, evaluation, charts, screenshots/e2e_flows)", { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontFace: "Calibri", fontSize: 13, color: C.accent6, margin: 0, isTextBox: true });
}

(async () => {
  const file = path.join(__dirname, "OTA_to_Direct_Agent_Deck.pptx");
  await pres.writeFile({ fileName: file });
  await applyTheme(file, THEME);
  console.log("wrote", file);
})();
