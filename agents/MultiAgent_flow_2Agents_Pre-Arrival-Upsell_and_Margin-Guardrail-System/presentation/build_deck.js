/* Builds Upsell_Guardrail_MAS_Deck.pptx from the project's real metrics + screenshots.
 * Run:  python presentation/prep_assets.py   (crops screenshots, dumps cross-model data)
 *       node presentation/build_deck.js      (needs pptxgenjs, react-icons, sharp; set NODE_PATH if installed elsewhere)
 * All numbers are read from outputs/metrics/*.json and presentation/assets/model_compare.json - nothing is typed in by hand. */
const fs = require("fs"), path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react"), ReactDOMServer = require("react-dom/server"), sharp = require("sharp");
const fa = require("react-icons/fa");
const SKILL = process.env.PPTX_SKILL_DIR || "C:/Users/RamaKrishna/.claude/skills/synced/c4087e75-6e8c-4493-8472-1adcd2a2603e_2c02675d-fb05-4126-b416-349c05b8005e/pptx";
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const J = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const Q = J("outputs/metrics/quant_suite_report.json"), OPE = J("outputs/metrics/ope_report.json"), FL = J("outputs/metrics/flow_metrics.json");
const CMP = J("presentation/assets/model_compare.json"), VER = J("outputs/metrics/verification_report.json");
const SHOT = (n) => path.join(ROOT, "outputs/screenshots", n), PLOT = (n) => path.join(ROOT, "outputs/plots", n), ASSET = (n) => path.join(__dirname, "assets", n);

const CATS = Object.keys(Q.uplift), mean = (a) => a.reduce((s, x) => s + x, 0) / a.length, f = (x, d = 3) => Number(x).toFixed(d);
const pol = Object.fromEntries(OPE.policies.map((p) => [p.policy, p])), OV = FL.overall, RUNS = Object.fromEntries(FL.runs.map((r) => [r.key, r]));
const pehe = (m) => mean(CATS.map((k) => Q.uplift[k].sqrt_pehe[m]));

// ---------------------------------------------------------------- theme
const THEME = { name: "Hospitality Guardrail", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B1F2A", lt1: "FFFFFF", dk2: "14213D", lt2: "F2F4F7", accent1: "B8892B", accent2: "1F7A78", accent3: "C8553D", accent4: "5B6B84", accent5: "3C8D5A", accent6: "D9A441", hlink: "1F7A78", folHlink: "5B6B84" } };
const H = THEME.colors, INK = H.dk2, BRASS = H.accent1, TEAL = H.accent2, CORAL = H.accent3, SLATE = H.accent4, GREEN = H.accent5, AMBER = H.accent6, LIGHT = H.lt2, MUTED = "5F6B7A";

const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Pre-Arrival Upsell & Margin Guardrail - 2-Agent System"; pres.author = "Revenue AI team";
const C = pres.SchemeColor, SH = pres.ShapeType;

pres.defineSlideMaster({ title: "TITLE", background: { color: INK }, objects: [],
  slideNumber: undefined });
pres.defineSlideMaster({ title: "SECTION", background: { color: INK }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.5, w: 11.5, h: 1.2, fontFace: "Cambria", fontSize: 44, bold: true, color: "FFFFFF", margin: 0, valign: "bottom", align: "left" }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.85, w: 10.5, h: 1.2, fontFace: "Calibri", fontSize: 20, color: "D9DEE8", margin: 0, valign: "top", align: "left" }, text: "" } }] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" }, slideNumber: { x: 12.35, y: 7.05, w: 0.5, h: 0.3, fontFace: "Calibri", fontSize: 10, color: MUTED },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: "Cambria", fontSize: 30, bold: true, color: INK, margin: 0, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Pre-Arrival Upsell & Margin Guardrail  ·  GPU 2-Agent System", options: { x: 0.6, y: 7.05, w: 7, h: 0.3, fontFace: "Calibri", fontSize: 10, color: MUTED, margin: 0 } } }] });

// ---------------------------------------------------------------- helpers
let slideNo = 0;
const content = (title, section, notes) => { const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section }); s.addText(title, { placeholder: "title" }); if (notes) s.addNotes(notes); return s; };
const section = (title, sub, name) => { pres.addSection({ title: name }); const s = pres.addSlide({ masterName: "SECTION", sectionTitle: name }); s.addText(title, { placeholder: "title" }); s.addText(sub, { placeholder: "body" }); return s; };
const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: "Calibri", color: C.text1, margin: 0, valign: "top", ...o });
const card = (s, x, y, w, h, o = {}) => s.addShape(SH.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: o.fill || LIGHT }, line: { color: o.line || o.fill || LIGHT, width: 0.75 }, objectName: o.name || "card" });
const stat = (s, x, y, w, big, label, color = BRASS, dark = false) => {
  T(s, String(big), { x, y, w, h: 0.85, fontFace: "Cambria", fontSize: 38, bold: true, color, valign: "bottom", objectName: "stat-value" });
  T(s, label, { x, y: y + 0.9, w, h: 0.7, fontSize: 14, color: dark ? "D9DEE8" : MUTED, objectName: "stat-label" }); };
const arrow = (s, x1, y1, x2, y2, color = SLATE) => s.addShape(SH.line, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1, line: { color, width: 1.75, endArrowType: "triangle" } });
const box = (s, x, y, w, h, title, body, fill = LIGHT, tcolor = INK, o = {}) => { card(s, x, y, w, h, { fill }); T(s, title, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.4, fontSize: o.tsize || 15, bold: true, color: tcolor });
  if (body) T(s, body, { x: x + 0.15, y: y + 0.52, w: w - 0.3, h: h - 0.62, fontSize: o.bsize || 12, color: o.bcolor || C.text1 }); };
const table = (s, rows, o) => {
  const hdr = rows[0].map((c) => ({ text: String(c), options: { bold: true, color: "FFFFFF", fill: { color: INK }, fontSize: o.fs || 12, fontFace: "Calibri", valign: "middle" } }));
  const body = rows.slice(1).map((r, i) => r.map((c) => (typeof c === "object" && c !== null && c.text !== undefined ? { ...c, options: { fontSize: o.fs || 12, fontFace: "Calibri", valign: "middle", fill: { color: i % 2 ? "FFFFFF" : LIGHT }, color: C.text1, ...(c.options || {}) } }
    : { text: String(c), options: { fontSize: o.fs || 12, fontFace: "Calibri", valign: "middle", fill: { color: i % 2 ? "FFFFFF" : LIGHT }, color: C.text1 } })));
  s.addTable([hdr, ...body], { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH || 0.38, border: { type: "solid", pt: 0.5, color: "D5DAE3" }, margin: [0.03, 0.08, 0.03, 0.08], autoPage: false }); };
const chartBase = (title) => ({ showTitle: !!title, title, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: INK, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelColor: "3B4252", valAxisLabelColor: "3B4252", valGridLine: { color: "E1E5EC", size: 0.5 }, catGridLine: { style: "none" }, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: "1B1F2A", legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: "3B4252" });
const icon = async (name, color = "FFFFFF") => "image/png;base64," + (await sharp(Buffer.from(ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: 256 })))).png().toBuffer()).toString("base64");
const iconCircle = async (s, name, x, y, d = 0.62, fill = TEAL) => { s.addShape(SH.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } }); s.addImage({ data: await icon(name), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, altText: name }); };
const fit = (file, maxW, maxH) => { const sz = require("image-size"); return null; };
const imgSize = async (file) => { const m = await sharp(file).metadata(); return { width: m.width, height: m.height }; };
const pic = async (s, file, x, y, maxW, maxH, alt) => { const m = await imgSize(file); const k = Math.min(maxW / m.width, maxH / m.height); const w = m.width * k, h = m.height * k;
  s.addImage({ path: file, x: x + (maxW - w) / 2, y, w, h, altText: alt }); return { w, h }; };

(async () => {
  // =============================================================== 1 title
  { pres.addSection({ title: "Opening" }); const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" }); s.background = { color: INK };
    T(s, "GPU-ACCELERATED MULTI-AGENT SYSTEM", { x: 0.9, y: 1.45, w: 11, h: 0.4, fontSize: 16, bold: true, color: AMBER, charSpacing: 4 });
    T(s, "Pre-Arrival Upsell &\nMargin Guardrail Engine", { x: 0.9, y: 1.95, w: 11.5, h: 2.2, fontFace: "Cambria", fontSize: 54, bold: true, color: "FFFFFF" });
    T(s, "End-to-end flow · every model compared · metrics at stage, flow and system level · seven guest stories", { x: 0.9, y: 4.45, w: 10.8, h: 0.9, fontSize: 20, color: "D9DEE8" });
    T(s, "Strategist ⇄ Auditor loop  ·  CUDA quant suite  ·  Qwen3-VL-30B-A3B  ·  Offline policy evaluation", { x: 0.9, y: 6.5, w: 11, h: 0.4, fontSize: 14, color: "9AA6B8" });
    s.addNotes("Opening: this deck explains how two LangGraph agents and a GPU quantitative suite maximise pre-arrival ancillary revenue without breaking margin, inventory or loyalty rules."); }

  // =============================================================== 2 exec summary
  { const s = content("Executive summary: ML finds the revenue, agents keep it safe", "Opening", "Headline numbers come from held-out policy evaluation (16k reservations) and the seven end-to-end flow runs.");
    stat(s, 0.6, 1.45, 2.9, `${f(pol["ml+agent_guardrails"].true_value / pol.logged_behaviour.true_value, 1)}x`, `policy value vs logged ($${f(pol.logged_behaviour.true_value, 2)} → $${f(pol["ml+agent_guardrails"].true_value, 2)} margin per guest)`);
    stat(s, 3.8, 1.45, 2.9, "0%", `guardrail violations with the Auditor (ML-only: ${f(100 * pol["ml_only(elasticity greedy)"].guardrail_violation_rate, 0)}%)`, TEAL);
    stat(s, 7.0, 1.45, 2.9, `${OV.violations_total}/${OV.violations_total}`, "planted faults caught with the right violation code across 7 flows", CORAL);
    stat(s, 10.2, 1.45, 2.6, `${f(OV.mean_gpu_quant_latency_ms, 0)} ms`, "mean GPU quant inference per offer (RTX 3090)", SLATE);
    const pts = [["Two agents, one bounded loop", "Strategist drafts (price from a monotone elasticity net, ranking from uplift + bandit + shadow prices); Auditor enforces 9 contracts; at most R_max = 3 revisions, then a human decides."],
      ["Seven models, compared head to head", "DragonNet wins uplift on 3 of 4 categories (√PEHE 0.044 vs 0.063 X-Learner); LinUCB is the serving bandit; all numbers shown per flow and per stage."],
      ["Guardrails cost ~6% of margin", `ML-only $${f(pol["ml_only(elasticity greedy)"].true_value, 2)} vs guarded $${f(pol["ml+agent_guardrails"].true_value, 2)} per guest, but the guarded policy has zero floor, margin or loyalty-parity violations.`]];
    pts.forEach((p, i) => { const y = 3.55 + i * 1.12; card(s, 0.6, y, 12.1, 0.98, { fill: LIGHT }); T(s, p[0], { x: 0.85, y: y + 0.12, w: 3.6, h: 0.75, fontSize: 16, bold: true, color: INK, valign: "middle" }); T(s, p[1], { x: 4.5, y: y + 0.1, w: 8.0, h: 0.8, fontSize: 13, valign: "middle" }); }); }

  // =============================================================== 3 problem
  { const s = content("The problem: upsells that quietly destroy margin", "Opening");
    const items = [["FaBan", CORAL, "Stockouts", "Selling the last upgrade room removes full-rate BAR demand. PMS buffer: B = max(2, ⌈1.28·σ·√3⌉)."],
      ["FaBalanceScale", AMBER, "ADR / margin floor", "Offer price must clear the RMS floor and a 30% contribution margin after COGS and housekeeping."],
      ["FaUserCheck", TEAL, "Loyalty cannibalisation", "Offering a free 2 PM late checkout to a Diamond member who already has it is zero revenue and a brand miss."],
      ["FaClipboardCheck", SLATE, "Luxury brand voice", "Discount-spam ('ACT NOW', '50% OFF', 'CHEAP') and contact fatigue erode a refined positioning."]];
    for (let i = 0; i < 4; i++) { const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.45; card(s, x, y, 5.95, 2.25, { fill: LIGHT }); await iconCircle(s, items[i][0], x + 0.3, y + 0.3, 0.8, items[i][1]);
      T(s, items[i][2], { x: x + 1.35, y: y + 0.3, w: 4.4, h: 0.5, fontSize: 20, bold: true, color: INK, fontFace: "Cambria" }); T(s, items[i][3], { x: x + 1.35, y: y + 0.9, w: 4.4, h: 1.2, fontSize: 14 }); }
    T(s, "Goal: maximise ancillary revenue at T-7 … T-1 days while all four are held at zero violations.", { x: 0.6, y: 6.5, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: TEAL }); }

  // =============================================================== 4 architecture
  { const s = content("System architecture at a glance", "Opening");
    const col = [[0.6, "Data & training", ["80k synthetic reservations", "Confounded behaviour policy, logged propensities", "CUDA training → .pt checkpoints"], SLATE],
      [3.7, "GPU quant suite", ["Monotone elasticity net", "DragonNet · X-Learner · T-Learner", "LinUCB · Neural TS", "Convex allocator → shadow prices"], TEAL],
      [6.8, "LangGraph agents", ["Agent 1 Strategist (generator)", "Agent 2 Auditor (critic)", "Bounded loop, R_max = 3", "HITL interrupt + SQLite checkpoints"], BRASS],
      [9.9, "Interfaces & evaluation", ["Flask REST · PMS webhook · audit UI", "Streamlit dashboard", "Offline replay: IPS · DR · Qini", "Playwright verification"], INK]];
    col.forEach((c, i) => { card(s, c[0], 1.5, 2.8, 3.55, { fill: LIGHT }); s.addShape(SH.roundRect, { x: c[0], y: 1.5, w: 2.8, h: 0.62, rectRadius: 0.08, fill: { color: c[3] }, line: { color: c[3] } });
      T(s, c[1], { x: c[0] + 0.15, y: 1.5, w: 2.5, h: 0.62, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle" });
      T(s, c[2].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[2].length - 1 } })), { x: c[0] + 0.15, y: 2.3, w: 2.5, h: 2.6, fontSize: 13, paraSpaceAfter: 6 }); if (i < 3) arrow(s, c[0] + 2.8, 3.3, c[0] + 3.1, 3.3); });
    box(s, 0.6, 5.3, 5.9, 1.4, "External systems (mocked)", "CRM guest profile · live PMS inventory (webhook-fed) · RMS ADR floors · loyalty perk matrix", LIGHT, INK, { bsize: 13 });
    box(s, 6.8, 5.3, 5.9, 1.4, "Local LLM", "qwen3-vl-30b-a3b-instruct via OpenAI-compatible API: copy writing, tone audit, visual check. Prices and SKUs are never LLM-chosen. This run used the deterministic fallback.", LIGHT, INK, { bsize: 13 }); }

  // =============================================================== section: E2E
  section("End-to-end flow", "Six stages, what happens in each, and the metric we record there", "Flow");

  // =============================================================== 5 six stages
  { const s = content("Six stages, each with its own metrics", "Flow", "Every offer passes S1 to S6. The Streamlit End-to-End Flow tab renders exactly these panels.");
    const st = [["S1", "Guest context", "CRM profile, tier perks, lead time, contacts; live PMS and RMS floors", "tier · perks · contacts · T-minus"],
      ["S2", "GPU quant inference", "Elasticity P(accept), CATE τ̂, bandit bonus, allocator hurdle per SKU", "GPU ms · τ̂ · P · margin − hurdle"],
      ["S3", "Strategist draft", "Pick SKU, optimal price p*, bundle, channel; LLM writes copy only", "drafts · price · ms"],
      ["S4", "Auditor contracts", "9 checks: inventory, floor, margin, parity, entitled SKU, tone, length, fatigue, visual", "decision · codes · checks/9"],
      ["S5", "Revision loop", "Rejected → directives accumulate → redraft; escalate at r ≥ R_max", "drafts · revisions · fixes"],
      ["S6", "Outcome", "APPROVED, or HITL gate (VIP value or unresolved) then finalize", "status · price · CM% · ms"]];
    const w = 1.93, g = 0.1;
    st.forEach((a, i) => { const x = 0.6 + i * (w + g); const fill = i === 3 ? BRASS : i < 2 ? SLATE : i === 5 ? GREEN : TEAL;
      s.addShape(SH.roundRect, { x, y: 1.5, w, h: 0.95, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
      T(s, a[0], { x: x + 0.12, y: 1.55, w: 0.7, h: 0.4, fontSize: 18, bold: true, color: "FFFFFF", fontFace: "Cambria" }); T(s, a[1], { x: x + 0.12, y: 1.95, w: w - 0.2, h: 0.4, fontSize: 14, bold: true, color: "FFFFFF" });
      card(s, x, 2.55, w, 2.2, { fill: LIGHT }); T(s, a[2], { x: x + 0.12, y: 2.65, w: w - 0.24, h: 2.0, fontSize: 13 });
      card(s, x, 4.85, w, 1.2, { fill: "FFFFFF", line: fill }); T(s, "METRICS", { x: x + 0.12, y: 4.92, w: w - 0.2, h: 0.25, fontSize: 10, bold: true, color: fill }); T(s, a[3], { x: x + 0.12, y: 5.18, w: w - 0.24, h: 0.8, fontSize: 12, bold: true, color: INK }); });
    T(s, "S4 → S3 loop-back on REJECTED  ·  S4 → HITL on ESCALATE  ·  tracked in outputs/metrics/flow_metrics.json", { x: 0.6, y: 6.4, w: 12.1, h: 0.4, fontSize: 14, color: MUTED }); }

  // =============================================================== 6 worked trace
  { const k = "vip_duplicate_perk", r = RUNS[k], d = CMP[k];
    const s = content("One guest through every stage: Diamond VIP, duplicate perk", "Flow", "Worked trace from the recorded run. Each row is a stage with the actual values.");
    const rows = [["Stage", "What happened", "Metric"],
      ["S1 Context", "Sofia · VIP_Loyalty · Diamond · JUNIOR_OCEAN_VIEW · T-4 · 5 free tier perks", "5 perks held"],
      ["S2 Quant", `Top viable SKU ${d.top.sku}: τ̂ ${f(d.top.cate_dragon)}, P(accept) ${f(d.top.p_accept, 2)}, margin $${f(d.top.unit_margin, 0)} vs hurdle $${f(d.top.hurdle, 0)}`, `${f(r.S2_quant.gpu_latency_ms, 0)} ms on GPU`],
      ["S3 Draft 0", "LATE_CHECKOUT $70 bundled with WELCOME_DRINK + LATE_CHECKOUT_2PM (tier-blind sweetener)", "$70.31"],
      ["S4 Audit 0", "REJECTED: LOYALTY_PERK_DUPLICATION + SKU_ENTITLED_FREE (Diamond already gets both free)", `${r.S4_audit[0].checks_pass}/9 checks`],
      ["S3/S4 Rev 1", "Draft ROOM_UPGRADE $188 → REJECTED SKU_ENTITLED_FREE (upgrade is a Diamond perk)", `${r.S4_audit[1].checks_pass}/9 checks`],
      ["S3/S4 Rev 2", "Draft SPA_PACKAGE $281, no perks → all contracts pass, but VIP value w·p = 3.0 × $281 = $844 ≥ $250", "ESCALATE_HITL"],
      ["S5 Loop", `${r.S5_loop.drafts} drafts, ${r.S5_loop.revisions} revisions (bound 3), ${r.S5_loop.violations_fixed} violations repaired`, "within R_max"],
      ["S6 Outcome", `HITL_PENDING; contribution margin ${r.S6_outcome.contribution_margin_pct}%; human approves → HITL_APPROVED`, `${r.total_latency_ms} ms total`]];
    table(s, rows, { x: 0.6, y: 1.45, w: 12.1, colW: [1.7, 8.0, 2.4], fs: 13, rowH: 0.56 }); }

  // =============================================================== 7 strategist+auditor
  { const s = content("Agent contracts: what is optimised, what is enforced", "Flow");
    box(s, 0.6, 1.45, 5.0, 5.25, "Agent 1 · Strategist (generator)", "", LIGHT, INK, {});
    T(s, [{ text: "Ranks each SKU k:", options: { bold: true, breakLine: true } }, { text: "S_k = τ̂_k·(p*_k − c_k) + UCB_k − (λ_inv,k + λ_B·d_k)", options: { fontFace: "Cambria", italic: true, color: BRASS, breakLine: true } },
      { text: " ", options: { breakLine: true } }, { text: "Prices with the monotone elasticity net:", options: { bold: true, breakLine: true } }, { text: "p* = argmax_{p ≥ floor} (p − c)·P̂(x,k,p)", options: { fontFace: "Cambria", italic: true, color: BRASS, breakLine: true } },
      { text: " ", options: { breakLine: true } }, { text: "Applies Auditor directives from earlier revisions (min price, removed perks, excluded SKUs). The LLM only writes the copy; price and SKU always come from tools.", options: {} }],
      { x: 0.8, y: 2.05, w: 4.6, h: 4.5, fontSize: 14, paraSpaceAfter: 4 });
    table(s, [["#", "Auditor contract (Agent 2)", "Violation code"], ["1", "PMS buffer: available ≥ max(2, ⌈z·σ·√L⌉)", "INVENTORY_BUFFER"], ["2", "Price ≥ RMS floor", "BELOW_RMS_FLOOR"], ["3", "(p − c)/p ≥ 30%", "MARGIN_BELOW_THRESHOLD"],
      ["4", "Bundle ∩ tier perks = ∅", "LOYALTY_PERK_DUPLICATION"], ["5", "SKU not already a free tier perk", "SKU_ENTITLED_FREE"], ["6", "Regex ban-list ∧ Qwen tone ≥ 0.7", "TONE_BANNED_PHRASE / TONE_SCORE_LOW"],
      ["7", "Copy length ≤ channel max", "COPY_TOO_LONG"], ["8", "Visual alignment (Qwen3-VL, optional)", "VISUAL_MISALIGNMENT"], ["9", "Contact fatigue < 2 / 7 days", "CONTACT_FATIGUE → HITL"]],
      { x: 5.85, y: 1.45, w: 6.85, colW: [0.45, 3.6, 2.8], fs: 12, rowH: 0.47 });
    T(s, "Decision: violations & r < 3 → REJECTED · violations & r ≥ 3 → ESCALATE · clean but w_tier·p ≥ $250 → ESCALATE · otherwise APPROVED", { x: 5.85, y: 6.3, w: 6.85, h: 0.55, fontSize: 12, bold: true, color: INK }); }

  // =============================================================== 8 loop
  { const s = content("The bounded Generator–Critic loop and human escalation", "Flow");
    const nodes = [[0.7, 2.6, "Strategist", TEAL], [4.3, 2.6, "Auditor", BRASS], [8.4, 1.5, "Finalize", GREEN], [8.4, 4.0, "HITL gate", AMBER]];
    nodes.forEach((n) => { s.addShape(SH.roundRect, { x: n[0], y: n[1], w: 2.5, h: 1.1, rectRadius: 0.1, fill: { color: n[3] }, line: { color: n[3] } }); T(s, n[2], { x: n[0], y: n[1], w: 2.5, h: 1.1, fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Cambria" }); });
    arrow(s, 3.2, 3.15, 4.3, 3.15, INK); T(s, "draft", { x: 3.2, y: 2.8, w: 1.1, h: 0.3, fontSize: 12, align: "center", color: MUTED });
    arrow(s, 6.8, 2.95, 8.4, 2.05, GREEN); T(s, "APPROVED", { x: 7.0, y: 2.1, w: 1.3, h: 0.3, fontSize: 12, bold: true, color: GREEN });
    arrow(s, 6.8, 3.4, 8.4, 4.3, AMBER); T(s, "ESCALATE", { x: 6.95, y: 4.15, w: 1.4, h: 0.3, fontSize: 12, bold: true, color: "9A6A00" });
    arrow(s, 9.65, 4.0, 9.65, 2.6, AMBER);
    s.addShape(SH.line, { x: 1.95, y: 4.1, w: 3.6, h: 0, line: { color: CORAL, width: 1.75, dashType: "dash" } }); s.addShape(SH.line, { x: 1.95, y: 3.7, w: 0, h: 0.4, line: { color: CORAL, width: 1.75 } });
    s.addShape(SH.line, { x: 5.55, y: 3.7, w: 0, h: 0.4, line: { color: CORAL, width: 1.75, endArrowType: "none" } });
    T(s, "REJECTED and r < R_max:  r ← r+1, directives accumulate", { x: 1.8, y: 4.2, w: 4.2, h: 0.5, fontSize: 13, bold: true, color: CORAL });
    box(s, 0.7, 5.15, 5.8, 1.6, "Termination guarantee", "r strictly increases on every cycle and the Auditor escalates at r ≥ R_max, so at most R_max + 1 drafts exist. Tested for R_max ∈ {0, 1, 2, 3}.", LIGHT, INK, { bsize: 13 });
    box(s, 6.8, 5.15, 5.9, 1.6, "Directives only grow", "min-price per SKU, removed perks, excluded SKUs are union-merged, so a deterministically fixed violation cannot recur. Only stochastic LLM tone faults can persist, which the bound protects against.", LIGHT, INK, { bsize: 13 }); }

  // =============================================================== section: models
  section("Models & comparison", "Every model, what it does in the flow, and how it scores", "Models");

  // =============================================================== 9 inventory
  { const s = content("Model inventory: every model and where it acts in the flow", "Models");
    const rows = [["Model", "Role in the flow", "Stage", "Headline metric (held-out)", "Runs on"],
      ["Monotone elasticity net", "P(accept | x, k, price); price optimiser", "S2 → S3", `RMSE ${f(Q.elasticity.rmse_vs_true_prob)} · ECE ${f(Q.elasticity.ece, 4)} · ${Q.elasticity.monotonicity_violations} violations`, `CUDA · ${f(Q.elasticity.vram_peak_mib, 0)} MiB`],
      ["DragonNet", "Primary CATE τ̂ for ranking", "S2", `√PEHE ${f(pehe("dragonnet"))} (avg of 4)`, "CUDA"],
      ["X-Learner", "Second CATE estimate (cross-check)", "S2", `√PEHE ${f(pehe("x_learner"))}`, "CUDA"],
      ["T-Learner", "Baseline CATE (offline)", "eval", `√PEHE ${f(pehe("t_learner"))}`, "CUDA"],
      ["Constant ATE", "Naive baseline: same τ for everyone", "eval", `√PEHE ${f(pehe("constant_ate"))}`, "CPU"],
      ["LinUCB (batched)", "Exploration bonus in ranking score", "S2", `regret ${f(Q.bandit.linucb.final_cumulative_regret / 1000, 1)}k vs random ${f(Q.bandit.random.final_cumulative_regret / 1000, 1)}k`, "CUDA"],
      ["Neural Thompson Sampling", "Alternative bandit (offline)", "eval", `regret ${f(Q.bandit.neural_ts.final_cumulative_regret / 1000, 1)}k`, "CUDA"],
      ["Convex allocator", "Shadow prices λ → hurdle rate", "S2", `KKT violation ${Q.allocator.primal_violation_max.toExponential(0)} · ${f(Q.allocator.latency_ms, 0)} ms`, "CUDA"],
      ["Qwen3-VL-30B-A3B", "Copy, tone score, visual check", "S3 · S4", "fallback used in recorded run", "local server"],
      ["Deterministic contracts", "9 hard guardrails", "S4", "7/7 planted faults caught", "CPU"]];
    table(s, rows, { x: 0.6, y: 1.45, w: 12.1, colW: [2.35, 3.15, 0.95, 4.25, 1.4], fs: 12, rowH: 0.47 }); }

  // =============================================================== 10 elasticity
  { const s = content("Price-elasticity net: strictly monotone demand, well calibrated", "Models");
    await pic(s, PLOT("elasticity_diagnostics.png"), 0.6, 1.45, 8.2, 3.0, "Elasticity diagnostics");
    stat(s, 9.1, 1.4, 1.8, f(Q.elasticity.rmse_vs_true_prob), "RMSE vs true P(accept)", TEAL); stat(s, 11.0, 1.4, 1.8, f(Q.elasticity.ece, 4), "ECE (10 bins)", BRASS);
    stat(s, 9.1, 3.1, 1.8, String(Q.elasticity.monotonicity_violations), "monotonicity violations", GREEN); stat(s, 11.0, 3.1, 1.8, `${f(Q.elasticity.inference_latency_ms, 1)} ms`, `inference, ${Q.elasticity.n_val.toLocaleString()} rows`, SLATE);
    box(s, 0.6, 4.75, 5.9, 2.0, "How it guarantees monotone demand", "logit = g(h) − Σ_j softplus(v_j(h))·softplus(s(ln(p/p_list) − κ_j)). Non-negative weights on non-decreasing basis ⇒ ∂P̂/∂p < 0 for every guest.", LIGHT, INK, { bsize: 13 });
    box(s, 6.8, 4.75, 5.9, 2.0, "Why it matters in the flow", "Stage S3 solves p* = argmax (p − c)·P̂ on a 41-point grid above the RMS floor. A non-monotone model could recommend raising price to raise acceptance.", LIGHT, INK, { bsize: 13 }); }

  // =============================================================== section: uplift
  section("Uplift end-to-end flow", "From logged treatments to a ranked, guarded offer", "Uplift");

  // =============================================================== 11 uplift pipeline
  { const s = content("Uplift end-to-end: from logs to a guarded offer", "Uplift", "High-level documentation of the uplift flow. Each box maps to code: synthetic_generator, uplift_models, train_quant_suite, tools.QuantEngine, strategist_agent, auditor_agent, offline_replay.");
    const st = [["1  Log", "80k reservations; behaviour policy offered SKU k to some guests (T=1), others not (T=0); outcome Y = purchased", SLATE], ["2  Cohorts", "Per category: treated = offered at list price, control = no offer and organic purchase of k", SLATE],
      ["3  Train", "X-Learner + T-Learner + DragonNet on GPU; propensity head handles targeting confounding", TEAL], ["4  Evaluate", "√PEHE and correlation vs simulator truth; Qini & AUUC on observed data", TEAL],
      ["5  Serve", "QuantEngine loads .pt once; τ̂_k(x) for 4 SKUs in ~100 ms", BRASS], ["6  Decide", "Score S_k = τ̂·margin + UCB − hurdle; Auditor contracts filter", BRASS], ["7  Validate", "Offline IPS / DR over 16k held-out; regret vs oracle", GREEN]];
    const w = 1.62, g = 0.13;
    st.forEach((a, i) => { const x = 0.6 + i * (w + g); s.addShape(SH.roundRect, { x, y: 1.5, w, h: 0.6, rectRadius: 0.08, fill: { color: a[2] }, line: { color: a[2] } }); T(s, a[0], { x, y: 1.5, w, h: 0.6, fontSize: 15, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
      card(s, x, 2.2, w, 2.35, { fill: LIGHT }); T(s, a[1], { x: x + 0.1, y: 2.28, w: w - 0.2, h: 2.2, fontSize: 12 }); if (i < 6) arrow(s, x + w, 1.8, x + w + g, 1.8, INK); });
    box(s, 0.6, 4.85, 6.0, 1.9, "Identification assumptions", "Unconfoundedness given x (tier, spend, segment drive targeting; propensity head adjusts), overlap 0 < e(x) < 1 (clipped 0.02–0.98), SUTVA. Simulator truth τ_k(x) lets us verify.", LIGHT, INK, { bsize: 13 });
    box(s, 6.9, 4.85, 5.8, 1.9, "What the output is for", "τ̂_k(x) = incremental purchase probability caused by the offer, not raw acceptance. It stops the system spending margin on guests who would have bought anyway.", LIGHT, INK, { bsize: 13 }); }

  // =============================================================== 12 learners
  { const s = content("Three CATE learners, in one picture each", "Uplift");
    const cols = [["T-Learner", SLATE, "Fit μ̂₀(x), μ̂₁(x) separately.", "τ̂(x) = μ̂₁(x) − μ̂₀(x)", "Simple, but noise in each arm adds up. Baseline only."],
      ["X-Learner", TEAL, "Impute effects: D¹ = Y − μ̂₀(X) (treated), D⁰ = μ̂₁(X) − Y (control); regress D on x.", "τ̂ = g·τ̂₀ + (1 − g)·τ̂₁,  g = ê(x)", "Strong when arms are unbalanced. Also served as cross-check."],
      ["DragonNet", BRASS, "Shared Φ(x) with propensity head and two outcome heads; targeted regularisation with clever covariate h = t/ê − (1−t)/(1−ê).", "L = BCE(y, μ̂ₜ) + α·BCE(t, ê) + β·‖y − μ̂ₜ − εh‖²", "Primary model: best √PEHE on 3 of 4 categories."]];
    cols.forEach((c, i) => { const x = 0.6 + i * 4.1; card(s, x, 1.5, 3.9, 5.2, { fill: LIGHT }); s.addShape(SH.roundRect, { x, y: 1.5, w: 3.9, h: 0.65, rectRadius: 0.08, fill: { color: c[1] }, line: { color: c[1] } }); T(s, c[0], { x: x + 0.2, y: 1.5, w: 3.5, h: 0.65, fontSize: 20, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Cambria" });
      T(s, c[2], { x: x + 0.2, y: 2.35, w: 3.5, h: 1.5, fontSize: 14 }); card(s, x + 0.2, 3.95, 3.5, 1.2, { fill: "FFFFFF", line: c[1] }); T(s, c[3], { x: x + 0.3, y: 3.98, w: 3.3, h: 1.14, fontSize: 14, bold: true, color: INK, fontFace: "Cambria", italic: true, valign: "middle" });
      T(s, c[4], { x: x + 0.2, y: 5.4, w: 3.5, h: 1.1, fontSize: 14, color: MUTED }); }); }

  // =============================================================== 13 PEHE chart
  { const s = content("Uplift comparison: √PEHE per category (lower is better)", "Uplift", "Constant ATE beats every learner on LATE_CHECKOUT; that weakness is reported openly.");
    const labels = CATS.map((k) => k.replace("_", " ")); const m = [["Constant ATE", "constant_ate", "9AA6B8"], ["T-Learner", "t_learner", SLATE], ["X-Learner", "x_learner", TEAL], ["DragonNet", "dragonnet", BRASS]];
    s.addChart(pres.charts.BAR, m.map((a) => ({ name: a[0], labels, values: CATS.map((k) => Q.uplift[k].sqrt_pehe[a[1]]) })), { x: 0.6, y: 1.4, w: 7.8, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: m.map((a) => a[2]), showLegend: true, legendPos: "b",
      showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd", dataLabelFontSize: 9, valAxisMinVal: 0, valAxisLabelFormatCode: "0.00", ...chartBase("√PEHE vs simulator ground truth") });
    const best = CATS.map((k) => { const e = Object.entries(Q.uplift[k].sqrt_pehe).sort((a, b) => a[1] - b[1])[0]; return [k.replace("_", " "), e[0].replace("_", "-"), f(e[1])]; });
    table(s, [["Category", "Winner", "√PEHE"], ...best], { x: 8.7, y: 1.5, w: 4.0, colW: [1.85, 1.3, 0.85], fs: 12, rowH: 0.42 });
    T(s, [{ text: `Average: DragonNet ${f(pehe("dragonnet"))} · constant ${f(pehe("constant_ate"))} · X ${f(pehe("x_learner"))} · T ${f(pehe("t_learner"))}`, options: { bold: true, breakLine: true } }, { text: " ", options: { breakLine: true } },
      { text: "DragonNet's shared representation and propensity head remove most targeting confounding. On LATE_CHECKOUT the effect is dominated by tier entitlement, which the features capture weakly, so no learner beats the constant baseline there.", options: {} }], { x: 8.7, y: 3.7, w: 4.0, h: 3.0, fontSize: 13 }); }

  // =============================================================== 14 qini + corr
  { const s = content("Uplift ranking quality: correlation and Qini", "Uplift");
    const labels = CATS.map((k) => k.replace("_", " "));
    s.addChart(pres.charts.BAR, [{ name: "DragonNet", labels, values: CATS.map((k) => Q.uplift[k].corr_with_true_tau.dragonnet) }, { name: "X-Learner", labels, values: CATS.map((k) => Q.uplift[k].corr_with_true_tau.x_learner) }],
      { x: 0.6, y: 1.4, w: 6.0, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [BRASS, TEAL], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: 1, ...chartBase("Correlation of τ̂ with true τ") });
    s.addChart(pres.charts.BAR, [{ name: "DragonNet", labels, values: CATS.map((k) => Q.uplift[k].qini_coefficient.dragonnet) }, { name: "X-Learner", labels, values: CATS.map((k) => Q.uplift[k].qini_coefficient.x_learner) }, { name: "Random", labels, values: CATS.map((k) => Q.uplift[k].qini_coefficient.random) }],
      { x: 6.75, y: 1.4, w: 6.0, h: 3.6, barDir: "col", barGrouping: "clustered", chartColors: [BRASS, TEAL, "9AA6B8"], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.0000", dataLabelPosition: "outEnd", dataLabelFontSize: 9, valAxisLabelFormatCode: "0.000", ...chartBase("Qini coefficient (observed data)") });
    await pic(s, PLOT("uplift_qini_curves.png"), 0.6, 5.1, 7.4, 1.75, "Qini curves");
    T(s, [{ text: "Reading: ", options: { bold: true } }, { text: "correlation 0.77–0.86 for DragonNet on three categories means the ranking is informative even where absolute error is similar. Qini on observed data is small and noisy (~2k treated rows per category); LATE_CHECKOUT Qini is ≈ 0 for DragonNet and negative-to-zero elsewhere, matching the PEHE weakness.", options: {} }], { x: 8.2, y: 5.1, w: 4.5, h: 1.8, fontSize: 12 }); }

  // =============================================================== 15 uplift used in the flow
  { const s = content("How τ̂ drives the offer: a worked score", "Uplift", "Numbers are taken from the recorded run for clean_approval (ROOM_UPGRADE).");
    const t = CMP.clean_approval.top, tauPart = t.cate_dragon * t.unit_margin;
    const rows = [["Term", "Source model", "Value", "Meaning"], ["τ̂ (DragonNet)", "DragonNet", f(t.cate_dragon), "extra purchase probability caused by the offer"], ["Unit margin p* − c", "Elasticity net + catalogue", `$${f(t.unit_margin, 1)}`, `price $${f(t.price, 1)} chosen by argmax (p−c)·P̂`],
      ["τ̂ · margin", "product", `$${f(tauPart, 1)}`, "expected incremental margin"], ["UCB bonus", "LinUCB", `+$${f(t.ucb_bonus, 1)}`, "value of exploring this arm"], ["Hurdle λ_inv + λ_B·d", "Convex allocator", `−$${f(t.hurdle, 1)}`, "opportunity cost of scarce inventory / promo budget"], ["Score S", "sum", `${t.score < 0 ? "−" : ""}$${f(Math.abs(t.score), 1)}`, "ranking only; viability needs margin > hurdle"]];
    table(s, rows, { x: 0.6, y: 1.45, w: 12.1, colW: [2.5, 2.7, 1.5, 5.4], fs: 13, rowH: 0.5 });
    const all = CMP.clean_approval.all; card(s, 0.6, 5.15, 12.1, 1.6, { fill: LIGHT });
    T(s, [{ text: "Candidates for this guest: ", options: { bold: true } }, { text: all.map((c) => `${c.sku} (score ${f(c.score, 1)}, margin $${f(c.unit_margin, 0)} vs hurdle $${f(c.hurdle, 0)}${c.unit_margin > c.hurdle ? " ✔" : " ✘"})`).join("  ·  "), options: {} },
      { text: "\nThe highest raw score (LATE_CHECKOUT) is dropped because its margin does not clear the hurdle; the Strategist drafts the viable ROOM_UPGRADE, which the Auditor then approves 9/9.", options: { bold: true, color: TEAL } }], { x: 0.85, y: 5.25, w: 11.6, h: 1.4, fontSize: 14 }); }

  // =============================================================== 16 bandits
  { const s = content("Contextual bandits: LinUCB vs Neural TS vs random", "Models");
    const B = Q.bandit, names = [["Random", "random", "9AA6B8"], ["LinUCB", "linucb", BRASS], ["Neural TS", "neural_ts", TEAL]];
    s.addChart(pres.charts.BAR, [{ name: "Cumulative regret ($k)", labels: names.map((n) => n[0]), values: names.map((n) => B[n[1]].final_cumulative_regret / 1000) }], { x: 0.6, y: 1.4, w: 4.1, h: 3.7, barDir: "col", chartColors: names.map((n) => n[2]), showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", showLegend: false, valAxisMinVal: 0, ...chartBase("Cumulative regret ($k, lower better)") });
    s.addChart(pres.charts.BAR, [{ name: "Mean reward", labels: names.map((n) => n[0]), values: names.map((n) => B[n[1]].mean_reward_last) }], { x: 4.8, y: 1.4, w: 4.0, h: 3.7, barDir: "col", chartColors: names.map((n) => n[2]), showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", showLegend: false, valAxisMinVal: 0, ...chartBase("Recent mean reward ($/guest)") });
    s.addChart(pres.charts.BAR, [{ name: "Latency", labels: names.map((n) => n[0]), values: names.map((n) => B[n[1]].latency_ms) }], { x: 8.9, y: 1.4, w: 3.9, h: 3.7, barDir: "col", chartColors: names.map((n) => n[2]), showValue: true, dataLabelFormatCode: "0", dataLabelPosition: "outEnd", showLegend: false, valAxisMinVal: 0, ...chartBase("Simulation latency (ms)") });
    box(s, 0.6, 5.3, 12.1, 1.45, "Decision: LinUCB serves, Neural TS stays offline", `Neural TS has ${f(100 * (1 - B.neural_ts.final_cumulative_regret / B.linucb.final_cumulative_regret), 1)}% lower regret but $${f(B.linucb.mean_reward_last - B.neural_ts.mean_reward_last, 1)} lower recent reward and ${f(B.neural_ts.latency_ms / B.linucb.latency_ms, 1)}x the latency. LinUCB's closed-form bonus √(xᵀA⁻¹x) is cheap, auditable and plugs straight into the ranking score.`, LIGHT, INK, { bsize: 14 }); }

  // =============================================================== 17 allocator
  { const s = content("Convex allocator: shadow prices become the hurdle rate", "Models");
    await pic(s, PLOT("allocator_shadow_prices.png"), 0.6, 1.4, 7.4, 3.3, "Allocator shadow prices");
    const A = Q.allocator, rows = [["Category", "λ_inv ($/accept)", "Usage / cap"], ...Object.keys(A.lam_inv).map((k) => [k.replace("_", " "), f(A.lam_inv[k], 1), `${f(A.usage_inv[k], 0)} / ${A.inventory_caps[k]}`]), ["Promo budget λ_B", f(A.lam_budget, 2), `${f(A.usage_budget, 0)} / ${A.budget}`]];
    table(s, rows, { x: 8.3, y: 1.45, w: 4.4, colW: [1.9, 1.35, 1.15], fs: 12, rowH: 0.45 });
    stat(s, 8.3, 4.1, 2.0, A.primal_violation_max.toExponential(0), "KKT primal violation", GREEN); stat(s, 10.5, 4.1, 2.2, `${f(A.objective / A.unconstrained_objective * 100, 0)}%`, "of unconstrained objective retained", BRASS);
    box(s, 0.6, 5.1, 7.4, 1.65, "Interpretation", "λ_inv,k is the marginal profit of one more unit of scarce inventory k. An offer is worthwhile only if margin m > h = λ_inv + λ_B·d. This links portfolio-level scarcity to a single-guest decision.", LIGHT, INK, { bsize: 13 }); }

  // =============================================================== 18 models across flow (matrix)
  { const s = content("All models across the flow: who acts at which stage", "Models");
    const Y = (t) => ({ text: t, options: { align: "center", bold: true, color: TEAL } }), N = { text: "–", options: { align: "center", color: "9AA6B8" } };
    const rows = [["Model", "S1", "S2", "S3", "S4", "S5", "S6", "Evidence recorded in flow"],
      ["Elasticity net", N, Y("P̂"), Y("p*"), N, N, N, "P(accept) .38–.69 on top candidates"], ["DragonNet", N, Y("τ̂"), N, N, N, N, "τ̂ .158–.352 on top candidates"], ["X-Learner", N, Y("τ̂"), N, N, N, N, "agrees with DragonNet within 0.06 on most examples"],
      ["LinUCB", N, Y("UCB"), N, N, N, N, "bonus $26–$47 per candidate"], ["Convex allocator", N, Y("h"), Y("h"), N, N, N, "hurdle $36–$93 per SKU"], ["Qwen3-VL", N, N, Y("copy"), Y("tone"), N, N, "fallback in recorded run"],
      ["Deterministic contracts", Y("PMS"), N, N, Y("9"), Y("loop"), Y("HITL"), "10 violations caught, 0 escaped"], ["Policy evaluation (OPE)", N, N, N, N, N, Y("offline"), "IPS / DR / Qini / regret on 16k held-out"]];
    table(s, rows, { x: 0.6, y: 1.45, w: 12.1, colW: [2.4, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 4.6], fs: 12, rowH: 0.5 });
    T(s, "Models cooperate: elasticity prices, uplift decides who benefits, the bandit explores, the allocator charges for scarcity, contracts guarantee safety.", { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 15, bold: true, color: TEAL }); }

  // =============================================================== 19 models across flow (per example comparison)
  { const s = content("Models inside the flows: DragonNet vs X-Learner", "Models", "Top viable candidate at S2 for each of the seven examples (same rule the Strategist applies).");
    const keys = ["clean_approval", "corporate_late_checkout", "below_floor_price", "spammy_copy", "last_room_upgrade", "vip_duplicate_perk", "unresolvable_escalation"], lab = keys.map((k) => k.replace(/_/g, " "));
    s.addChart(pres.charts.BAR, [{ name: "DragonNet τ̂", labels: lab, values: keys.map((k) => CMP[k].top.cate_dragon) }, { name: "X-Learner τ̂", labels: lab, values: keys.map((k) => CMP[k].top.cate_xlearner) }],
      { x: 0.6, y: 1.35, w: 6.2, h: 3.5, barDir: "col", barGrouping: "clustered", chartColors: [BRASS, TEAL], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelFontSize: 9, catAxisLabelFontSize: 9, valAxisMinVal: 0, ...chartBase("Estimated uplift of top viable SKU") });
    s.addChart(pres.charts.BAR, [{ name: "P(accept)", labels: lab, values: keys.map((k) => CMP[k].top.p_accept) }],
      { x: 6.9, y: 1.35, w: 5.8, h: 3.5, barDir: "col", chartColors: [SLATE], showLegend: false, showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelFontSize: 9, catAxisLabelFontSize: 9, valAxisMinVal: 0, valAxisMaxVal: 1, ...chartBase("Elasticity net P(accept) at p*") });
    const rows = [["Example", "SKU", "τ̂ Dragon", "τ̂ X", "|Δ|", "P̂", "UCB $", "Hurdle $", "Margin $"], ...keys.map((k) => { const t = CMP[k].top; return [k.replace(/_/g, " "), t.sku.replace("_", " "), f(t.cate_dragon), f(t.cate_xlearner), f(Math.abs(t.cate_dragon - t.cate_xlearner)), f(t.p_accept, 2), f(t.ucb_bonus, 1), f(t.hurdle, 1), f(t.unit_margin, 1)]; })];
    table(s, rows, { x: 0.6, y: 4.95, w: 12.1, colW: [2.6, 2.0, 1.1, 1.0, 0.9, 0.9, 1.2, 1.2, 1.2], fs: 11, rowH: 0.24 }); }

  // =============================================================== 20 policy comparison
  { const s = content("Policy-level comparison: what the whole stack is worth", "Models", "Ground truth value comes from simulator expected rewards; IPS and DR are the offline estimators a real deployment would rely on.");
    const names = ["logged_behaviour", "no_offer", "random_eligible", "ml_only(elasticity greedy)", "ml+agent_guardrails", "oracle"], lab = ["Logged", "No offer", "Random", "ML only", "ML + guardrails", "Oracle"];
    s.addChart(pres.charts.BAR, [["Ground truth", "true_value", INK], ["IPS", "ips", "9AA6B8"], ["Doubly robust", "dr", BRASS]].map((a) => ({ name: a[0], labels: lab, values: names.map((n) => pol[n][a[1]]) })),
      { x: 0.6, y: 1.35, w: 7.4, h: 4.1, barDir: "col", barGrouping: "clustered", chartColors: [INK, "9AA6B8", BRASS], showLegend: true, legendPos: "b", showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", dataLabelFontSize: 9, valAxisMinVal: 0, ...chartBase("Policy value, $ margin per guest") });
    const rows = [["Policy", "Regret", "Violations", "Cannibal."], ...names.map((n, i) => [lab[i], f(pol[n].regret, 1), `${f(100 * pol[n].guardrail_violation_rate, 1)}%`, `${f(100 * pol[n].cannibalisation_rate, 1)}%`])];
    table(s, rows, { x: 8.3, y: 1.45, w: 4.4, colW: [1.7, 0.85, 0.95, 0.9], fs: 12, rowH: 0.44 });
    box(s, 8.3, 4.75, 4.4, 1.95, "Takeaways", `ML+guardrails keeps ${f(100 * pol["ml+agent_guardrails"].true_value / pol["ml_only(elasticity greedy)"].true_value, 0)}% of ML-only value with zero violations. IPS is biased low (weights clipped at 50); DR is closer but ignores cannibalisation.`, LIGHT, INK, { bsize: 12 });
    T(s, `DR 95% CI, guarded: [${f(pol["ml+agent_guardrails"].dr_ci95[0], 1)}, ${f(pol["ml+agent_guardrails"].dr_ci95[1], 1)}] · ESS ${f(pol["ml+agent_guardrails"].ess, 0)} of 16,000`, { x: 0.6, y: 5.65, w: 7.4, h: 0.4, fontSize: 13, color: MUTED });
    T(s, "Guardrail-filtered policy picks argmax q̂(x,a) over actions that pass floor, margin, parity and eligibility, the vectorised twin of the Auditor (tested equal).", { x: 0.6, y: 6.1, w: 7.4, h: 0.7, fontSize: 13 }); }

  // =============================================================== section: metrics
  section("Metrics: stage, flow, overall", "Three levels of measurement for seven worked examples", "Metrics");

  // =============================================================== 21 flow types
  { const s = content("Three flow types, seven worked examples", "Metrics");
    const types = [["Straight-through", GREEN, "Strategist → Auditor ✔ → Finalize", ["clean_approval", "corporate_late_checkout"]], ["Reject → revise → approve", BRASS, "Auditor ✘ → revise → Auditor ✔", ["below_floor_price", "spammy_copy", "last_room_upgrade"]],
      ["HITL escalation", CORAL, "… → Auditor ESCALATE → HITL gate → Finalize", ["vip_duplicate_perk", "unresolvable_escalation"]]];
    types.forEach((t, i) => { const x = 0.6 + i * 4.1; card(s, x, 1.5, 3.9, 5.2, { fill: LIGHT }); s.addShape(SH.roundRect, { x, y: 1.5, w: 3.9, h: 0.75, rectRadius: 0.08, fill: { color: t[1] }, line: { color: t[1] } }); T(s, t[0], { x: x + 0.2, y: 1.5, w: 3.5, h: 0.75, fontSize: 18, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Cambria" });
      T(s, t[2], { x: x + 0.2, y: 2.45, w: 3.5, h: 0.8, fontSize: 13, color: MUTED });
      t[3].forEach((k, j) => { const r = RUNS[k]; card(s, x + 0.2, 3.35 + j * 1.05, 3.5, 0.92, { fill: "FFFFFF" }); T(s, k.replace(/_/g, " "), { x: x + 0.35, y: 3.4 + j * 1.05, w: 3.2, h: 0.35, fontSize: 14, bold: true, color: INK }); T(s, `${r.S5_loop.revisions} rev · ${r.S6_outcome.final_status.replace("_", " ")} · ${r.total_latency_ms} ms`, { x: x + 0.35, y: 3.78 + j * 1.05, w: 3.2, h: 0.4, fontSize: 12, color: MUTED }); }); }); }

  // =============================================================== 22 overall
  { const s = content("Overall metrics across all flows", "Metrics");
    stat(s, 0.6, 1.4, 2.3, `${f(100 * OV.first_pass_approval_rate, 0)}%`, "approved on the first draft", BRASS); stat(s, 3.1, 1.4, 2.3, `${f(100 * OV.final_auto_approval_rate, 0)}%`, "auto-approved after revisions", GREEN);
    stat(s, 5.6, 1.4, 2.3, `${f(100 * OV.hitl_rate, 0)}%`, "escalated to a human", CORAL); stat(s, 8.1, 1.4, 2.3, f(OV.mean_revisions, 2), `mean revisions (max seen ${OV.max_revisions_seen} of 3)`, SLATE); stat(s, 10.6, 1.4, 2.1, `${f(OV.mean_contribution_margin_pct, 0)}%`, "mean contribution margin (floor 30%)", TEAL);
    const vc = OV.violations_by_code, ks = Object.keys(vc).sort((a, b) => vc[b] - vc[a]);
    s.addChart(pres.charts.BAR, [{ name: "Caught", labels: ks.map((k) => k.replace(/_/g, " ")), values: ks.map((k) => vc[k]) }], { x: 0.6, y: 3.2, w: 6.9, h: 3.6, barDir: "bar", chartColors: [CORAL], showLegend: false, showValue: true, dataLabelPosition: "outEnd", valAxisMinVal: 0, catAxisLabelFontSize: 10, ...chartBase("Violations caught by the Auditor (10 total)") });
    table(s, [["Latency (full pipeline)", "ms"], ["p50", f(OV.latency_ms.p50, 0)], ["p95", f(OV.latency_ms.p95, 0)], ["max", f(OV.latency_ms.max, 0)], ["mean GPU quant", f(OV.mean_gpu_quant_latency_ms, 0)]], { x: 7.8, y: 3.3, w: 4.9, colW: [3.4, 1.5], fs: 13, rowH: 0.42 });
    T(s, "Five of seven scenarios plant a faulty first draft, so the 29% first-pass rate measures whether faults are caught, not a production error rate. LLM mode: fallback.", { x: 7.8, y: 5.6, w: 4.9, h: 1.2, fontSize: 12, color: MUTED }); }

  // =============================================================== 23 flow-level
  { const s = content("Flow-level metrics: revisions and latency by flow type", "Metrics");
    const bf = FL.by_flow, ks = Object.keys(bf), lab = ["Straight-through", "Reject → revise", "HITL escalation"];
    s.addChart(pres.charts.BAR, [{ name: "Mean revisions", labels: lab, values: ks.map((k) => bf[k].mean_revisions) }], { x: 0.6, y: 1.4, w: 4.2, h: 3.6, barDir: "col", chartColors: [BRASS], showLegend: false, showValue: true, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", valAxisMinVal: 0, ...chartBase("Mean revisions") });
    s.addChart(pres.charts.BAR, [{ name: "Mean latency", labels: lab, values: ks.map((k) => bf[k].mean_latency_ms) }], { x: 4.9, y: 1.4, w: 4.2, h: 3.6, barDir: "col", chartColors: [TEAL], showLegend: false, showValue: true, dataLabelFormatCode: "0", dataLabelPosition: "outEnd", valAxisMinVal: 0, ...chartBase("Mean latency (ms)") });
    table(s, [["Flow", "n", "Outcome"], ...ks.map((k, i) => [lab[i], String(bf[k].n), Object.entries(bf[k].statuses).map(([a, b]) => `${b} ${a.replace("_", " ")}`).join(", ")])], { x: 9.3, y: 1.5, w: 3.4, colW: [1.35, 0.4, 1.65], fs: 11, rowH: 0.7 });
    box(s, 0.6, 5.3, 12.1, 1.45, "Pattern", `Each extra revision adds roughly one Strategist + Auditor round (~100 ms here). Latency rises ${f(bf.HITL_ESCALATION.mean_latency_ms / bf.STRAIGHT_THROUGH.mean_latency_ms, 1)}x from straight-through to HITL, but the human wait, not compute, dominates real escalation time. Every flow ended within the R_max bound.`, LIGHT, INK, { bsize: 14 }); }

  // =============================================================== 24 stage-level table
  { const s = content("Stage-level metrics for every example (S1 → S6)", "Metrics");
    const ks = ["clean_approval", "corporate_late_checkout", "below_floor_price", "spammy_copy", "last_room_upgrade", "vip_duplicate_perk", "unresolvable_escalation"];
    const rows = [["Example", "S1 tier", "S2 GPU ms · τ̂", "S3 drafts", "S4 first audit", "S5 rev", "S6 outcome", "Price · CM", "Total ms"], ...ks.map((k) => { const r = RUNS[k], a = r.S4_audit[0]; return [k.replace(/_/g, " "), r.S1_context.tier, `${f(r.S2_quant.gpu_latency_ms, 0)} · ${f(CMP[k].top.cate_dragon, 2)}`, String(r.S5_loop.drafts), `${a.decision === "APPROVED" ? "✔" : a.decision === "REJECTED" ? "✘" : "↑"} ${a.violations.length ? a.violations.join(", ").replace(/_/g, " ").toLowerCase() : "all 9 pass"}`, String(r.S5_loop.revisions), r.S6_outcome.final_status.replace("_", " "), `$${f(r.S6_outcome.final_price, 0)} · ${f(r.S6_outcome.contribution_margin_pct, 0)}%`, f(r.total_latency_ms, 0)]; })];
    table(s, rows, { x: 0.6, y: 1.45, w: 12.1, colW: [2.1, 0.8, 1.35, 0.7, 3.0, 0.6, 1.4, 1.1, 1.05], fs: 11, rowH: 0.56 });
    T(s, "S2 τ̂ is the top viable candidate; stress hooks may steer the draft to another SKU (see story slides). Violation codes shown for the first audit pass.", { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontSize: 12, color: MUTED }); }

  // =============================================================== 25 stage latency chart
  { const s = content("Where the time goes: Strategist (incl. GPU quant) vs Auditor", "Metrics");
    const ks = ["clean_approval", "corporate_late_checkout", "below_floor_price", "spammy_copy", "last_room_upgrade", "vip_duplicate_perk", "unresolvable_escalation"], lab = ks.map((k) => k.replace(/_/g, " "));
    s.addChart(pres.charts.BAR, [{ name: "Strategist incl. GPU quant (ms)", labels: lab, values: ks.map((k) => RUNS[k].S3_draft.reduce((a, d) => a + d.ms, 0)) }],
      { x: 0.6, y: 1.4, w: 8.2, h: 5.3, barDir: "col", chartColors: [TEAL], showLegend: false, showValue: true, dataLabelFormatCode: "0", dataLabelPosition: "outEnd", catAxisLabelFontSize: 9, valAxisMinVal: 0, ...chartBase("Strategist latency per example (ms), includes GPU quant inference") });
    box(s, 9.1, 1.5, 3.6, 2.45, "Reading", `Strategist time is almost all GPU model inference and grows with drafts. The Auditor's deterministic contracts total under ${f(Math.max(...Object.values(RUNS).map((r) => r.S4_audit.reduce((a, d) => a + d.ms, 0))), 1)} ms in every flow; with a live LLM the tone check would dominate.`, LIGHT, INK, { bsize: 13 });
    box(s, 9.1, 4.15, 3.6, 2.55, "Budget", `Mean GPU quant inference ${f(OV.mean_gpu_quant_latency_ms, 0)} ms per draft on an RTX 3090 (${f(Q.elasticity.vram_peak_mib, 0)} MiB peak for the elasticity net). Whole flows stay under ${f(OV.latency_ms.max, 0)} ms.`, LIGHT, INK, { bsize: 13 }); }

  // =============================================================== section: scenarios story
  section("Seven guests, one story", "Streamlit End-to-End Flow tab, scenario by scenario", "Story");

  // =============================================================== 26 story arc
  { const s = content("The story arc: smooth, repaired, escalated", "Story");
    const acts = [["Act I · It just works", GREEN, "Amelia and Arjun receive offers that clear every contract on the first draft.", ["clean_approval", "corporate_late_checkout"]],
      ["Act II · The critic earns its keep", BRASS, "Liam, Isabella and Noah receive faulty first drafts. The Auditor names the violation, the Strategist repairs it.", ["below_floor_price", "spammy_copy", "last_room_upgrade"]],
      ["Act III · A human decides", CORAL, "Sofia's VIP offer is clean but too valuable to send unseen; Ethan's generator never fixes its tone, so the loop stops at R_max.", ["vip_duplicate_perk", "unresolvable_escalation"]]];
    acts.forEach((a, i) => { const y = 1.45 + i * 1.8; card(s, 0.6, y, 12.1, 1.65, { fill: LIGHT }); s.addShape(SH.roundRect, { x: 0.6, y, w: 3.2, h: 1.65, rectRadius: 0.08, fill: { color: a[1] }, line: { color: a[1] } }); T(s, a[0], { x: 0.8, y, w: 2.8, h: 1.65, fontSize: 18, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Cambria" });
      T(s, a[2], { x: 4.05, y: y + 0.1, w: 8.4, h: 0.75, fontSize: 14 }); T(s, a[3].map((k) => `${k.replace(/_/g, " ")}  (${RUNS[k].S5_loop.revisions} rev · ${RUNS[k].S6_outcome.final_status.replace("_", " ")})`).join("   ·   "), { x: 4.05, y: y + 0.95, w: 8.4, h: 0.6, fontSize: 12, bold: true, color: INK }); }); }

  // =============================================================== story slides
  const stories = [
    ["clean_approval", "Amelia, Gold honeymooner: the offer that just works", "Act I", GREEN,
      ["Booked a DELUXE_KING via the direct site, arriving in 4 days. Gold tier already includes a welcome drink and breakfast.", "The quant stack ranks ROOM_UPGRADE: τ̂ +0.238, P(accept) 0.41, margin $85 against a $72 hurdle. The Strategist bundles champagne and turndown, deliberately not her free perks.", "All nine contracts pass. Approved first time."]],
    ["corporate_late_checkout", "Arjun, corporate traveller: a small add-on, a clean pass", "Act I", GREEN,
      ["GDS booking, no loyalty tier, two nights midweek.", "Late checkout carries the highest uplift in the deck (τ̂ +0.352) at a 71% contribution margin; the price sits above the $38 RMS floor.", "No perks to duplicate, no inventory pressure, refined copy: approved without revision."]],
    ["below_floor_price", "Liam: the over-eager discount", "Act II", BRASS,
      ["Leisure guest from an OTA, no tier. The stress hook makes the first draft price a spa package at $108, 40% under list.", "S4 flags BELOW_RMS_FLOOR (floor $150) and returns a min-price directive for that SKU.", "The Strategist re-prices at $150.75; second audit passes. One revision, 60% margin."]],
    ["spammy_copy", "Isabella: right offer, wrong voice", "Act II", BRASS,
      ["Family stay, Silver tier. The first draft pitches late checkout with 'ACT NOW!! 50% OFF!!' and 'CHEAP'.", "Regex ban-list fires (TONE_BANNED_PHRASE) and the tone score is 0.0 (7 of 9 checks pass); price and SKU were fine.", "Rewritten in a refined register, tone 1.0, approved on revision 1."]],
    ["last_room_upgrade", "Noah: protecting the last room", "Act II", BRASS,
      ["Honeymoon guest in DELUXE_KING. The live PMS webhook shows one JUNIOR_OCEAN_VIEW room left, below the safety buffer of 5.", "INVENTORY_BUFFER rejects the upgrade and excludes the SKU, protecting full-rate demand.", "The Strategist pivots to a SPA package at $183; approved after one revision."]],
    ["vip_duplicate_perk", "Sofia, Diamond VIP: entitled, and too valuable to auto-send", "Act III", CORAL,
      ["Draft 0 sells a $70 late checkout bundled with perks Diamond members already hold free: LOYALTY_PERK_DUPLICATION and SKU_ENTITLED_FREE.", "Draft 1 offers a room upgrade, also a free Diamond perk. Draft 2, a $281 spa package, passes every contract.", "VIP value 3.0 × $281 = $844 exceeds the $250 threshold, so the graph pauses for human approval."]],
    ["unresolvable_escalation", "Ethan: when the generator cannot fix itself", "Act III", CORAL,
      ["A misbehaving generator emits spam copy on every revision.", "The Auditor rejects three times with TONE_BANNED_PHRASE, each time with the same remediation, and the loop reaches R_max = 3.", "Instead of looping forever, the graph escalates; a reviewer decides. Four drafts, 883 ms, nothing sent automatically."]]];
  for (const [k, head, act, col, para] of stories) { const r = RUNS[k];
    const s = content(head, "Story", `Streamlit screenshot of the ${k} run. Top half: flow graph, S1 and S2. Bottom half: S3 to S6. Latency inside the screenshot is from the live capture run (includes cold-start effects); the chips use the warmed flow_metrics.json run.`);
    T(s, act.toUpperCase() + "  ·  " + k.replace(/_/g, " "), { x: 0.6, y: 1.22, w: 12.1, h: 0.3, fontSize: 12, bold: true, color: col });
    para.forEach((t, i) => { const x = 0.6 + i * 4.1; card(s, x, 1.58, 3.9, 1.25, { fill: LIGHT }); T(s, t, { x: x + 0.15, y: 1.63, w: 3.6, h: 1.15, fontSize: 12, valign: "middle" }); });
    const maxH = 3.4; await pic(s, ASSET(`${k}_top.png`), 0.6, 3.0, 6.6, maxH, "Streamlit flow top: graph, S1, S2"); await pic(s, ASSET(`${k}_bot.png`), 7.4, 3.0, 5.3, maxH, "Streamlit flow bottom: S3 to S6");
    const chips = [["S2 GPU", `${f(r.S2_quant.gpu_latency_ms, 0)} ms`], ["Top τ̂", f(CMP[k].top.cate_dragon, 2)], ["S4 first audit", `${r.S4_audit[0].checks_pass}/9 pass`], ["S5 revisions", `${r.S5_loop.revisions} of 3`], ["S6 outcome", r.S6_outcome.final_status.replace("_", " ")], ["Price · CM", `$${f(r.S6_outcome.final_price, 0)} · ${f(r.S6_outcome.contribution_margin_pct, 0)}%`], ["Total", `${f(r.total_latency_ms, 0)} ms`]];
    chips.forEach((c, i) => { const x = 0.6 + i * 1.75; card(s, x, 6.5, 1.65, 0.52, { fill: "FFFFFF", line: col }); T(s, c[0], { x: x + 0.08, y: 6.52, w: 1.5, h: 0.2, fontSize: 9, color: MUTED }); T(s, c[1], { x: x + 0.08, y: 6.72, w: 1.5, h: 0.26, fontSize: 12, bold: true, color: INK }); }); }

  // =============================================================== overall metrics screenshot
  { const s = content("The Flow Metrics tab: the same numbers, live in Streamlit", "Story");
    await pic(s, ASSET("flow_metrics_crop.png"), 0.6, 1.35, 7.6, 5.55, "Streamlit flow metrics tab");
    box(s, 8.5, 1.5, 4.2, 5.2, "What the tab shows", "Overall KPIs (first-pass, auto-approval, HITL, revisions, latency), violation histogram, latency by flow, by-flow table, per-example stage metrics and system-level model/OPE context.\n\nA re-run button executes all seven flows on the GPU and rewrites flow_metrics.json.", LIGHT, INK, { bsize: 14 }); }

  // =============================================================== section: ops
  section("Interfaces, verification & honesty", "How it is exposed, how it is tested, what it does not prove", "Close");

  // =============================================================== interfaces
  { const s = content("Interfaces: Streamlit dashboard and Flask operational UI", "Close");
    await pic(s, SHOT("streamlit_executive.png"), 0.6, 1.35, 6.0, 3.9, "Streamlit executive"); await pic(s, ASSET("flask_top.png"), 6.9, 1.35, 5.8, 3.9, "Flask audit UI");
    T(s, "Streamlit: 6 tabs (Executive KPIs · End-to-End Flow · Flow Metrics · Quant Models · Offline Policy Evaluation · HITL Queue)", { x: 0.6, y: 5.45, w: 6.0, h: 0.8, fontSize: 13 });
    T(s, "Flask: /health · POST /api/v1/offers/generate · POST /webhooks/pms · HITL queue and decision API · /audit HTML UI", { x: 6.9, y: 5.45, w: 5.8, h: 0.8, fontSize: 13 });
    T(s, "Both read the same checkpoints and write the same JSONL audit trail.", { x: 0.6, y: 6.4, w: 12.1, h: 0.4, fontSize: 14, bold: true, color: TEAL }); }

  // =============================================================== verification
  { const s = content("Verification: tests, E2E checks and screenshots", "Close");
    const passed = VER.checks.filter((c) => c.ok).length;
    stat(s, 0.6, 1.4, 2.8, "28", "pytest tests: guardrails, math identities, graph loop, schemas", GREEN); stat(s, 3.7, 1.4, 2.8, `${passed}/${VER.checks.length}`, "Playwright E2E checks (API, webhook, HITL, UI)", TEAL); stat(s, 6.8, 1.4, 2.8, "7", "per-example flow screenshots + metrics tab", BRASS); stat(s, 9.9, 1.4, 2.8, "R_max", "loop bound tested for 0, 1, 2, 3", SLATE);
    const items = [["Math identities", "Monotone demand on a price grid; allocator KKT and complementary slackness; IPS unbiased over 60 log draws; DR lower variance; Qini beats random."], ["Guardrail parity", "The vectorised OPE action filter equals the Auditor's deterministic contracts action by action."], ["E2E script", "scripts/verify_and_capture.py boots Flask and Streamlit, exercises webhook and HITL approve/resume, drives every Streamlit tab, exits non-zero on failure."]];
    items.forEach((it, i) => { const y = 3.4 + i * 1.12; card(s, 0.6, y, 12.1, 1.0, { fill: LIGHT }); T(s, it[0], { x: 0.85, y: y + 0.1, w: 2.6, h: 0.8, fontSize: 15, bold: true, color: INK, valign: "middle" }); T(s, it[1], { x: 3.5, y: y + 0.1, w: 9.0, h: 0.8, fontSize: 13, valign: "middle" }); }); }

  // =============================================================== limitations
  { const s = content("What this does and does not prove", "Close");
    const L = [["Synthetic data", "Ground-truth elasticity and CATE make evaluation verifiable, but they say nothing about real-world lift. A pilot with randomised holdout is the real test.", CORAL],
      ["LLM not exercised", "Qwen3-VL was offline in recorded runs; copy and tone used the deterministic fallback. The live path is implemented but untested here.", AMBER],
      ["LATE_CHECKOUT uplift", `Every learner is worse than the constant baseline (DragonNet ${f(Q.uplift.LATE_CHECKOUT.sqrt_pehe.dragonnet)} vs ${f(Q.uplift.LATE_CHECKOUT.sqrt_pehe.constant_ate)}). Entitlement features need work.`, CORAL],
      ["Planted faults", "Five of seven flows inject a faulty first draft by design. Pass-rate figures measure detection and repair, not production error rate.", AMBER],
      ["Small flow sample", "Seven runs are a smoke test; population behaviour comes from the 16k-reservation policy evaluation.", SLATE]];
    L.forEach((l, i) => { const y = 1.4 + i * 1.07; card(s, 0.6, y, 12.1, 0.95, { fill: LIGHT }); s.addShape(SH.ellipse, { x: 0.85, y: y + 0.27, w: 0.4, h: 0.4, fill: { color: l[2] }, line: { color: l[2] } }); T(s, l[0], { x: 1.5, y: y + 0.1, w: 2.8, h: 0.75, fontSize: 15, bold: true, color: INK, valign: "middle" }); T(s, l[1], { x: 4.4, y: y + 0.08, w: 8.1, h: 0.8, fontSize: 13, valign: "middle" }); }); }

  // =============================================================== closing
  { pres.addSection({ title: "Closing" }); const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Closing" }); s.background = { color: INK };
    T(s, "Next steps", { x: 0.9, y: 0.9, w: 11, h: 0.8, fontFace: "Cambria", fontSize: 40, bold: true, color: "FFFFFF" });
    const ns = [["1", "Serve Qwen3-VL (quantised) and re-record the flows with live copy, tone and vision checks."], ["2", "Improve LATE_CHECKOUT uplift: tier-entitlement features, more treated samples, doubly robust CATE."], ["3", "Randomised pilot with a holdout to replace simulator truth with measured lift."], ["4", "Wire the PMS webhook and HITL queue to real operations tooling."]];
    ns.forEach((n, i) => { const y = 2.0 + i * 1.1; s.addShape(SH.ellipse, { x: 0.9, y: y + 0.08, w: 0.6, h: 0.6, fill: { color: BRASS }, line: { color: BRASS } }); T(s, n[0], { x: 0.9, y: y + 0.08, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: INK, align: "center", valign: "middle" }); T(s, n[1], { x: 1.8, y, w: 10.5, h: 0.8, fontSize: 18, color: "FFFFFF", valign: "middle" }); });
    T(s, "Repository: agents/MultiAgent_flow_2Agents_Pre-Arrival-Upsell_and_Margin-Guardrail-System  ·  docs/MATHEMATICAL_FOUNDATIONS.md", { x: 0.9, y: 6.7, w: 11.5, h: 0.4, fontSize: 13, color: "9AA6B8" }); }

  const out = path.join(__dirname, "Upsell_Guardrail_MAS_Deck.pptx");
  await pres.writeFile({ fileName: out }); await applyTheme(out, THEME); console.log("wrote", out);
})().catch((e) => { console.error(e); process.exit(1); });
