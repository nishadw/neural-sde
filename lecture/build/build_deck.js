// Neural SDEs lecture (CMSC 838B) — v3: light theme, running example from Xavier et al. (Foods 2024).
// Build:  python figs.py && node build_deck.js   (optional: APPLY_THEME=/path/to/apply_theme.js)
// Paper figures (img/p_*.jpg) are from Xavier, Rodrigues & Silva, Foods 13(8):1150 (2024), CC BY 3.0.
const pptxgen = require("pptxgenjs");
let applyTheme = async () => {};
try { ({ applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js")); } catch (e) { console.warn("apply_theme.js not found; skipping theme colours"); }
const fs = require("fs");
const sizeOf = (f) => {
  const b = fs.readFileSync(f);
  if (b[0] === 0x89) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; // PNG
  let i = 2; while (i < b.length) { const m = b[i + 1], len = b.readUInt16BE(i + 2); if (m >= 0xc0 && m <= 0xc3) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7) }; i += 2 + len; }
  throw new Error("size " + f);
};

const THEME = {
  name: "Neural SDE", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "1F2328", lt1: "FFFFFF", dk2: "6B7280", lt2: "F5F6F7", accent1: "C8102E", accent2: "1F2328", accent3: "9AA3AF", accent4: "E5E7EB", accent5: "F5F6F7", accent6: "374151", hlink: "C8102E", folHlink: "6B7280" },
};
const INK = "1F2328", RED = "C8102E", MUTED = "6B7280", CARD = "F5F6F7", GRAY = "9AA3AF", LINE = "E5E7EB";
const CREDIT = "Xavier, Rodrigues & Silva, Foods 13:1150 (2024)";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Nishad Wajge";
pres.title = "Neural SDEs: Differentiating Through Stochastic Dynamics";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.5, y: 0.3, w: 9, h: 0.28, fontSize: 9.5, bold: true, color: C.text2, charSpacing: 1.5, margin: 0, valign: "top" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.54, w: 9, h: 0.6, fontSize: 25, bold: true, color: C.text1, margin: 0, valign: "top", align: "left", fontFace: "Arial" }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 5.28, w: 0.4, h: 0.22, fontSize: 9, color: MUTED, align: "right" },
});

// ---------------- helpers ----------------
function content(section, kicker, title, takeaway) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  if (takeaway) {
    s.addShape(pres.shapes.LINE, { x: 0.5, y: 4.78, w: 9, h: 0, line: { color: LINE, width: 1 } });
    s.addText(takeaway, { x: 0.5, y: 4.84, w: 8.6, h: 0.36, fontSize: 12.5, bold: true, color: INK, margin: 0, valign: "middle", isTextBox: true, objectName: "takeaway" });
  }
  return s;
}
function img(s, file, x, y, o = {}) {
  const { w: pw, h: ph } = sizeOf("img/" + file);
  let w, h;
  if (o.w && !o.h) { w = o.w; h = w * ph / pw; }
  else if (o.h && !o.w) { h = o.h; w = h * pw / ph; }
  else { const sc = Math.min(o.w / pw, o.h / ph); w = pw * sc; h = ph * sc; }
  if (o.center) x = x + (o.w - w) / 2;
  if (o.vcenter) y = y + (o.h - h) / 2;
  s.addImage({ path: "img/" + file, x, y, w, h, objectName: file });
  return { x, y, w, h };
}
const asset = (s, file, x, y, o) => {
  const p = "../../assets/" + file; const { w: pw, h: ph } = sizeOf(p);
  const sc = Math.min(o.w / pw, o.h / ph); const w = pw * sc, h = ph * sc;
  s.addImage({ path: p, x: x + (o.w - w) / 2, y, w, h, objectName: file });
};
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: o.fill || CARD }, line: { color: o.line || o.fill || CARD, width: o.lineW || 1 }, rectRadius: 0.05 });
}
function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontSize: 12.5, color: INK, margin: 0, valign: "top", isTextBox: true }, o));
}
const label = (s, t, x, y, w, color = MUTED) => txt(s, t, x, y, w, 0.24, { fontSize: 10.5, bold: true, color });
const credit = (s, t, x, y, w) => txt(s, t, x, y, w, 0.22, { fontSize: 8, color: MUTED, italic: true });
function arrow(s, x1, y1, x2, y2, color = INK, w = 1.5) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color, width: w, endArrowType: "triangle" } });
}
// table: rows of strings or {text, options}; first row is header
function table(s, rows, x, y, colW, o = {}) {
  const fs = o.fontSize || 11.5;
  const body = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "string" ? { text: c, options: {} } : c;
    const base = { fontSize: fs, color: INK, valign: "middle", align: j === 0 ? "left" : (o.align || "center") };
    if (i === 0) Object.assign(base, { bold: true, color: MUTED, fontSize: fs - 1, border: [{ type: "none" }, { type: "none" }, { type: "solid", pt: 1, color: INK }, { type: "none" }] });
    else Object.assign(base, { border: [{ type: "none" }, { type: "none" }, { type: "solid", pt: 0.5, color: LINE }, { type: "none" }] });
    return { text: cell.text, options: Object.assign(base, cell.options) };
  }));
  s.addTable(body, { x, y, w: colW.reduce((a, b) => a + b, 0), colW, rowH: o.rowH || 0.32, fontFace: "Calibri", margin: [0.02, 0.06, 0.02, 0.06] });
}
const R = (t, o = {}) => ({ text: t, options: Object.assign({ color: RED, bold: true }, o) });
const B = (t, o = {}) => ({ text: t, options: Object.assign({ bold: true }, o) });
const M = (t, o = {}) => ({ text: t, options: Object.assign({ color: MUTED }, o) });
const notes = (s, lines) => s.addNotes(lines.join("\n"));

// ================= 1. TITLE =================
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
  txt(s, "CMSC 838B  ·  STUDENT LECTURE  ·  FALL 2026", 0.5, 0.55, 9, 0.28, { fontSize: 10, bold: true, color: MUTED, charSpacing: 1.5 });
  txt(s, "Neural SDEs", 0.5, 0.9, 9, 0.7, { fontSize: 36, bold: true, fontFace: "Arial" });
  txt(s, "Differentiating through stochastic dynamics", 0.5, 1.6, 9, 0.4, { fontSize: 18, color: "374151" });
  txt(s, [M("Running example: post-harvest ripening of 'Hass' avocado  ·  "), M(CREDIT, { italic: true })], 0.5, 2.05, 9, 0.3, { fontSize: 12 });
  txt(s, "Nishad Wajge  ·  University of Maryland", 0.5, 2.5, 9, 0.3, { fontSize: 12.5, bold: true });
  img(s, "p_fig1_stages.jpg", 0.5, 3.0, { w: 9 });
  credit(s, "Ripening stages 1–5. Fig. 1, " + CREDIT + ", CC BY 3.0.", 0.5, 5.02, 8);
  notes(s, [
    "[0:00 – 0:20]",
    "PLAN: ~23 min talk + 5 min discussion. SKIP-IF-SHORT: slides 7 and 20. Appendix (A1–A9) is not presented; jump there only if asked.",
    "• Topic: gradients through stochastic differential equations, and how to train models built from them.",
    "• Running example from a real food-science paper: Xavier et al., Foods 2024. Photos of avocados ripening, stages 1 to 5 shown here.",
  ]);
}

// ================= 2. THE DATA =================
{
  const s = content("Opening", "RUNNING EXAMPLE", "Xavier et al. (2024): Daily Images of 478 Avocados", "Three storage regimes, daily images, ordinal labels: an irregularly observed latent process.");
  table(s, [
    ["Group", "Storage", "n", "Stage 1→5 (fit)"],
    ["T10", "10 °C, 85% RH", "192", "17.6 d"],
    ["T20", "20 °C, 85% RH", "143", "8.5 d"],
    ["Tamb", "15.8–21.7 °C (mean 18.7), RH not logged", "143", "7.7 d"],
  ], 0.5, 1.4, [0.8, 2.6, 0.5, 1.15], { rowH: 0.36 });
  table(s, [
    ["Images", "Labels", "Split"],
    ["14,722 (2 sides × daily)", "5-stage index; 10-stage refinement", "70/15/15 by fruit"],
  ], 0.5, 3.15, [1.65, 2.25, 1.15], { fontSize: 10.5, rowH: 0.34 });
  img(s, "p_fig2_index.jpg", 5.5, 1.4, { w: 4.0 });
  credit(s, "Fig. 2 (cropped): 5- vs. 10-stage index; end of shelf life = entry to stage 5. " + CREDIT + ", CC BY 3.0.", 5.5, 2.83, 4.0);
  txt(s, [M("Shelf life:  "), B("τ = first day at stage 5")], 5.5, 3.25, 4.0, 0.3, { fontSize: 13 });
  credit(s, "Stage-1→5 durations computed from the paper's Table 4 slopes (4 × |α|).", 0.5, 4.45, 6);
  notes(s, [
    "[0:20 – 1:20]",
    "• Data: Xavier, Rodrigues and Silva, Foods 2024, open access. 478 Hass avocados photographed daily on both sides until overripe.",
    "• Three groups: 10 °C, 20 °C (both 85% humidity), and room temperature, which drifted between 15.8 and 21.7 °C with humidity not logged.",
    "• Labels: a 5-stage ripening index, refined to 10 stages. End of shelf life is reaching stage 5 (stage 9 on the 10-stage scale).",
    "• Last column: from their fitted slopes, going from stage 1 to 5 takes ~17.6 days at 10 °C, ~8.5 at 20 °C.",
    "• Splits are by fruit, so no avocado appears in both train and test.",
    "",
    "IF ASKED — why is the 10-stage index there? Finer resolution; the paper reports it's less accurate exactly (≈60%) but ≈93% within one stage.",
    "IF ASKED — license? The paper is CC BY 3.0; figures are reused with attribution. The dataset is on Mendeley Data (doi:10.17632/3xd9n945v8).",
  ]);
}

// ================= 3. THEIR PIPELINE =================
{
  const s = content("Opening", "RUNNING EXAMPLE", "Their Pipeline: Classify, Then Map to Days Left", "A strong per-image classifier; shelf life is a deterministic read-out of the current stage.");
  const boxes = ["Photo x", "CNN (AlexNet / ResNet-18)", "Stage ŝ ∈ {1,…,5}", "Days Left = α_g (ŝ − 5)"];
  boxes.forEach((t, i) => {
    const x = 0.5 + i * 2.3;
    card(s, x, 1.35, 2.0, 0.5, i === 3 ? { fill: "FFFFFF", line: RED } : {});
    txt(s, t, x, 1.35, 2.0, 0.5, { fontSize: 11.5, bold: true, align: "center", valign: "middle", color: i === 3 ? RED : INK });
    if (i < 3) arrow(s, x + 2.02, 1.6, x + 2.28, 1.6);
  });
  label(s, "Stage accuracy (test, by picture)", 0.5, 2.1, 4.3);
  table(s, [
    ["Network", "5-stage exact", "±1", "10-stage exact", "±1"],
    ["AlexNet", "76.9%", "99.4%", "57.4%", "92.6%"],
    ["ResNet-18", "78.4%", "99.7%", "60.8%", "93.3%"],
  ], 0.5, 2.38, [1.0, 1.0, 0.65, 1.05, 0.65], { fontSize: 11 });
  label(s, "Shelf-life MAE (days, 5-stage)", 5.3, 2.1, 4.2);
  table(s, [
    ["Group", "true stage", "predicted"],
    ["T10", "1.70", R("1.82")],
    ["T20", "0.73", "0.75"],
    ["Tamb", "0.53", "0.54"],
    ["All", "0.99", "1.04"],
  ], 5.3, 2.38, [1.2, 1.4, 1.6], { fontSize: 11 });
  credit(s, "Tables 3 and 6, " + CREDIT + ". 'Predicted' = by sample.", 0.5, 4.45, 8);
  notes(s, [
    "[1:20 – 2:20]",
    "• Their method: a CNN with transfer learning classifies the stage from one photo; a per-group linear regression turns the stage into days left.",
    "• Left: ~78% exact on 5 stages, >99% within one stage. 10 stages: ~60% exact, ~93% within one.",
    "• Right: mean absolute error in shelf life. Even with the TRUE stage label ('true stage' column), error is 1.7 days at 10 °C. So most of the error isn't the classifier; it's the stage → days map.",
    "• Note the paper's headline 88.8% is accuracy on the best side of each fruit; the table shows per-picture averages.",
    "",
    "IF ASKED — what's 'by sample'? Each fruit has two photos (sides); 'by sample' uses the better/combined side, which improves accuracy because one side can have marks.",
  ]);
}

// ================= 4. A DRIFT =================
{
  const s = content("Opening", "RUNNING EXAMPLE", "Their Shelf-Life Model, Read as a Constant-Drift ODE", "The regression line is the drift; the error bars are the spread it leaves out.");
  img(s, "p_fig6a_T10.jpg", 0.5, 1.3, { w: 4.7 });
  credit(s, "Fig. 6a (T10, cropped from Fig. 6): mean ± sd days left per stage. " + CREDIT + ", CC BY 3.0.", 0.5, 4.27, 4.7);
  img(s, "v3_drift.png", 5.45, 1.42, { w: 4.05 });
  table(s, [
    ["Group", "α (d / stage)", "v = 1/|α|", "R²"],
    ["T10", "−4.390", "0.228", "0.953"],
    ["T20", "−2.116", "0.473", "0.963"],
    ["Tamb", "−1.929", "0.518", "0.959"],
  ], 5.45, 2.15, [0.9, 1.15, 1.1, 0.9], { fontSize: 11 });
  credit(s, "α: Table 4 of the paper. v in stages / day (derived).", 5.45, 3.5, 4.05);
  txt(s, [M("Missing:  "), B("dW"), M(" (spread around the line), memory of past images, uncertainty in τ.")], 5.45, 3.8, 4.05, 0.6, { fontSize: 11.5 });
  notes(s, [
    "[2:20 – 3:20]",
    "• Their linear regression, Days Left = α(stage − 5), can be read as a constant-velocity ODE in stage: ds/dt ≈ v with v = 1/|α|.",
    "• Rates: 0.23 stages/day at 10 °C, 0.47 at 20 °C, 0.52 at room temperature. 20 °C ripens ~2× faster than 10 °C, as the paper reports.",
    "• Left plot (their Fig. 6a, T10): the error bars are standard deviations of days left at each stage, often ±2–3 days. A deterministic ODE can't represent that spread. That's the diffusion term.",
    "",
    "IF ASKED — is that inversion exact? No: regressing days on stage and inverting isn't the same as regressing stage on days (slopes differ by a factor R², ~4% here); stage is ordinal; and their fit is pinned at stage 5. Close enough to read off rates.",
    "IF ASKED — R² ≈ 0.96 looks great; why change it? R² measures fit of the mean; the decision (sell/discount by Friday) needs the distribution around the mean, which R² doesn't capture.",
  ]);
}

// ================= 5. RESIDUAL IS STOCHASTIC =================
{
  const s = content("Opening", "RUNNING EXAMPLE", "A Point Estimate Hides Days of Spread", "Stage quantization explains most of the error; the excess is largest at 10 °C, the longest horizon.");
  img(s, "p_fig11_T10_kde.jpg", 0.5, 1.3, { w: 5.2 });
  credit(s, "Fig. 11 (T10, cropped): shelf-life error vs. ripening index; left attributed labels, right predicted. " + CREDIT + ", CC BY 3.0.", 0.5, 3.75, 5.2);
  table(s, [
    ["Group", "|α| (d)", "floor |α|/4", "MAE", "floor / MAE"],
    ["T10", "4.39", "1.10", R("1.70"), R("65%")],
    ["T20", "2.12", "0.53", "0.73", "72%"],
    ["Tamb", "1.93", "0.48", "0.53", "91%"],
  ], 5.95, 1.4, [0.7, 0.7, 0.85, 0.6, 0.75], { fontSize: 10.5 });
  credit(s, "MAE in days with attributed (human) labels, Table 6. Floor = MAE from position within a stage alone (uniform, width |α|).", 5.95, 2.78, 3.6);
  txt(s, [M("Errors up to "), B("~6 days"), M(" at stages 1–4 (T10); the paper reports only the point.")], 5.95, 3.45, 3.6, 0.6, { fontSize: 11.5 });
  notes(s, [
    "[3:20 – 4:20]",
    "• Their Fig. 11, 10 °C group: density of shelf-life error against ripening index. Left uses the human labels, right the model's predictions. Average error ≈ 2 days at stages 1–4, tails to ~6 days; zero at stage 5 by definition.",
    "• Table: part of this is unavoidable. A stage lasts |α| days, and one label can't tell day 1 of a stage from day 4. Uniform position within a stage alone gives MAE = |α|/4: 65% of the error at 10 °C, 91% at room temperature.",
    "• The excess (fruit-to-fruit rate variation, labeling noise, within-fruit randomness) is what the model has to represent as a distribution. It's largest at 10 °C, where the horizon is longest.",
    "• Slide 13 compares the groups with a first-passage calculation.",
    "",
    "IF ASKED — couldn't this just be label noise? Partly: the 'attributed' labels are human, so they carry labeling noise too. Separating diffusion (sd ∝ √horizon) from fruit-to-fruit rate variation (sd ∝ horizon) needs the per-stage sds in Fig. 6; this slide doesn't settle it.",
    "IF ASKED — why |α|/4? The MAE of a uniform distribution of width w about its center is w/4.",
  ]);
}

// ================= 6. REQUIREMENTS =================
{
  const s = content("Opening", "RUNNING EXAMPLE", "What a Better Model Needs", "Continuous time + intrinsic noise + structured drift → latent neural SDE.");
  const Y = { text: "✓", options: { bold: true } }, N = { text: "—", options: { color: GRAY } }, P = { text: "partial", options: { color: MUTED, italic: true, fontSize: 10.5 } };
  table(s, [
    ["Requirement", "Per-image CNN (paper)", "Sequence model", "Latent ODE", { text: "Latent SDE", options: { color: RED } }],
    ["Use all past images", N, Y, Y, Y],
    ["Irregular / missing days", N, P, Y, Y],
    ["Distribution over τ", N, P, P, Y],
    ["Temperature-structured rate", P, N, Y, Y],
    ["Within-fruit noise (sd ∝ √h)", N, N, N, Y],
  ], 0.5, 1.4, [3.0, 1.6, 1.45, 1.4, 1.55], { rowH: 0.42, fontSize: 12 });
  notes(s, [
    "[4:20 – 5:00]",
    "• The paper's model uses one image and outputs a point. Rows: what we want.",
    "• A sequence model (GRU/Transformer with time-gap features and a hazard head) gets partway; a latent ODE handles irregular time and can spread through an uncertain initial state or per-fruit rate (sd ∝ horizon), but that uncertainty collapses as images accumulate. It has no noise that keeps arriving.",
    "• Latent SDE: all five by construction; its diffusion adds variance that grows like √h along the way.",
    "",
    "IF ASKED — 'partial' for the paper on temperature: they fit a separate slope per storage group; that's temperature-dependent but not a function of temperature (can't interpolate to 15 °C).",
  ]);
}

// ================= 7. NEURAL ODE =================
pres.addSection({ title: "Part 1: Differentiating solvers" });
{
  const s = content("Part 1: Differentiating solvers", "PART 1  ·  DIFFERENTIATING SOLVERS", "Neural ODEs: Continuous-Depth Dynamics", "ResNet = explicit Euler; a neural ODE delegates stepping to the solver.");
  label(s, "residual block", 0.5, 1.38, 4);
  img(s, "eq_resnet.png", 0.5, 1.64, { h: 0.33 });
  label(s, "Δt → 0", 0.5, 2.12, 4, RED);
  label(s, "neural ODE", 0.5, 2.48, 4);
  img(s, "eq_ode.png", 0.5, 2.74, { h: 0.6 });
  label(s, "latent version (paper = special case f = v_g)", 0.5, 3.6, 4.3);
  img(s, "v2_node_food.png", 0.5, 3.88, { h: 0.48 });
  img(s, "resnet_vs_ode.png", 4.8, 1.3, { w: 4.7, h: 3.2, center: true });
  notes(s, [
    "[5:00 – 5:45] — SKIP-IF-SHORT (most of the room knows this).",
    "• Residual block h + f(h) = one explicit Euler step. Δt → 0 gives dh/dt = f_θ(h, t); the solver chooses the steps.",
    "• Bottom: in the avocado setting, z(0) is an encoding of the first photo, f evolves it under storage condition c. The paper's model is the special case f = constant rate v_g in stage space.",
    "",
    "IF ASKED — costs? Each forward pass is a solve; adaptive solvers' step counts (NFE) tend to grow during training.",
  ]);
}

// ================= 8. THREE WAYS =================
{
  const s = content("Part 1: Differentiating solvers", "PART 1  ·  DIFFERENTIATING SOLVERS", "Three Ways to Differentiate a Solver", "Memory, compute and gradient exactness trade off; no method wins all three.");
  const mono = (t) => ({ text: t, options: { fontFace: "Courier New", fontSize: 9 } });
  table(s, [
    ["Method", "Memory (N steps)", "Extra compute", "Gradient", "Diffrax"],
    ["Backprop through steps", "O(N)", "≈ 1 backward pass", "exact (discrete)", mono("RecursiveCheckpointAdjoint(checkpoints=N)")],
    ["Binomial checkpointing", "O(log N)", "O(N log N)", "exact (discrete)", mono("RecursiveCheckpointAdjoint()")],
    [{ text: "Continuous adjoint", options: { color: RED, bold: true } }, "O(1) in N", "backward solve, dim 2d + p", R("approximate", { bold: false }), mono("BacksolveAdjoint()")],
  ], 0.5, 1.4, [1.9, 1.35, 1.75, 1.45, 2.55], { rowH: 0.55, fontSize: 11.5 });
  txt(s, [M("d = state dim, p = #params. Rows 1–2: discretize-then-differentiate. Row 3: differentiate-then-discretize.")], 0.5, 3.75, 9, 0.3, { fontSize: 10.5 });
  notes(s, [
    "[5:45 – 6:45]",
    "• Row 1: treat the solver as code and backprop. Exact gradient of the computed solution; stores every step.",
    "• Row 2: Griewank's binomial checkpointing (revolve): O(log N) checkpoints, O(N log N) recomputation, still exact.",
    "• Row 3: derive the continuous gradient as a backward ODE (next slide) and solve it. Memory independent of N, but the backward re-solve doesn't retrace the forward discrete path, so the gradient is approximate.",
    "",
    "IF ASKED — 2d + p? Backward state = z (d), adjoint a (d), running parameter gradient (p).",
  ]);
}

// ================= 9. ADJOINT =================
{
  const s = content("Part 1: Differentiating solvers", "PART 1  ·  DIFFERENTIATING SOLVERS", "The Adjoint Method", "Reverse-mode AD in continuous time: one VJP per solver stage.");
  label(s, "backward from T → 0 (derivation: A1)", 0.5, 1.38, 5);
  [["state", "v2_aug1.png"], ["adjoint a = ∂L/∂z(t)", "v2_aug2.png"], ["parameter gradient", "v2_aug3.png"]].forEach(([lab, f], i) => {
    const y = 1.78 + i * 0.72;
    txt(s, lab, 0.5, y, 1.75, 0.5, { fontSize: 10.5, bold: true, color: MUTED, valign: "middle" });
    img(s, f, 2.3, y + 0.04, i === 2 ? { w: 3.65, h: 0.5 } : { h: 0.36 });
  });
  txt(s, [M("a(T) = ∂L/∂z(T)")], 0.5, 4.02, 5, 0.3, { fontSize: 11 });
  const X0 = 6.35, X1 = 9.4;
  txt(s, "0", X0 - 0.1, 1.42, 0.3, 0.3, { fontSize: 11, bold: true, color: MUTED });
  txt(s, "T", X1 - 0.2, 1.42, 0.3, 0.3, { fontSize: 11, bold: true, color: MUTED, align: "right" });
  arrow(s, X0, 1.85, X1, 1.85, INK, 2);
  txt(s, "forward: z(T); store nothing", X0, 1.95, 3.1, 0.3, { fontSize: 10.5 });
  arrow(s, X1, 2.5, X0, 2.5, RED, 2);
  txt(s, "backward: (z, a, a_θ) jointly", X0, 2.6, 3.1, 0.3, { fontSize: 10.5, color: RED, bold: true });
  table(s, [["Term", "Primitive"], ["aᵀ ∂f/∂z", "VJP of f in z"], ["aᵀ ∂f/∂θ", "VJP of f in θ"]], 6.2, 3.15, [1.5, 1.8], { fontSize: 11 });
  notes(s, [
    "[6:45 – 8:30] — core slide.",
    "• a(t) = ∂L/∂z(t): sensitivity of the loss to the state at time t. Linear ODE backward from a(T) = ∂L/∂z(T).",
    "• a_θ starts at 0 at T; its value at 0 is dL/dθ.",
    "• z is re-integrated backward rather than stored: source of O(1) memory and of the approximation.",
    "• Both terms are vector–Jacobian products: one reverse-mode call of f per solver stage (Lecture 4's primitive). Never form the Jacobian.",
    "",
    "IF ASKED — derivation? Appendix A1 (Lagrange multiplier on ż = f, integrate by parts).",
    "IF ASKED — relation to Pontryagin? a is the costate; the adjoint ODE is the costate equation.",
  ]);
}

// ================= 10. ITÔ =================
pres.addSection({ title: "Part 2: Stochastic dynamics" });
{
  const s = content("Part 2: Stochastic dynamics", "PART 2  ·  STOCHASTIC DYNAMICS", "Itô Calculus: What You Need", "Itô for modeling; Stratonovich for gradients (ordinary chain rule).");
  [["SDE (integral form)", "v2_ito_def.png", 0.36], ["Brownian increments", "v2_bm.png", 0.3], ["Itô's lemma", "v2_itolemma.png", 0.32], ["Itô → Stratonovich drift", "v2_conv.png", 0.42]].forEach(([lab, f, h], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.36 + Math.floor(i / 2) * 1.2;
    card(s, x, y, 4.4, 1.08, i === 3 ? { fill: "FFFFFF", line: RED } : {});
    label(s, lab, x + 0.18, y + 0.08, 4.0, i === 3 ? RED : MUTED);
    img(s, f, x + 0.18, y + 0.4, { w: 4.0, h: h + 0.22, vcenter: true });
  });
  table(s, [["Scheme", "Calculus", "Strong", "Weak"], ["Euler–Maruyama", "Itô", "½", "1"], ["Milstein", "Itô / Strat.", "1", "1"], ["Reversible Heun", "Strat.", "½", "—"]],
    0.5, 3.83, [2.2, 1.6, 0.9, 0.9], { fontSize: 10, rowH: 0.22 });
  txt(s, [M("Full table: A7")], 6.3, 4.4, 3, 0.25, { fontSize: 10 });
  notes(s, [
    "[8:30 – 10:15]",
    "• An SDE is shorthand for the integral equation; the dW integral is Itô (left endpoint).",
    "• Increments are independent N(0, Δt); (dW)² = dt is why stochastic calculus differs.",
    "• Itô's lemma: the extra ½ f″ σ² dt term. Example: d log S picks up −½σ².",
    "• Stratonovich (midpoint) obeys the ordinary chain rule; the drift changes by −½ Σ_j (∂σ_j/∂z) σ_j. Zero when σ doesn't depend on z.",
    "• Orders: Euler–Maruyama strong ½ (paths), weak 1 (expectations).",
    "",
    "IF ASKED — strong vs weak? Strong: E|Z_T − Ẑ_T|. Weak: |E f(Z_T) − E f(Ẑ_T)|. Probabilities like P(τ ≤ t*) need weak; pathwise gradients care about strong.",
    "IF ASKED — which does Euler converge to? Euler–Maruyama → Itô; Heun / midpoint → Stratonovich.",
  ]);
}

// ================= 11. NEURAL SDE FOR RIPENING =================
{
  const s = content("Part 2: Stochastic dynamics", "PART 2  ·  STOCHASTIC DYNAMICS", "A Latent Neural SDE for Ripening", "Drift = rate (physics prior × network); diffusion = fruit-to-fruit variability; τ = first passage.");
  const boxes = [["images x(t_k)", "irregular t_k"], ["encoder e_φ", "CNN / ViT"], ["latent SDE Z_t", "storage c"], ["ordinal head", "stages 1–5"], ["τ", "first passage"]];
  boxes.forEach(([a, b], i) => {
    const x = 0.5 + i * 1.84;
    card(s, x, 1.35, 1.6, 0.66, i === 2 ? { fill: "FFFFFF", line: RED } : {});
    txt(s, a, x + 0.05, 1.39, 1.5, 0.3, { fontSize: 11.5, bold: true, align: "center", color: i === 2 ? RED : INK });
    txt(s, b, x + 0.05, 1.68, 1.5, 0.28, { fontSize: 9.5, color: MUTED, align: "center" });
    if (i < 4) arrow(s, x + 1.62, 1.68, x + 1.82, 1.68);
  });
  label(s, "dynamics", 0.5, 2.2, 3);
  img(s, "v2_ripen.png", 0.5, 2.45, { w: 4.5, h: 0.38 });
  img(s, "v2_arrh.png", 5.3, 2.4, { w: 4.2, h: 0.46 });
  label(s, "observation model", 0.5, 3.05, 3);
  img(s, "v3_ord.png", 0.5, 3.3, { w: 4.5, h: 0.34 });
  label(s, "target", 5.3, 3.05, 3);
  img(s, "v2_tau.png", 5.3, 3.3, { w: 4.2, h: 0.4 });
  table(s, [["Paper", "Here"], ["ŝ from one image", "filtered Z_t from all images"], ["Days Left = α_g(ŝ − 5)", "P(τ ≤ t* | images up to day k, c)"]], 0.5, 3.82, [3.0, 3.6], { fontSize: 10.5, rowH: 0.27 });
  notes(s, [
    "[10:15 – 11:45]",
    "• Encode each photo; Z_t evolves as an SDE between photos; an ordinal head maps Z to stage probabilities; shelf life τ is when the ripeness score crosses the stage-4/5 cut.",
    "• Drift: Arrhenius factor κ(c) times a neural rate. E_a activation energy, R gas constant, T in kelvin. Diffusion σ_θ: variability across fruit and storage.",
    "• Ordinal head: cumulative-logit model, P(y ≤ k) = sigmoid(c_k − q(Z)), cut points c_k ordered. b = c_4 defines 'end of shelf life'.",
    "• Bottom: what changes vs the paper. They classify one image and map to a point; we filter all images and output a probability.",
    "",
    "IF ASKED — how do you compute P(τ ≤ t*)? Monte Carlo from the filtered posterior at the last photo; count crossings, with a Brownian-bridge correction between steps (A6).",
    "IF ASKED — should σ scale with temperature? If temperature speeds up the whole biological clock, σ ∝ √κ. Slide 13 tests exactly this against the paper's numbers.",
  ]);
}

// ================= 12. ARRHENIUS =================
{
  const s = content("Part 2: Stochastic dynamics", "PART 2  ·  CALIBRATING THE PRIOR", "Fitting the Drift Prior to the Paper's Rates", "Two temperatures pin Eₐ; ambient ripens 20% faster than Arrhenius predicts → learned residual.");
  img(s, "v3_arrhenius.png", 0.5, 1.3, { w: 4.8 });
  img(s, "v3_arrh_fit.png", 5.6, 1.4, { h: 0.55 });
  table(s, [
    ["Quantity", "Value"],
    ["Q₁₀ = v₂₀ / v₁₀", "2.07"],
    ["Eₐ", "50.4 kJ/mol"],
    ["Tamb, predicted (18.7 °C)", "2.32 d / stage"],
    ["Tamb, observed", R("1.93 d / stage")],
    ["Temperature implied", "21.3 °C"],
  ], 5.6, 2.15, [2.4, 1.5], { fontSize: 11 });
  credit(s, "Rates from the paper's Table 4; fit is ours (A5).", 5.6, 4.15, 3.9);
  notes(s, [
    "[11:45 – 12:45]",
    "• Using their per-group slopes: rates 0.228 (10 °C) and 0.473 (20 °C) stages/day. Q10 ≈ 2.07, so Eₐ ≈ 50 kJ/mol, a plausible enzymatic value.",
    "• Predict room temperature (mean 18.7 °C): 2.32 days per stage. Observed: 1.93. Ambient fruit ripened ~20% faster than temperature alone explains; you'd need 21.3 °C, near the top of the recorded range.",
    "• Likely culprits: humidity wasn't controlled in Tamb, and temperature fluctuated. Point: a physics prior gets you most of the way; a learned residual absorbs what it misses. That's why the drift is κ(c) × network.",
    "",
    "IF ASKED — is two points enough to fit Arrhenius? It determines Eₐ exactly but gives no check of the functional form; that's why I treat it as a prior, not a law.",
    "IF ASKED — could temperature fluctuation explain it? Rate is convex in T, but with sd 1.2 °C the Jensen effect is under 1%. Not enough for 20%.",
  ]);
}

// ================= 13. FIRST-PASSAGE CHECK =================
{
  const s = content("Part 2: Stochastic dynamics", "PART 2  ·  A QUICK TEST", "Do the Paper's Errors Have a Drift–Diffusion Shape?", "T10/T20 = 2.33 exceeds quantization alone (2.07): extra spread at 10 °C. Three numbers can't pin σ(T).");
  label(s, "first passage of drifted Brownian motion (A4)", 0.5, 1.38, 5);
  img(s, "v3_ig_a.png", 0.5, 1.64, { h: 0.3 });
  img(s, "v3_ig_b.png", 0.5, 2.0, { h: 0.36 });
  img(s, "v3_ratio.png", 0.5, 2.5, { w: 4.4 });
  table(s, [["Ratio", "σ fixed", "σ² ∝ v", "quant. only", "observed"], ["T10 / T20", "2.99", "2.07", "2.07", R("2.33")], ["Tamb / T20", "0.87", "0.91", "0.91", R("0.73")]], 0.5, 3.25, [1.05, 0.8, 0.8, 0.95, 0.85], { fontSize: 10.5 });
  credit(s, "Observed = MAE ratios with attributed labels (paper, Table 6).", 0.5, 4.25, 4.6);
  img(s, "v3_fpt_ratio.png", 5.0, 1.3, { w: 4.5, h: 3.3, center: true });
  notes(s, [
    "[12:45 – 14:00]",
    "• Back-of-envelope: if stage drifts at rate v with noise σ, the time to cover distance L is inverse-Gaussian with mean L/v and sd √(Lσ²/v³).",
    "• Two hypotheses. σ fixed across temperatures: error ratio (v₂₀/v₁₀)^{3/2} ≈ 3.0 for T10/T20. Temperature rescales the whole clock (σ² ∝ v): ratio v₂₀/v₁₀ ≈ 2.1.",
    "• Confound: stage quantization alone (slide 5's floor, |α|/4) also scales as 1/v, so it predicts the same 2.07.",
    "• Observed (MAE as a proxy for sd): 2.33. Above 2.07, so there is extra spread at 10 °C beyond quantization; below 3.0. Three numbers can't separate σ-scaling from quantization, so I read this as 'consistent with drift + diffusion', not a fit of σ(T).",
    "• Tamb comes out lower than either. Consistent with Tamb being a different regime (humidity, fluctuating temperature), which slide 12 also showed.",
    "• Caveat: MAE of a stage-based estimate, not the sd of τ, and ignores classification noise. It's a sanity check that a drift–diffusion model is the right shape, not a fit.",
    "",
    "IF ASKED — why MAE ∝ sd? For a fixed distribution family, MAE is a constant multiple of sd (0.8 for Gaussian). That assumes the coefficient of variation is small; under fixed σ the inverse-Gaussian CV ∝ v^(−1/2) differs 1.44× between T10 and T20, so treat the ratios as rough.",
  ]);
}

// ================= 14. PATHWISE VS SCORE =================
pres.addSection({ title: "Part 3: Gradients through noise" });
{
  const s = content("Part 3: Gradients through noise", "PART 3  ·  GRADIENTS THROUGH NOISE", "Pathwise vs. Score-Function Gradients", "Our target, P(τ ≤ t*), is discontinuous pathwise: the reparameterized gradient is zero.");
  label(s, "pathwise (reparameterization)", 0.5, 1.36, 4.8);
  img(s, "v2_pathwise.png", 0.5, 1.62, { w: 4.6 });
  label(s, "score function (likelihood ratio)", 0.5, 2.12, 4.8);
  img(s, "v2_score.png", 0.5, 2.38, { w: 4.6 });
  img(s, "indicator.png", 5.2, 1.3, { w: 4.3, h: 1.65, center: true });
  table(s, [
    ["Estimator for P(τ ≤ t*)", "Bias", "Variance", "Notes"],
    ["Pathwise on 1{τ ≤ t*}", R("total (∇ = 0)"), "—", "fails"],
    ["Sigmoid surrogate (ε)", "O(ε)", "↑ as ε → 0", "needs ε schedule"],
    ["Brownian-bridge crossing", "low", "low", "smooth per step (A6)"],
    ["Score function (Girsanov)", "none", "high", "drift params only"],
  ], 0.5, 2.98, [2.6, 1.6, 1.3, 2.0], { fontSize: 10.5, rowH: 0.3 });
  notes(s, [
    "[14:00 – 15:45]",
    "• Pathwise: Z_T as a deterministic function of θ and fixed noise W; differentiate inside E. VAE reparameterization trick. Low variance; needs f differentiable.",
    "• Score function: E[f ∇ log p_θ]; any f, high variance.",
    "• Plot: for each fixed noise path, 'spoiled by t*' is a step in θ, derivative zero almost everywhere. The expectation (red) has nonzero slope. Pathwise returns 0: a biased gradient, not a noisy one.",
    "• Table: four estimators. Brownian-bridge crossing probability is the one I'd use: smooth in the endpoints, low variance, exact for frozen coefficients within a step.",
    "",
    "IF ASKED — Girsanov ratio? For drift parameters: ∇_θ log p = ∫ (∂_θ(σ⁻¹μ))ᵀ dW. Parameters in σ change quadratic variation → path measures mutually singular in continuous time; on a grid the EM density exists but its estimator's variance blows up as Δt → 0.",
    "IF ASKED — finance analogue? Digital and barrier options: same problem; same fixes (smoothing, likelihood-ratio Greeks, bridge corrections). Mohamed et al. JMLR 2020 for the ML view.",
  ]);
}

// ================= 15. STOCHASTIC ADJOINT =================
{
  const s = content("Part 3: Gradients through noise", "PART 3  ·  GRADIENTS THROUGH NOISE", "The Stochastic Adjoint", "ODE adjoint + a noise term driven by the replayed Brownian path.");
  label(s, "backward from T, Stratonovich (∘), same W; μ = Stratonovich drift", 0.5, 1.36, 5.4);
  img(s, "v2_sadj.png", 0.5, 1.66, { w: 4.6 });
  img(s, "v2_sadj_th.png", 0.5, 2.3, { w: 4.6 });
  credit(s, "Li, Wong, Chen & Duvenaud, AISTATS 2020. Why Stratonovich: A2.", 0.5, 3.26, 5);
  table(s, [
    ["Requirement", "Solution", "Cost"],
    ["Same W forward and backward", "virtual Brownian tree / Brownian Interval", "O(1) memory, O(log 1/ε) per query"],
    ["Chain rule must hold", "Stratonovich form (convert Itô drift)", "—"],
    [{ text: "Exact gradients", options: { color: RED, bold: true } }, "reversible Heun + matching discrete adjoint (torchsde)", "exact up to round-off"],
  ], 0.5, 3.55, [2.3, 3.6, 3.1], { fontSize: 10.5, rowH: 0.3, align: "left" });
  notes(s, [
    "[15:45 – 17:15]",
    "• Compare slide 9: an extra term per noise channel, aᵀ ∂σ_j/∂z ∘ dW^j, and the parameter gradient gets an integral against dW.",
    "• Requirement 1: backward pass must see the same noise. A virtual Brownian tree regenerates W(s) from a seed by recursive Brownian-bridge sampling (A6).",
    "• Requirement 2: in Stratonovich form the ordinary chain rule holds, so the ODE derivation carries over. μ here is the Stratonovich drift: convert the Itô model first (μ̃ = μ − ½Σ(∂σ_j/∂z)σ_j), or the gradients are wrong.",
    "• Exactness: Kidger et al. NeurIPS 2021: reversible Heun is algebraically reversible; with its matching discrete adjoint, gradients are exact up to floating-point round-off at O(1) memory. In torchsde. Diffrax's BacksolveAdjoint is the continuous adjoint, approximate even with reversible Heun.",
    "",
    "IF ASKED — Itô directly? Possible, but the backward-in-time Itô SDE needs extra correction terms.",
  ]);
}

// ================= 16. MEASURED =================
{
  const s = content("Part 3: Gradients through noise", "PART 3  ·  MEASURED", "Three Ways to Get the Gradient, Measured", "In Diffrax, checkpointing beats the adjoint on memory, time and exactness at this scale.");
  asset(s, "adjoint_tradeoffs.png", 0.5, 1.25, { w: 9, h: 2.5 });
  table(s, [["512 steps", "Memory", "Time / grad", "Gradient"], ["Backprop", "10.3 MB", "2.1 s", "exact"], ["Checkpointing", "1.0 MB", "3.0 s", "exact"], ["Adjoint", R("154 MB"), R("137 s"), "rel. err 1.8e-6"]],
    0.5, 3.82, [1.6, 1.2, 1.2, 1.6], { fontSize: 9.5, rowH: 0.21 });
  credit(s, "16-dim neural SDE, 32 paths, reversible Heun, CPU. github.com/nishadw/neural-sde. Setup: A8.", 6.3, 3.88, 3.2);
  notes(s, [
    "[17:15 – 18:00]",
    "If time or asked for a demo: run lecture/neural_sde_demo.py live, or play assets/demo.mp4 from the repo.",
    "• Memory (log scale): backprop linear in steps, checkpointing ~log, adjoint flat but ~150 MB. The method needs only 2d + p ≈ 110 KB; the rest is a Diffrax/XLA implementation constant growing with params and noise channels (A8).",
    "• Time: adjoint ~60× slower.",
    "• Error: adjoint vs backprop falls ~1/steps: discretization error of optimize-then-discretize. Plain Heun slightly better than reversible Heun here: reversibility alone doesn't make the continuous adjoint exact.",
    "• Matches Diffrax's own recommendation: use checkpointing.",
    "",
    "IF ASKED — why so large? Most likely per-noise-channel parameter cotangents materialized (512 × 12.7k × 8 B ≈ 52 MB, plus temporaries); measured scaling is consistent, not profiled buffer by buffer.",
  ]);
}

// ================= 17. ELBO =================
pres.addSection({ title: "Part 4: Training" });
{
  const s = content("Part 4: Training", "PART 4  ·  TRAINING", "Latent-SDE ELBO on the Avocado Data", "Girsanov: path-space KL = running cost ∫½‖u‖²dt, integrated with the state.");
  label(s, "prior (forecast)", 0.5, 1.36, 5.4);
  img(s, "v2_prior.png", 0.5, 1.6, { w: 5.2 });
  label(s, "posterior (filter, sees images)", 0.5, 2.0, 5.4, RED);
  img(s, "v2_post.png", 0.5, 2.24, { w: 5.2 });
  label(s, "objective", 0.5, 2.68, 5.4);
  img(s, "v2_elbo.png", 0.5, 2.94, { w: 5.2 });
  img(s, "v2_u.png", 0.5, 3.52, { h: 0.27 });
  credit(s, "Li et al. 2020. KL derivation: A3.", 0.5, 3.9, 5);
  table(s, [["ELBO term", "Avocado data"], ["y_k", "stage label of image k"], ["t_k", "photo day (irregular)"], ["c", "T10 / T20 / Tamb"], ["shared σ", "else KL = ∞"]], 6.0, 1.4, [1.3, 2.2], { fontSize: 10.5, rowH: 0.33 });
  notes(s, [
    "[18:00 – 19:30]",
    "• Latent SDE (Li et al. 2020): a VAE whose latent is a path.",
    "• Prior: our ripening SDE given only storage. Posterior: same σ, drift also sees the images (e.g., a GRU run backward over encoded photos gives a context per time).",
    "• ELBO: expected log-likelihood of observed stage labels at photo times − KL of initial state − KL between the SDEs.",
    "• Girsanov: path KL = E ∫ ½‖u‖² dt with σu = h_φ − h_θ. Append it as one extra state; each solve gives a Monte Carlo estimate.",
    "• Gradients of all of it flow through the solver: slides 8–16.",
    "",
    "IF ASKED — why shared σ? Girsanov changes drift only; different σσᵀ ⇒ different quadratic variation ⇒ mutually singular measures ⇒ KL = ∞.",
    "IF ASKED — forecasting? After the last photo, switch to the prior and sample forward for P(τ ≤ t*).",
  ]);
}

// ================= 18. DIFFUSION =================
{
  const s = content("Part 4: Training", "PART 4  ·  CONNECTION", "Diffusion Models Are Neural SDEs", "Diffusion = neural SDE with fixed σ, trained simulation-free; ours learns σ and backprops through the solver.");
  img(s, "diffusion_paths.png", 0.5, 1.38, { w: 4.5, h: 2.3, center: true });
  let y = 1.34;
  [["forward (fixed)", "eq_fwd.png", 0.25], ["reverse (learned score)", "eq_rev.png", 0.25], ["probability-flow ODE", "eq_pf.png", 0.46], ["denoising score matching", "v2_dsm.png", 0.3]].forEach(([lab, f, h], i) => {
    label(s, lab, 5.3, y, 4.2, i === 1 ? RED : MUTED);
    img(s, f, 5.3, y + 0.26, { w: 4.2, h });
    y += 0.26 + h + 0.16;
  });
  table(s, [["", "Latent SDE (avocado)", "Diffusion model"], ["learns", "μ, σ", "score only"], ["training", "backprop through solver", "simulation-free"]], 0.5, 3.75, [0.9, 1.9, 1.7], { fontSize: 10, rowH: 0.24 });
  notes(s, [
    "[19:30 – 20:30]",
    "• Song et al. 2021: a fixed forward SDE noises data; Anderson (1982): its time reversal needs the score ∇ log p_t. Learn s_θ: the reverse SDE is a neural SDE.",
    "• Probability-flow ODE: same marginals as the reverse SDE when s_θ is the true score; deterministic sampling; likelihoods via change of variables (Hutchinson trace estimator). Flow matching trains that ODE's velocity directly (π0 robot policies).",
    "• Training is simulation-free: forward SDE is affine with known time-varying coefficients, so p(x_t | x_0) is Gaussian in closed form. No solver, no adjoint during training.",
    "• Contrast: our model learns the dynamics, so it must backprop through the solve.",
  ]);
}

// ================= 19. PROPOSED EVALUATION =================
{
  const s = content("Part 4: Training", "RUNNING EXAMPLE", "Proposed Evaluation Against the Paper's Baseline", "Same by-fruit split and MAE as the paper, plus calibration metrics that reward a distribution.");
  const T = { text: "TBD", options: { color: GRAY, italic: true } };
  table(s, [
    ["Shelf-life metric", "Xavier et al. (CNN + linear)", "Latent ODE", { text: "Latent SDE", options: { color: RED } }],
    ["MAE, T10 (days)", "1.82", T, T],
    ["MAE, T20 (days)", "0.75", T, T],
    ["MAE, Tamb (days)", "0.54", T, T],
    ["MAE, all (days)", "1.04", T, T],
    ["CRPS of τ, all (days)", { text: "1.04 (= MAE for a point)", options: { color: MUTED } }, T, T],
    ["80% interval coverage", { text: "n/a", options: { color: MUTED } }, T, T],
  ], 0.5, 1.4, [2.6, 2.6, 1.7, 1.7], { fontSize: 11.5, rowH: 0.38 });
  credit(s, "Baseline: Table 6 (5-stage, by sample), " + CREDIT + ". TBD = experiment in progress.", 0.5, 4.2, 8);
  notes(s, [
    "[20:30 – 21:30]",
    "TODO (project): fill the TBD cells once the experiment runs; keep the paper's 70/15/15 by-fruit split so numbers are comparable.",
    "• Left column: the paper's numbers, by sample, 5-stage.",
    "• MAE compares point accuracy (use the predictive median of τ).",
    "• CRPS (continuous ranked probability score) and interval coverage evaluate the whole predictive distribution. A point forecast has CRPS equal to its MAE, so the baseline is 1.04 days; a calibrated distribution should beat it. Coverage needs an interval, which the paper doesn't give (a fairer baseline: Gaussian intervals from Fig. 6's per-stage sds).",
    "• Caveats on the baseline: the paper doesn't say whether Table 6 is test-only or whether α was fit on all fruit, and 'by sample' keeps the better of the two sides. Refit α on the training fruit when reproducing it.",
    "",
    "IF ASKED — CRPS? CRPS(F, y) = ∫ (F(t) − 1{y ≤ t})² dt; reduces to MAE for a point forecast; lower is better; proper scoring rule.",
    "IF ASKED — what if results aren't ready? Present this as the experimental plan; the baseline column is real.",
  ]);
}

// ================= 20. WHERE ELSE =================
{
  const s = content("Part 4: Training", "BEYOND PRODUCE", "Same Structure Elsewhere", "Wherever decisions depend on the distribution over trajectories.");
  table(s, [
    ["Domain", "Latent state", "Neural SDE role", "Reference"],
    ["Weather", "atmosphere", "diffusion sampling of ensembles", "Price et al., Nature 2025 (GenCast)"],
    ["Robotics", "action trajectory", "reverse diffusion / flow matching", "Chi et al. 2023; Black et al. 2024"],
    ["Quant finance", "price, volatility", "calibrated market models", "Gierjatowicz 2022; Cohen 2021"],
    ["Clinical", "patient state", "irregular vitals (ODE/CDE family)", "Rubanova 2019; Kidger 2020"],
    ["Generative models", "image / audio", "score-based sampling", "Song et al. 2021"],
  ], 0.5, 1.4, [1.6, 1.6, 2.7, 3.1], { fontSize: 11, rowH: 0.4, align: "left" });
  notes(s, [
    "[21:30 – 22:00] — SKIP-IF-SHORT.",
    "• Same pattern as the avocado: hidden state, randomness, sparse observations, decisions on the distribution of futures.",
    "• My repo also has the finance case: a neural local-vol SDE calibrated to a Heston smile by backprop through Monte Carlo (0.31 vol pts RMSE vs 4.21 flat).",
  ]);
}

// ================= 21. OPEN PROBLEMS =================
{
  const s = content("Part 4: Training", "LIMITS", "Open Problems", "O(1) memory is real; exact, unbiased, identifiable gradients each cost extra.");
  table(s, [
    ["Issue", "Symptom", "Evidence / mitigation"],
    ["Adjoint error", "backward re-solve ≠ forward path", "ANODE (2019); 1e-4–1e-6 here"],
    ["Stiffness", "NFE grows during training", "kinetic / Jacobian regularization (Finlay 2020)"],
    ["Discontinuous targets", "∇ 1{τ ≤ t*} = 0 pathwise", "bridge crossing, surrogate, score function"],
    ["Identifiability", "drift ↔ diffusion trade-off; latent reparam.", "ordinal head anchors q; synthetic recovery test"],
    ["Visual confounds", "lighting, marks, wind rub look like ripening", "paper: per-sample > per-picture accuracy"],
    ["Uncontrolled covariates", "Tamb 20% faster than Arrhenius", "condition on humidity if logged"],
  ], 0.5, 1.4, [1.9, 3.3, 3.8], { fontSize: 11, rowH: 0.42, align: "left" });
  notes(s, [
    "[22:00 – 23:00] — analysis portion.",
    "• Adjoint gradient isn't the gradient of what the forward solver computed; ANODE showed it can be badly wrong on stiff problems; measured 1e-4 to 1e-6 here.",
    "• Learned dynamics stiffen; adaptive solvers take more steps; regularize kinetic energy / Jacobian norms.",
    "• Our main target is discontinuous: naive pathwise training gives zero gradient.",
    "• Identifiability: latent reparameterizations leave the likelihood unchanged; fast drift + small noise can mimic slow drift + big noise with sparse observations. Test by recovering a known SDE on synthetic data.",
    "• Paper-specific: marks and wind rub (they report atypical visual damage that season) look like ripening; and ambient storage behaves differently from what temperature predicts (slide 12).",
  ]);
}

// ================= 22. DISCUSSION =================
{
  const s = content("Part 4: Training", "DISCUSSION", "Questions for the Room", null);
  const qs = [
    "Which estimator would you train P(τ ≤ t*) with: surrogate, bridge crossing, or score function?",
    "Slide 13 suggests extra spread at low temperature. How would you test whether σ depends on temperature?",
    "Which differentiable simulator from this course would you make stochastic, and what happens to its gradients?",
  ];
  qs.forEach((q, i) => {
    const y = 1.45 + i * 1.0;
    txt(s, String(i + 1), 0.5, y, 0.4, 0.6, { fontSize: 22, bold: true, color: i === 0 ? RED : GRAY });
    txt(s, q, 1.0, y + 0.05, 8.4, 0.7, { fontSize: 16 });
  });
  txt(s, [M("Code, benchmarks, slides: github.com/nishadw/neural-sde")], 0.5, 4.6, 9, 0.3, { fontSize: 11 });
  notes(s, [
    "[23:00 – 28:00]",
    "• Q1: bias vs variance. Bridge crossing: exact per step for frozen coefficients, low variance. Score function: unbiased, noisy, drift params only. Sigmoid: needs ε annealing.",
    "• Q2: fit the latent SDE with σ_θ(c) free per group, compare to σ ∝ √κ(c) by held-out likelihood; or estimate per-group variance of daily stage increments directly from the label sequences.",
    "• Q3 (Prof. Lin's lectures): stochastic forcing adds Monte Carlo variance and needs same-noise replay; for contact-rich systems noise can smooth the loss landscape (randomized smoothing).",
    "• Close: thank you; questions. Appendix follows (not presented).",
  ]);
}

// ================= APPENDIX (not presented) =================
pres.addSection({ title: "Appendix" });
const appx = (code, title, takeaway) => content("Appendix", "APPENDIX  ·  " + code + "  ·  NOT PRESENTED", title, takeaway);

{ // A1
  const s = appx("A1", "Derivation: ODE Adjoint", "Choose a(t) to cancel every ∂z/∂θ term; what remains is dL/dθ.");
  label(s, "1. Lagrangian with multiplier a(t) on ż = f", 0.5, 1.38, 9);
  img(s, "v2_lag.png", 0.5, 1.64, { h: 0.6 });
  label(s, "2. Differentiate in θ; integrate aᵀ (d/dt)(∂z/∂θ) by parts", 0.5, 2.38, 9);
  txt(s, "∂z/∂θ terms:  [ ∂L/∂z(T) − a(T) ]·∂z(T)/∂θ  +  ∫ (ȧ + aᵀ ∂f/∂z)·∂z/∂θ dt  +  a(0)ᵀ ∂z(0)/∂θ", 0.5, 2.64, 9, 0.35, { fontSize: 12 });
  label(s, "3. Set both brackets to zero", 0.5, 3.12, 9);
  img(s, "v2_lag2.png", 0.5, 3.38, { h: 0.6 });
  txt(s, [M("Add a(0)ᵀ ∂z(0)/∂θ if the initial state depends on θ (e.g., z(0) = e_φ(x₀)).")], 0.5, 4.1, 9, 0.3, { fontSize: 11 });
  notes(s, ["Appendix A1. Since ż − f = 0 on solutions, adding ∫aᵀ(ż − f)dt leaves L unchanged for any a. Integrate by parts to move d/dt off ∂z/∂θ; choose a to kill those terms."]);
}
{ // A2
  const s = appx("A2", "Why Stratonovich for the Stochastic Adjoint", "Stratonovich SDEs obey the ordinary chain rule, so the ODE adjoint carries over term by term.");
  img(s, "v2_strat_ode.png", 0.5, 1.45, { w: 9, h: 0.45 });
  table(s, [
    ["", "Itô", "Stratonovich"],
    ["integral evaluated at", "left endpoint", "midpoint"],
    ["martingale property", "yes", "no"],
    ["chain rule", "Itô's lemma (+½ f″σ² dt)", "ordinary"],
    ["time reversal", "extra correction terms", "same form"],
    ["drift conversion", "μ", "μ − ½ Σⱼ (∂σⱼ/∂z) σⱼ"],
  ], 0.5, 2.15, [2.6, 3.0, 3.4], { fontSize: 11.5, rowH: 0.36 });
  notes(s, ["Appendix A2. Rough-path view: σ ∘ dW behaves like an input signal; the ODE adjoint (A1) applies with one extra term per noise channel."]);
}
{ // A3
  const s = appx("A3", "Derivation: Path-Space KL via Girsanov", "Same σσᵀ, different drift: log dQ/dP = stochastic integral + running cost.");
  label(s, "Girsanov (σu = h_φ − h_θ)", 0.5, 1.38, 9);
  img(s, "v2_girs1.png", 0.5, 1.64, { h: 0.66 });
  label(s, "Under Q: W^P = W^Q + ∫u dt; the Itô integral has zero mean", 0.5, 2.45, 9);
  img(s, "v2_girs2.png", 0.5, 2.71, { h: 0.66 });
  txt(s, [B("Conditions: "), M("same σσᵀ; Novikov: E_P exp(½∫‖u‖²dt) < ∞. "), B("In code: "), M("dℓ = ½‖u‖² dt appended to the state; ℓ_T is a one-sample MC estimate.")], 0.5, 3.6, 9, 0.6, { fontSize: 11.5 });
  notes(s, ["Appendix A3. Different σσᵀ ⇒ different quadratic variation ⇒ mutually singular measures ⇒ KL = ∞."]);
}
{ // A4
  const s = appx("A4", "Derivation: First-Passage Ratios (Slide 13)", "Inverse-Gaussian first passage gives sd(τ) ∝ √(Lσ²/v³).");
  label(s, "density of τ for X = vt + σW, barrier at L > 0", 0.5, 1.38, 9);
  img(s, "v3_ig_den.png", 0.5, 1.64, { h: 0.6 });
  txt(s, [M("⇒  E τ = L / v,   Var τ = L σ² / v³   (inverse Gaussian, mean L/v, shape L²/σ²)")], 0.5, 2.36, 9, 0.3, { fontSize: 12 });
  label(s, "ratios between storage groups at equal distance L", 0.5, 2.8, 9);
  img(s, "v3_ratio.png", 0.5, 3.06, { w: 8, h: 0.5 });
  txt(s, [M("Rates from Table 4: v₁₀ = 0.228, v₂₀ = 0.473, v_amb = 0.518 stages/day. MAE used as a proxy for sd (assumes small CV); quantization predicts v₂₀/v too; labeling noise ignored.")], 0.5, 3.75, 9, 0.6, { fontSize: 11 });
  notes(s, ["Appendix A4. Derivation via the Laplace transform of the hitting time of Brownian motion with drift, or via Wald's identities: E τ = L/v; Var τ = Lσ²/v³."]);
}
{ // A5
  const s = appx("A5", "Arrhenius Fit (Slide 12)", "Eₐ from two rates; prediction at the ambient mean; implied temperature.");
  table(s, [
    ["Step", "Expression", "Value"],
    ["rates", "v = 1 / |α|", "0.228, 0.473, 0.518 d⁻¹"],
    ["Q₁₀", "v₂₀ / v₁₀", "2.07"],
    ["Eₐ", "R ln Q₁₀ / (1/283.15 − 1/293.15)", "50.4 kJ/mol"],
    ["v_amb predicted", "v₂₀ exp[−Eₐ/R (1/291.85 − 1/293.15)]", "0.431 d⁻¹ (2.32 d/stage)"],
    ["v_amb observed", "1 / 1.929", R("0.518 d⁻¹ (1.93 d/stage)")],
    ["T implied", "solve Arrhenius for v = 0.518", "21.3 °C"],
    ["Jensen (sd 1.2 °C)", "½ (Eₐ/RT²)² σ_T²", "< 1% effect"],
  ], 0.5, 1.4, [1.9, 4.6, 2.5], { fontSize: 11, rowH: 0.38, align: "left" });
  notes(s, ["Appendix A5. All inputs from the paper's Table 4 and storage description; computations are mine."]);
}
{ // A6
  const s = appx("A6", "Brownian Bridges: Replay and Crossings", "One conditional Gaussian regenerates W and gives a smooth crossing probability.");
  label(s, "virtual Brownian tree: recursive bisection from a seed", 0.5, 1.38, 9);
  img(s, "v2_vbt.png", 0.5, 1.64, { h: 0.5 });
  label(s, "crossing probability within a step (σ_q = local vol of q, frozen)", 0.5, 2.35, 9);
  img(s, "v2_bridge.png", 0.5, 2.61, { h: 0.55 });
  txt(s, [M("P(τ ≤ t*) ≈ 1 − Π_k (1 − p_k), p_k = 1 if an endpoint ≥ b.  σ_q² = ‖∇q(Z_k)ᵀ σ_θ(Z_k)‖².  Drift cancels after conditioning on endpoints.")], 0.5, 3.35, 9, 0.6, { fontSize: 11.5 });
  notes(s, ["Appendix A6. Standard Brownian-bridge (continuity) correction from barrier-option pricing; removes discrete-monitoring bias for the frozen-coefficient interpolant; solver weak error remains."]);
}
{ // A7
  const s = appx("A7", "SDE Solvers", "Match the solver to the calculus you wrote and the order to the quantity you need.");
  table(s, [
    ["Solver", "Converges to", "Strong order", "Weak order", "Notes"],
    ["Euler–Maruyama", "Itô", "½ (1 additive)", "1", "default"],
    ["Milstein", "Itô / Strat.", "1", "1", "needs ∂σ; commutative noise if d > 1"],
    ["Heun", "Stratonovich", "½ (general)", "—", "predictor–corrector"],
    ["Reversible Heun", "Stratonovich", "½ (general)", "—", "algebraically reversible"],
    ["SRA1 / ShARK", "Stratonovich", "1.5 (additive only)", "—", "state-independent σ"],
  ], 0.5, 1.4, [1.7, 1.4, 1.9, 1.1, 2.9], { fontSize: 11, rowH: 0.42, align: "left" });
  credit(s, "Kloeden & Platen (1992); Kidger et al. (2021); Diffrax docs.", 0.5, 4.1, 8);
  notes(s, ["Appendix A7. '—' = not claimed here."]);
}
{ // A8
  const s = appx("A8", "Benchmark Setup (Slide 16)", "Trend robust; absolute numbers depend on hardware and size.");
  table(s, [
    ["Item", "Setting"],
    ["Model", "16-dim state; drift + diagonal-diffusion MLPs, width 64, depth 2 (12.7k params)"],
    ["Batch", "32 paths as one 512-dim SDE, diagonal noise"],
    ["Solver", "reversible Heun; virtual Brownian tree tol 1/4096; float64; 2-core CPU"],
    ["Memory", "XLA compiled temp buffers"],
    ["Adjoint memory scaling", "256 → 1,024 channels: 62 → 438 MB; 4k → 42k params: 95 → 389 MB"],
    ["Method's own state", "2d + p ≈ 110 KB → rest is implementation constant"],
  ], 0.5, 1.4, [2.2, 6.8], { fontSize: 11, rowH: 0.4, align: "left" });
  notes(s, ["Appendix A8. Scripts: experiments/adjoint_tradeoffs.py, adjoint_memory_scaling.py."]);
}
{ // A9
  const s = appx("A9", "References and Figure Credits", null);
  const L = [
    "Xavier, Rodrigues & Silva. Shelf-life management and ripening assessment of 'Hass' avocado using deep learning. Foods 13(8):1150, 2024. doi:10.3390/foods13081150. CC BY 3.0 (Figs. 1, 2, 6a, 11; Tables 3, 4, 6).",
    "'Hass' Avocado Ripening Photographic Dataset. Mendeley Data, 2024. doi:10.17632/3xd9n945v8.",
    "Chen et al. Neural ODEs. NeurIPS 2018.",
    "Li, Wong, Chen, Duvenaud. Scalable Gradients for SDEs. AISTATS 2020.",
    "Kidger, Foster, Li, Lyons. Efficient and Accurate Gradients for Neural SDEs. NeurIPS 2021.",
    "Kidger. On Neural Differential Equations. PhD thesis, 2022.",
  ];
  const Rr = [
    "Mohamed et al. Monte Carlo Gradient Estimation in ML. JMLR 2020.",
    "Gholami, Keutzer, Biros. ANODE. IJCAI 2019.",
    "Finlay et al. How to Train Your Neural ODE. ICML 2020.",
    "Song et al. Score-Based Generative Modeling through SDEs. ICLR 2021.",
    "Kloeden & Platen. Numerical Solution of SDEs. 1992.",
    "Särkkä & Solin. Applied SDEs. CUP 2019.",
    "Griewank & Walther. Revolve. ACM TOMS 2000.",
    "Rubanova et al. 2019; Kidger et al. 2020 (CDEs); Price et al. 2025; Chi et al. 2023; Black et al. 2024; Gierjatowicz et al. 2022.",
  ];
  const runs = (a) => a.map((t, i) => ({ text: t, options: { breakLine: i < a.length - 1, paraSpaceAfter: 5 } }));
  txt(s, runs(L), 0.5, 1.38, 4.4, 3.9, { fontSize: 9.5 });
  txt(s, runs(Rr), 5.1, 1.38, 4.4, 3.9, { fontSize: 9.5 });
  notes(s, ["Appendix A9."]);
}

(async () => {
  const out = process.env.OUT || "../neural_sde_lecture.pptx";
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
