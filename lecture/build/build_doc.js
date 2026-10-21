const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ExternalHyperlink, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, AlignmentType, Footer, PageNumber,
} = require("docx");

const INK = "1F2328", RED = "C8102E", MUTED = "6B7280", CARD = "F3F4F6";

// ---------- helpers ----------
const t = (text, o = {}) => new TextRun({ text, ...o });
const b = (text, o = {}) => new TextRun({ text, bold: true, ...o });
const link = (text, url) => new ExternalHyperlink({ link: url, children: [new TextRun({ text, style: "Hyperlink" })] });
const P = (children, o = {}) => new Paragraph({ children: Array.isArray(children) ? children : [t(children)], spacing: { after: 120 }, ...o });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [t(text)] });
const bullet = (children, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: Array.isArray(children) ? children : [t(children)], spacing: { after: 60 } });
const num = (children, ref = "nums") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: Array.isArray(children) ? children : [t(children)], spacing: { after: 80 } });
const note = (text) => new Paragraph({ children: [t(text, { color: MUTED, italics: true, size: 20 })], spacing: { after: 160 } });

// Reading entry: title line + "why / what to read" line
function entry(title, url, meta, focus) {
  return [
    new Paragraph({ numbering: { reference: "reads", level: 0 }, spacing: { after: 20 }, keepNext: true, keepLines: true,
      children: [link(title, url), t("  ·  " + meta, { color: MUTED })] }),
    new Paragraph({ indent: { left: 720 }, spacing: { after: 120 }, children: [t(focus, { size: 21 })] }),
  ];
}

// table helper (US Letter, 1" margins => 9360 DXA content width)
const border = { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, header, rows) {
  const cell = (text, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: head ? { fill: INK, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: Array.isArray(text) ? text : [t(text, head ? { bold: true, color: "FFFFFF" } : {})] })],
  });
  return new Table({
    width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }))],
  });
}

// ---------- content ----------
const kids = [];

kids.push(new Paragraph({ spacing: { after: 60 }, children: [t("CMSC 838B  ·  STUDENT LECTURE  ·  FALL 2026", { bold: true, color: RED, size: 18 })] }));
kids.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [t("Neural SDEs: Reading and Prep Guide")] }));
kids.push(P([t("Lecture: ", { color: MUTED }), b("Backprop Through Continuous Time: Neural SDEs: Modeling, Gradients, and Training"), t("   ·   Running example: 'Hass' avocado ripening (Xavier, Rodrigues & Silva, Foods 2024)   ·   Nishad Wajge", { color: MUTED })]));
kids.push(P([t("Part 1 is the reading list for the class, ready to post with the slides. Part 2 is your prep plan, ordered to match the deck. Part 3 explains every equation on the slides in plain words. Part 4 is the questions you should be able to answer cold. Part 5 covers what's deliberately left out.")]));
kids.push(P([t("Deck layout: 22 presented slides plus appendix A1–A10 (derivations, solver table, benchmark setup, evaluation protocol, credits), which you don't present but should be able to open if asked. The running example uses the paper's own figures and tables (CC BY 3.0); every number derived from them (rates, Eₐ, first-passage ratios) is worked out in A4 and A5.")]));

// ---- Course requirements
kids.push(H1("Course requirements for the lecture"));
kids.push(P([t("From the course's Class Presentation page and its posting instructions. Grading: "), b("advance preparation 40%"), t(", "), b("analysis and discussion of the materials 40%"), t(", "), b("style and clarity 20%"), t(". Length isn't stated; last year's schedule fit two or three talks per 75-minute class. Each student lecture gets 25 minutes. The deck is timed to exactly that: 20:00 of talk, then 5:00 of discussion, hard stop at 25:00. If you fall behind, skip slides 7 and 19 (about 70 seconds).")]));
kids.push(table([3000, 2200, 4160], ["Step", "When", "Notes"], [
  [[b("Topic + reading list to Prof. Lin")], "Was due Oct 9", "If not already sent, send Part 1 now."],
  ["1:1 meeting with Prof. Lin", "Before the talk", "She goes through the materials with you in detail."],
  ["Draft slides + initial treatment", "1 week before your slot", "Send the .pptx, placeholders and all; the running example uses published data only."],
  ["Hand out readings not online", "1 lecture before", "Not needed: every reading below is online."],
  ["Post slides, readings, discussion questions", "24 hours before", "Slides as PowerPoint, readings as PDF, in a public_html subfolder. Email her the exact location."],
]));
kids.push(new Paragraph({ spacing: { after: 120 }, children: [] }));

// ---- Part 1
kids.push(H1("Part 1. Reading list for the class"));
kids.push(note("Post this with the slides. Estimated load: about 2 hours for the required pair."));
kids.push(H2("Required"));
kids.push(...entry("Neural Ordinary Differential Equations", "https://arxiv.org/abs/1806.07366", "Chen, Rubanova, Bettencourt, Duvenaud · NeurIPS 2018",
  "Read Sections 1–3: the ResNet-to-ODE idea and the adjoint method. Skim the rest."));
kids.push(...entry("Scalable Gradients for Stochastic Differential Equations", "https://arxiv.org/abs/2001.01328", "Li, Wong, Chen, Duvenaud · AISTATS 2020",
  "Read the stochastic adjoint, the virtual Brownian tree, and the latent SDE (ELBO via Girsanov). These are slides 15 and 17."));
kids.push(H2("Running example (short, skim)"));
kids.push(...entry("Shelf-Life Management and Ripening Assessment of 'Hass' Avocado Using Deep Learning", "https://doi.org/10.3390/foods13081150", "Xavier, Rodrigues & Silva · Foods 13(8):1150, 2024 · open access, CC BY",
  "About 20 minutes. Read the abstract, Section 2 (storage groups, ripening index, the Days Left regression) and look at Figs. 1, 2, 6 and 11 and Tables 3, 4 and 6. Slides 2–6, 12 and 13 use these directly."));
kids.push(H2("Optional"));
kids.push(...entry("Monte Carlo Gradient Estimation in Machine Learning", "https://arxiv.org/abs/1906.10652", "Mohamed, Rosca, Figurnov, Mnih · JMLR 2020",
  "Sections on pathwise and score-function estimators. Background for slide 14."));
kids.push(...entry("Score-Based Generative Modeling through Stochastic Differential Equations", "https://arxiv.org/abs/2011.13456", "Song et al. · ICLR 2021",
  "The SDE framework and the probability-flow ODE. Slide 18."));
kids.push(...entry("Efficient and Accurate Gradients for Neural SDEs", "https://arxiv.org/abs/2105.13493", "Kidger, Foster, Li, Lyons · NeurIPS 2021",
  "Reversible Heun and the Brownian Interval."));
kids.push(...entry("'Hass' Avocado Ripening Photographic Dataset", "https://data.mendeley.com/datasets/3xd9n945v8/1", "Mendeley Data · 2024 · doi:10.17632/3xd9n945v8",
  "The images behind the paper, for anyone who wants to run the model. Check the license on the page before reusing images."));
kids.push(H2("Discussion questions"));
kids.push(num("Shelf life is a first-passage time. Which estimator would you train P(τ ≤ t*) with: smooth surrogate, Brownian-bridge crossing, or score function? Why?", "dq"));
kids.push(num("Slide 13 suggests the noise partly scales with temperature. How would you test that properly on this data?", "dq"));
kids.push(num("Which differentiable simulator from this course would you make stochastic, and what would happen to its gradients?", "dq"));

// ---- Part 2
kids.push(H1("Part 2. Your prep plan"));
kids.push(H2("If you only have a week: the fast path (~9 hours)"));
kids.push(note("Do these in order. Each one covers specific slides; the full plan below goes deeper."));
kids.push(table([3700, 1000, 1500, 3160], ["Read or watch", "Time", "Slides", "Why"], [
  [[link("Xavier et al. 2024, Foods (avocado paper)", "https://doi.org/10.3390/foods13081150")], "20 min", "2–6, 12, 13", "Every number in the running example comes from Tables 4 and 6."],
  [[link("Video: Duvenaud, Latent SDEs", "https://www.microsoft.com/en-us/research/?p=750409")], "1 hr", "15, 17", "The stochastic adjoint and latent SDE, from the author."],
  [[link("Chen et al. 2018, Neural ODEs", "https://arxiv.org/abs/1806.07366")], "1 hr", "7–9", "Sections 1–3 and the adjoint appendix (A1)."],
  [[link("Higham 2001, SDE simulation", "https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P42.pdf")], "1 hr", "10, A7", "Itô, Euler–Maruyama, strong vs. weak order, with code."],
  [[link("Li et al. 2020, Scalable Gradients for SDEs", "https://arxiv.org/abs/2001.01328")], "1.5 hr", "15, 17, A3", "Class reading: adjoint, Brownian tree, ELBO."],
  [[link("Mohamed et al. 2020, MC gradients", "https://arxiv.org/abs/1906.10652")], "45 min", "14", "Pathwise vs. score function; why pathwise fails on steps."],
  [[link("Video: Yang Song, score-based models", "https://cbmm.mit.edu/video/diffusion-and-score-based-generative-models")], "1 hr", "18", "Diffusion models as SDEs."],
  [[link("Kidger thesis, neural SDE chapters", "https://arxiv.org/abs/2202.02435")], "1.5 hr", "15, 16, 20", "Reversible Heun, exact adjoints, practical advice."],
  [[b("Rehearse 3 full runs with a timer")], "1.5 hr", "all", "Use the practice script; hit the checkpoints."],
]));
kids.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
kids.push(H2("Rehearsal checkpoints"));
kids.push(bullet("Slide 9 by 6:00, slide 14 by 12:15, slide 20 by 18:30, discussion by 20:00."));
kids.push(bullet("You can't see notes while presenting: each slide's bottom line is your cue for its main point."));
kids.push(bullet("Behind by more than a minute at slide 14? Skip slide 19 and shorten slide 16 to its takeaway."));
kids.push(H2("Full plan"));
kids.push(P([t("About 23 hours if you have two weeks, in deck order. Where a stage lists a video, watch it before the paper.")]));
const stage = (title, hrs, slides) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [t(title), t(`   ${hrs}  ·  slides ${slides}`, { color: MUTED, bold: false, size: 20 })] });

kids.push(stage("Stage 1. Neural ODEs and the adjoint", "~4 hr", "7–9, A1"));
kids.push(...entry("Deep Implicit Layers: Neural ODE chapter, with JAX code", "https://implicit-layers-tutorial.org/neural_odes/", "Kolter, Duvenaud, Johnson · NeurIPS 2020 tutorial",
  "Best hands-on intro to the adjoint for people who know autodiff."));
kids[kids.length - 1] = new Paragraph({ indent: { left: 720 }, spacing: { after: 120 }, children: [
  t("Best hands-on intro to the adjoint for people who know autodiff. ", { size: 21 }), link("Tutorial video", "https://neurips.cc/virtual/2020/tutorial/16653"), t(".", { size: 21 })] });
kids.push(...entry("Video: Neural Ordinary Differential Equations", "https://av.fields.utoronto.ca/talks/Neural-Ordinary-Differential-Equations", "David Duvenaud · Fields Institute",
  "The author on the paper, including what didn't work."));
kids.push(...entry("Neural Ordinary Differential Equations", "https://arxiv.org/abs/1806.07366", "Chen et al. · NeurIPS 2018", "Class reading. Read the adjoint appendix too: it's the derivation in appendix A1."));
kids.push(...entry("Latent ODEs for Irregularly-Sampled Time Series", "https://arxiv.org/abs/1907.03907", "Rubanova, Chen, Duvenaud · NeurIPS 2019", "The deterministic version of the avocado model (the Latent ODE column on slide 6)."));
kids.push(...entry("ANODE", "https://arxiv.org/abs/1902.10298", "Gholami, Keutzer, Biros · IJCAI 2019", "When the adjoint gives wrong gradients (slide 20, first row)."));

kids.push(stage("Stage 2. Stochastic calculus and the avocado model", "~5.5 hr", "10–13, A2, A4, A5, A7"));
kids.push(...entry("An Algorithmic Introduction to Numerical Simulation of SDEs", "https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P42.pdf", "Higham · SIAM Review 2001",
  "Start here. Euler–Maruyama, strong vs. weak convergence, Itô vs. Stratonovich, Milstein, with runnable code. Covers slide 10 and the solver table in A7."));
kids.push(...entry("Applied Stochastic Differential Equations (free PDF)", "https://users.aalto.fi/~ssarkka/pub/sde_book.pdf", "Särkkä & Solin · Cambridge 2019",
  "Read the Itô calculus chapter (Itô's lemma, Itô vs. Stratonovich) and the numerical-simulation chapter; look up Girsanov's theorem for A3."));
kids.push(...entry("Inverse Gaussian distribution", "https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution", "Reference page", "The first-passage time of drifted Brownian motion. Know mean L/v and variance Lσ²/v³: slide 13 and A4 are built on them."));
kids.push(...entry("Arrhenius equation and Q10", "https://en.wikipedia.org/wiki/Arrhenius_equation", "Reference page", "Enough to explain how two storage temperatures pin the activation energy (slide 12, A5)."));

kids.push(stage("Stage 3. Gradients through randomness", "~5 hr", "14–16, A6, A8"));
kids.push(...entry("Monte Carlo Gradient Estimation in Machine Learning", "https://arxiv.org/abs/1906.10652", "Mohamed et al. · JMLR 2020",
  "Pathwise vs. score function, and why pathwise fails on discontinuous objectives. Slide 14."));
kids.push(...entry("Video: Latent Stochastic Differential Equations: An Unexplored Model Class", "https://www.microsoft.com/en-us/research/?p=750409", "David Duvenaud · Microsoft Research",
  "Walks through the stochastic adjoint and latent SDEs. Watch before Li et al."));
kids.push(...entry("Scalable Gradients for SDEs", "https://arxiv.org/abs/2001.01328", "Li et al. · AISTATS 2020", "Class reading. Slides 15 and 17 are built on it."));
kids.push(...entry("Efficient and Accurate Gradients for Neural SDEs", "https://arxiv.org/abs/2105.13493", "Kidger et al. · NeurIPS 2021", "The exact reversible adjoint and the Brownian Interval (slide 15)."));
kids.push(...entry("On Neural Differential Equations (PhD thesis)", "https://arxiv.org/abs/2202.02435", "Kidger · Oxford 2022",
  "Best single reference. Read the neural SDE chapter and the chapter on numerical solutions and backpropagation."));

kids.push(stage("Stage 4. Training and diffusion", "~4 hr", "17–18, A3"));
kids.push(...entry("Neural SDEs as Infinite-Dimensional GANs", "https://arxiv.org/abs/2102.03657", "Kidger et al. · ICML 2021", "Skim. The alternative to the ELBO, in case someone asks how else to train an SDE."));
kids.push(...entry("Video: Diffusion and Score-Based Generative Models", "https://cbmm.mit.edu/video/diffusion-and-score-based-generative-models", "Yang Song · MIT CBMM", "The SDE view of diffusion from its author."));
kids.push(...entry("Score-Based Generative Modeling through SDEs", "https://arxiv.org/abs/2011.13456", "Song et al. · ICLR 2021", "Forward SDE, reverse SDE, probability-flow ODE, denoising score matching."));
kids.push(...entry("Introduction to Flow Matching and Diffusion Models (lecture notes)", "https://diffusion.csail.mit.edu/2026/", "MIT 6.S184 · Holderrieth & Erives", "Builds diffusion and flow matching from ODEs and SDEs."));

kids.push(stage("Stage 5. Applications", "~1.5 hr", "19"));
kids.push(P([t("Abstract and main figure of each; one sentence per table row, plus one follow-up answer.")]));
const apps = [
  ["GenCast: Diffusion-based ensemble forecasting for medium-range weather", "https://arxiv.org/abs/2312.15796", "Price et al. · Nature 2025"],
  ["Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", "https://arxiv.org/abs/2303.04137", "Chi et al. · RSS 2023"],
  ["π0: A Vision-Language-Action Flow Model for General Robot Control", "https://arxiv.org/abs/2410.24164", "Black et al. · 2024"],
  ["Robust pricing and hedging via neural SDEs", "https://arxiv.org/abs/2007.04154", "Gierjatowicz et al. · J. Computational Finance 2022"],
  ["Arbitrage-free neural-SDE market models", "https://arxiv.org/abs/2105.11053", "Cohen, Reisinger, Wang · 2021"],
  ["Neural Controlled Differential Equations for Irregular Time Series", "https://arxiv.org/abs/2005.08926", "Kidger et al. · NeurIPS 2020"],
];
apps.forEach(([a, u, m]) => kids.push(bullet([link(a, u), t("  ·  " + m, { color: MUTED })])));

kids.push(stage("Stage 6. Hands-on", "~2 hr", "16, A8"));
kids.push(bullet([t("Run the "), link("Diffrax SDE example", "https://docs.kidger.site/diffrax/examples/sde_example/"), t(".")]));
kids.push(bullet([t("Run "), b("lecture/neural_sde_demo.py"), t(" and "), b("experiments/adjoint_tradeoffs.py --quick"), t(" from the "), link("neural-sde repo", "https://github.com/nishadw/neural-sde"), t(". These produce the numbers on slide 16.")]));
kids.push(bullet([t("Recompute slide 12 and 13 yourself from Table 4 and Table 6 of the paper (10 lines of Python). If you can reproduce 2.07, 50.4 kJ/mol, 2.99 and 2.33, you can defend them.")]));
kids.push(bullet([t("Gotcha: parameters must go through "), b("args="), t(", not a closure, or BacksolveAdjoint can't differentiate them.")]));

// ---- Part 3: every equation in plain words
kids.push(H1("Part 3. Every equation on the slides, in plain words"));
kids.push(note("If you can say each of these out loud without looking, you can handle the slides."));
const eqs = [
  ["Slide 4 · Days Left = α_g(s − 5)  ⇒  ds/dt ≈ v_g = 1/|α_g|", "The paper's per-group regression says each stage takes |α| days, so stage advances at about 1/|α| stages per day: a constant-drift ODE with no noise. 0.228 /day at 10 °C, 0.473 at 20 °C, 0.518 ambient. The inversion is approximate (slopes differ by a factor R² ≈ 0.96; stage is ordinal)."],
  ["Slide 5 · quantization floor |α|/4", "A stage lasts |α| days and one label can't say where inside it the fruit is. If the position is uniform over a window of width w, the best point guess is off by w/4 on average. That floor is 1.10, 0.53 and 0.48 days, i.e. 65%, 72% and 91% of the observed error."],
  ["Slide 7 · dh/dt = f_θ(h, t)", "A network gives the velocity of the hidden state; a solver integrates it. A ResNet block h + f(h) is one Euler step of this with step size 1. The paper's model is the special case f = v_g, a constant."],
  ["Slide 8 · memory table", "Backprop stores every step (O(N)). Binomial checkpointing stores O(log N) states and recomputes the rest (O(N log N) work). The adjoint stores nothing from the forward pass (O(1) in N) and solves a backward system of size 2d + p: state, adjoint, parameter gradient."],
  ["Slide 9 · ȧ = −aᵀ∂f/∂z, ȧ_θ = −aᵀ∂f/∂θ", "a(t) = ∂L/∂z(t), the sensitivity of the loss to the state at time t. Integrate backward from a(T) = ∂L/∂z(T); a_θ starts at 0 at time T and its value at time 0 is dL/dθ. Each term is a vector-Jacobian product: one reverse-mode call per solver stage. Derivation in A1."],
  ["Slide 10 · Z_t = Z_0 + ∫μ ds + ∫σ dW", "An SDE is shorthand for this integral equation. The second integral is Itô: evaluate σ at the left end of each little interval."],
  ["Slide 10 · W_t − W_s ~ N(0, (t−s)I), (dW)² = dt", "Brownian increments are independent Gaussians with variance equal to elapsed time. (dW)² = dt is why second-order terms survive."],
  ["Slide 10 · Itô's lemma", "df(Z) = f′ dZ + ½ f″ σ² dt. The extra term comes from (dW)² = dt. Example: d log S picks up −½σ²."],
  ["Slide 10 · μ_Strat = μ_Itô − ½ Σ_j (∂σ_j/∂z) σ_j", "Same process written with midpoint (Stratonovich) integrals needs a different drift. Zero correction when σ doesn't depend on Z."],
  ["Slide 10 · solver orders", "Euler–Maruyama: strong ½ (path error ~ √Δt), weak 1 (error in expectations ~ Δt). Milstein adds a correction for strong order 1. Reversible Heun is Stratonovich and algebraically reversible, which is what makes an exact adjoint possible. Full table in A7."],
  ["Slide 11 · dZ = κ(c) μ_θ dt + σ_θ dW", "Latent ripeness evolves with a learned rate scaled by a temperature factor, plus learned noise for fruit-to-fruit and day-to-day variability."],
  ["Slide 11 · κ(c) = exp[−(E_a/R)(1/T_c − 1/T_ref)]", "Arrhenius: reaction rates rise exponentially with temperature. E_a = activation energy, R = 8.314 J/(mol·K), T in kelvin. κ = 1 at the reference temperature."],
  ["Slide 11 · p(y ≤ k | Z_t) = sigmoid(c_k − q(Z_t))", "Ordinal (cumulative-logit) head. q(Z) is a scalar ripeness score; the cut points c_1 < … < c_4 split it into stages 1–5. Higher q, later stage. The cut c_4 is the shelf-life barrier b."],
  ["Slide 11 · τ = inf{t : q(Z_t) ≥ b}", "Shelf life is the first time the ripeness score crosses into stage 5. We want P(τ ≤ t* | images so far, storage)."],
  ["Slide 12 · E_a = R ln(v₂₀/v₁₀) / (1/T₁₀ − 1/T₂₀)", "Two temperatures pin the activation energy exactly: Q₁₀ = v₂₀/v₁₀ = 2.07 gives E_a ≈ 50.4 kJ/mol. Predicting ambient (18.7 °C): 0.431 /day, 2.32 days per stage. Observed 1.93: ambient ripens ~20% faster than temperature explains (would need 21.3 °C). Worked in A5."],
  ["Slide 13 · E τ = L/v, sd(τ) = √(Lσ²/v³)", "First-passage time of drifted Brownian motion to a barrier at distance L is inverse Gaussian. Faster drift → earlier and less spread-out crossing."],
  ["Slide 13 · ratio formulas", "Same distance L in two groups. If σ is the same everywhere, sd ratio = (v₂₀/v₁₀)^{3/2} = 2.99. If temperature speeds up the whole clock (σ² ∝ v), ratio = v₂₀/v₁₀ = 2.07. Stage quantization alone also gives 2.07. Observed MAE ratio: 2.33, so there is extra spread at 10 °C beyond quantization, but three numbers can't pin down σ(T)."],
  ["Slide 14 · pathwise vs score function", "Pathwise: write Z_T as a function of θ and fixed noise, differentiate inside the expectation (needs f differentiable). Score function: E[f · ∇ log p_θ], works for any f, high variance. For 1{τ ≤ t*} pathwise gives exactly zero."],
  ["Slide 15 · stochastic adjoint", "Same as slide 9 plus a term aᵀ ∂σ_j/∂z ∘ dW^j per noise channel, driven backward by the same Brownian path. μ here is the Stratonovich drift; convert an Itô model first (A2)."],
  ["Slide 17 · prior and posterior SDEs", "Prior: dynamics that know only the storage condition (used to forecast). Posterior: same σ, but the drift also sees the images (used to filter during training)."],
  ["Slide 17 · ELBO with ∫½‖u‖² dt, σu = h_φ − h_θ", "Expected log-likelihood of observed stage labels minus two KLs: one for the starting state, one for the whole path. Girsanov makes the path KL the integral of ½‖u‖², where u is the drift gap measured in units of σ. Add it as an extra state and integrate. Derivation in A3."],
  ["Slide 18 · forward, reverse, probability-flow, score matching", "Forward: fixed noising SDE. Reverse: same SDE run backward needs the score ∇ log p_t, learned as s_θ. Probability-flow ODE: deterministic, same marginals if s_θ is exact. Training regresses s_θ onto the closed-form score of p(x_t | x_0): no solver needed."],
  ["A9 · CRPS(F, y) = ∫(F(t) − 1{y ≤ t})² dt", "Scores a whole predictive distribution of τ against the true day y. Lower is better, and it's a proper scoring rule. For a point forecast it equals the absolute error, so the paper's baseline CRPS is its MAE (1.04 days overall)."],
  ["A6 · Brownian bridge", "Given W at two times, W in between is Gaussian: mean on the straight line, variance (s−u)(v−s)/(v−u). Used to regenerate W on demand and to compute the chance a path crossed b between steps: exp(−2(b−q_k)(b−q_{k+1})/(σ_q²Δt))."],
];
eqs.forEach(([q, a]) => {
  kids.push(new Paragraph({ spacing: { before: 100, after: 30 }, keepNext: true, children: [b(q)] }));
  kids.push(new Paragraph({ spacing: { after: 60 }, children: [t(a)] }));
});

// ---- Part 4: questions
kids.push(H1("Part 4. Questions to be ready for"));
const qa = [
  ["Is the adjoint gradient exact?", "No. It re-solves the trajectory backward, so it inherits solver error and can drift on stiff or chaotic dynamics (ANODE). Reversible Heun with its matching discrete adjoint (Kidger 2021, torchsde) is exact up to floating-point round-off. Diffrax's BacksolveAdjoint is the continuous adjoint, approximate even with reversible Heun: in the repo it was off by 7e-5 at 16 steps, 2e-6 at 512."],
  ["Why did the adjoint use 154 MB if it's 'constant memory'?", "Constant in the number of steps, yes. The method needs only 2d + p ≈ 110 KB; the rest is an implementation constant in Diffrax/XLA that grows with parameters and noise channels. Checkpointing was ~1 MB and exact."],
  ["Strong vs. weak order?", "Strong: error of individual paths. Weak: error in expectations. Probabilities like P(τ ≤ t*) need weak accuracy; pathwise gradients care about strong."],
  ["Why Stratonovich for the adjoint?", "It obeys the ordinary chain rule, so the ODE adjoint carries over term by term. Itô models are converted first with the drift correction; using the Itô drift directly gives wrong gradients."],
  ["Why is the pathwise gradient of P(spoiled) zero?", "For fixed noise, 'spoiled by t*' is a step function of θ: derivative zero almost everywhere. The expectation is smooth, so you can't swap ∇ and E. Fixes: sigmoid surrogate (biased), Brownian-bridge crossing probability (smooth per step), or score function (unbiased, noisy)."],
  ["Can you use the score function for parameters in σ?", "Not in continuous time: changing σ changes quadratic variation, so path measures become mutually singular. On a discrete grid the Euler–Maruyama density exists, but the estimator's variance blows up as Δt → 0."],
  ["Why must prior and posterior share σ?", "Girsanov only changes drift. Different σσᵀ means different quadratic variation, mutually singular path measures, KL = ∞."],
  ["Why Arrhenius?", "Two controlled temperatures (10 °C, 20 °C) can't train a free-form temperature dependence; Arrhenius is a one-parameter physical prior (slide 12: E_a ≈ 50 kJ/mol). Applying it only to drift is a choice; scaling diffusion by √κ is the alternative if temperature speeds up the whole clock."],
  ["Are diffusion models trained like your latent SDE?", "No. Diffusion training is simulation-free (denoising score matching, closed-form p(x_t | x_0)); the solver only runs at sampling. Our model learns the dynamics, so it must backprop through the solve."],
  ["How is the latent SDE's posterior conditioned on photos?", "Encode each photo, run a GRU or Transformer backward over the sequence to make a context vector per time, feed it to the posterior drift h_φ."],
  ["What's a virtual Brownian tree?", "Regenerate W(s) on demand by recursive Brownian-bridge bisection from a fixed seed: O(1) memory, O(log 1/ε) per query, same answer every time."],
  ["Identifiability?", "Reparameterizing the latent space leaves the likelihood unchanged, and fast drift + small noise can mimic slow drift + big noise under sparse observations. The ordinal head anchors the ripeness direction q; test by recovering a known SDE on synthetic data."],
  ["Isn't the paper's regression already good (R² ≈ 0.96)?", "R² measures the fit of the mean days-left per stage. The decision (sell, discount, discard by Friday) needs the spread around the mean, and Fig. 6's error bars are ±2–3 days at 10 °C."],
  ["Why is ambient 20% faster than Arrhenius predicts?", "Not temperature fluctuation: with sd 1.2 °C the Jensen effect is under 1%. Likely uncontrolled humidity (not logged for Tamb) and other covariates. That's the argument for drift = κ(c) × learned residual rather than pure Arrhenius."],
  ["Is two temperatures enough to fit Arrhenius?", "It determines E_a exactly but can't test the functional form, so I use it as a prior, not a law."],
  ["Doesn't slide 13 prove there's diffusion?", "No. Quantization alone predicts 2.07; observed is 2.33. That excess is consistent with extra randomness at 10 °C, but it's three numbers, MAE stands in for sd, and labels are human. It's a sanity check on the shape of the model."],
  ["Could a latent ODE produce spread too?", "Yes, via an uncertain initial state or per-fruit rate, and that spread grows linearly with horizon. But it collapses as images accumulate. The SDE adds noise that keeps arriving (sd ∝ √h). Fig. 6's per-stage sds could distinguish the two."],
  ["Is the paper's baseline fair to compare against?", "Mostly. Caveats: the paper doesn't say whether Table 6 is test-only or whether α was fit on all fruit, and 'by sample' keeps the better of two sides. When reproducing, refit α on training fruit only."],
  ["Why not a sequence model with a hazard head?", "It's a legitimate baseline (slide 6 column). The SDE gives irregular times, uncertainty over futures and a physics-structured rate by construction."],
];
qa.forEach(([q, a]) => {
  kids.push(new Paragraph({ spacing: { before: 120, after: 40 }, keepNext: true, children: [b(q)] }));
  kids.push(new Paragraph({ spacing: { after: 80 }, children: [t(a)] }));
});

// ---- Part 5: what is left out
kids.push(H1("Part 5. What's not in the talk"));
kids.push(note("The talk uses only the paper's published data. Nothing on the slides depends on new experiments."));
kids.push(bullet([b("No results slide: "), t("the evaluation plan moved to appendix A9 (baseline numbers from the paper, how a latent SDE would be scored). Open it only if asked.")]));
kids.push(bullet([b("If asked whether you ran the model: "), t("say no; this lecture analyzes the paper's data with the neural-SDE lens, and your semester project is where you'll fit it.")]));
kids.push(bullet([b("Figures: "), t("the paper's figures are CC BY 3.0 and credited on each slide (cropped panels say so). The originals are in your Downloads folder.")]));

// ---------- document ----------
const doc = new Document({
  creator: "Nishad Wajge",
  title: "Neural SDEs: Reading and Prep Guide",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { font: "Arial", size: 44, bold: true, color: INK }, paragraph: { spacing: { after: 160 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Arial", size: 30, bold: true, color: INK }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: "Arial", size: 24, bold: true, color: RED }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "dq", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "reads", levels: [{ level: 0, format: LevelFormat.BULLET, text: "▪", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } }, run: { color: RED } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [t("Neural SDEs lecture · prep guide · ", { color: MUTED, size: 18 }), new TextRun({ children: [PageNumber.CURRENT], color: MUTED, size: 18 })] })] }) },
    children: kids,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(process.env.OUT || "../neural_sde_reading_guide.docx", buf); console.log("doc written"); });
