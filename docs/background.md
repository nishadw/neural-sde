# Background: everything you need to read the rest of `docs/`

*A primer, written for the author of this repo. Each section ends with where the idea lives in the code.
Read in order: this file → `research_plan.md` → `theory.md` → `literature_review.md` / `venue_scan.md` →
`data_*.md` → `experiment_plan.md`.*

## 1. The problem, stated carefully

An **item** (one avocado, one patient, one engine) is observed at **visits** `t_0 < t_1 < … < t_n`. At each
visit we get some measurement (a photo, a microwave scan, blood labs, sensor readings) and sometimes a
label (a ripeness stage). At some unknown time `τ` the item crosses into an absorbing state (spoiled,
diseased, failed). We never see `τ`; we see at most that it happened **between two visits** — "unspoiled on
Tuesday, spoiled on Thursday" means `τ ∈ (Tue, Thu]`. That is **interval censoring**. If the study ends
before the item spoils, we only know `τ > t_n`: **right censoring**. The task: at any visit, output a
distribution over `τ − t_now`, not a number. The forecast must be *calibrated* (when it says 80 % it should be
right 80 % of the time) and *sharp* (intervals as narrow as the data allow).

Why a distribution: two fruit that look identical today can spoil days apart. A point estimate hides this; a
retailer acting on "1-in-5 chance it is bad by Tuesday" can price by risk.

→ `src/ripen/data/schema.py` (`Item`, `Visit`, the interval `(spoil_lo, spoil_hi]`).

## 2. Survival analysis in five definitions

- **Survival function** `S(t) = P(τ > t)`; **CDF** `F = 1 − S`; **hazard** `h(t) = −d/dt log S(t)` (instantaneous
  risk given survival so far).
- **Right-censored likelihood**: an observed event at `T` contributes the density `f(T)`; a censored item
  contributes `S(C)`. **Interval-censored likelihood**: an event in `(L, R]` contributes `F(R) − F(L)`.
- **Proper scoring rule**: a loss `ℓ(F, y)` whose expectation under the truth `G` is minimised only by `F = G`.
  The log score `−log P_F(observation)` is proper. **Survival-CRPS and the integrated Brier score are not
  proper under censoring** (Rindt et al., AISTATS 2022), so a model can "win" on them by being wrong. We
  report them for comparability but select models on the censored log score (`nll`).
- **Calibration**: for each horizon `h`, among items with predicted `P(τ ≤ h) ≈ p`, a fraction `≈ p` should have
  spoiled by `h`. Reliability diagram; expected calibration error (ECE).
- **Concordance (C-index)**: of all comparable pairs, the fraction where the item predicted to spoil sooner did.
  Discrimination only; a model can be concordant and badly calibrated.

→ `src/ripen/eval/metrics.py`.

## 3. Stochastic differential equations in five definitions

- An **SDE** `dZ = μ(Z) dt + σ(Z) dW` describes a state that drifts at rate `μ` and is kicked by Brownian
  motion `W` with size `σ`. Euler–Maruyama simulates it: `Z_{k+1} = Z_k + μ Δt + σ √Δt ε`.
- A **neural SDE** makes `μ` and `σ` neural networks. A **latent** neural SDE puts the SDE on a hidden state
  and decodes observations from it; it is a VAE in continuous time.
- **Training** (Li et al. 2020): a *posterior* SDE with an extra drift `g_φ(Z, context)` that knows the data
  generates paths; the loss is the ELBO: log-likelihood of the observations under the path, minus the
  KL between posterior and prior path laws, which by **Girsanov's theorem** is `½ ∫ |(f_post − f_prior)/σ|² dt`.
- **Gradients through the solver**: backprop through every step (exact, memory grows with steps), recursive
  checkpointing (exact, log memory), or the stochastic adjoint (constant memory, approximate). The repo's
  `nsde` benchmark measured these; checkpointing is the default.
- **First passage**: `τ = inf{t : R_t ≥ b}` for a scalar coordinate `R` and boundary `b`. For constant drift
  and diffusion `τ` is **inverse-Gaussian**; for general `μ, σ` there is no closed form — but the survival
  function `S(s, r)` solves the **backward Kolmogorov equation**
  `∂_s S = μ(r) ∂_r S + ½ σ(r)² ∂_rr S` with `S(s, b) = 0`, which a finite-difference solver handles exactly.

→ `src/nsde/solve.py`, `src/ripen/models/latent.py`, `src/ripen/models/fpt_pde.py`.

## 4. Physiological time and the Arrhenius clock

Chemical rates grow with temperature as `k(T) = A exp(−E_a / (R T))` (Arrhenius). Food scientists express
shelf life as a budget of *physiological time* `s = ∫ κ(T(t)) dt` consumed at a temperature-dependent rate
(Taoukis & Labuza 1989; Van Boekel 2008). We put that clock on the SDE's generator: in physiological time
every item follows the same `dR = μ(R) ds + σ(R) dW_s`; in calendar time the drift is `κ μ` and the diffusion
`√κ σ`. Two consequences: (1) one survival table in `s` serves every temperature — `P(τ − t > h) = S(κ h | r)`;
(2) `κ` is identifiable from how fast the stages advance, and the Arrhenius form lets the model extrapolate
to a temperature it never saw (the leave-one-group-out experiment). The published avocado rates
(0.228 / 0.473 stages per day at 10 / 20 °C) imply `E_a ≈ 50 kJ/mol`, `Q10 ≈ 2.1`; the ambient group breaks the
pattern (faster than 20 °C), which is why a per-group clock is kept as the unconstrained comparator.

→ `src/ripen/models/clock.py`; `tests/test_recovery.py::test_arrhenius_clock_recovers_activation_energy`.

## 5. The model in one picture

```
photos / scans / labs ──► frozen encoder ──► visit features ──► GRU (posterior context) ──► q(z₀)
                                                                                          │
   storage T ──► κ(T) ──────────────────────────────────────────────────────────────────► latent SDE  Z = (R, U)
                                                                                          │
                                           ┌──────────────── ordinal head: stage ~ Ordinal(R) ◄──┤
                                           │                                                     │
                      survival table S(s | r) from the backward Kolmogorov PDE  ◄── (μ_R, σ_R) ──┤
                                           │                                                     │
   P(spoiled by day h | history) = 1 − E_q[ S(κ h | R_now) ]      feature decoder (optional) ◄────┘
```

Likelihood of what we observe: stages at visits (ordinal), features at visits (Gaussian, optional), and the
censoring observation — for an event in `(t_m, t_{m+1}]`, the bridge no-crossing factors between observed
latents times `[1 − S(κ Δ_m | r_m)]` for the last, unobserved-end interval (exact by the Markov property;
`docs/theory.md` P1). Minus the Girsanov KL. That is the whole loss.

## 6. Why this is new (short)

Deep survival models learn hazards from static or sequential covariates; neural-SDE papers learn dynamics but
never derive an event time from them; the one "first-hitting-time" survival paper (DeepFHT, Phys Rev E 2026)
uses closed-form Brownian motion with a network only on its parameters; interval censoring reached ML venues
in 2025–26 for tabular data only. Nobody has combined a *learned nonlinear* latent diffusion, an *exact*
first-passage likelihood under *interval censoring*, and a *physical clock* that transfers across
conditions — on image or sensor sequences, with calibration reported. See `venue_scan.md` §Synthesis.

## 7. The datasets, in one line each

- **Mendeley avocado** (478 fruit, daily photos, three temperatures, stage labels) — the labelled benchmark.
- **A2SL / Stevens microwave avocados** (15 fruit, every other day, photos + 75×75 microwave reconstructions;
  labels are day-determined per the published paper) — the multimodal case study.
- **PBC2** (312 patients, irregular clinic visits over years, death/transplant) — the standard dynamic-survival
  benchmark.
- **C-MAPSS** (100–260 engines per subset, run to failure, inspections every ~10 cycles) — degradation to an
  absorbing failure, the industrial analogue of spoilage.
- **Synthetic** — ripening from a known SDE at the published rates; the ground truth for every unit test.

## 8. Glossary

ELBO · evidence lower bound. Girsanov · the theorem giving the KL between two diffusions with the same
diffusion coefficient. Markov-interval likelihood · the product of inter-visit survival probabilities.
Monotone scheme · a finite-difference scheme that preserves ordering (no spurious oscillation). Péclet number ·
drift × grid step / diffusion; decides central vs upwind differencing. Proper score · see §2. Q10 · rate ratio
per 10 °C. Rannacher start-up · two implicit-Euler steps before Crank–Nicolson to damp a discontinuous initial
condition.
