# Theory notes

*Statements P1–P4 of `docs/research_plan.md`, written out. Notation: visits at `t_0 < … < t_n`, latent
`Z = (R, U)`, progression `R` with prior `dR = κ μ(R) dt + σ(R) dW`, boundary `b`, event `τ = inf{t : R_t ≥ b}`,
physiological time `s = κ t`, survival `S̃(s | r) = P(no crossing within physiological time s | R(0) = r)`.
Working notes for the write-up; verifier pass done 2026-10-20 (`docs/verifier_report.md`).*

## P1 — Exactness of the Markov-interval likelihood

**Setting.** Under the prior, `R` is a time-homogeneous one-dimensional diffusion with locally Lipschitz,
bounded drift `κ μ` (we bound `μ ∈ (0, μ_max)` and `σ ∈ [σ_min, σ_max]`), so it is a strong Markov process and
`τ` is a stopping time. The censoring observation is the event `E = {τ ∈ (t_{m}, t_{m+1}]}` for an observed
spoilage between visits `m` and `m+1`, or `E = {τ > t_n}` for a right-censored item. Conditionally on the
latents at the visits `r_k = R(t_k)`:

**Claim.** `P(E | r_0, …, r_m) = Π_{k<m} B_k · [1 − S̃(κ Δ_m | r_m)]` for spoilage in `(t_m, t_{m+1}]` and
`Π_{k<n} B_k` for right censoring, where `Δ_k = t_{k+1} − t_k` and `B_k = P(no crossing in (t_k, t_{k+1}] |
r_k, r_{k+1})` is the **bridge** no-crossing probability — the probability that the diffusion pinned at both
ends stays below `b`.

**Proof sketch.** Write `A_k = {no crossing in (t_k, t_{k+1}]}`. By the Markov property at the deterministic
times `t_k`, conditionally on the skeleton `(r_0, …, r_m)` the segments of the path on disjoint intervals are
independent, so `P(∩_{k<m} A_k | r_0..r_m) = Π_{k<m} P(A_k | r_k, r_{k+1}) = Π B_k`. (Conditioning on `r_k`
alone would give `S̃(κΔ_k | r_k)`, which is *not* the right factor: the far endpoint is observed.) The last
interval's far end is unobserved, so its factor is the marginal `P(τ ∈ (t_m, t_{m+1}] | r_m) = 1 −
S̃(κΔ_m | r_m)`, with `S̃` in physiological time under the change of clock `s = κt` that turns the calendar
generator `κμ ∂_r + ½κσ² ∂_rr` into `μ ∂_r + ½σ² ∂_rr`. ∎

**Bridge factor.** For constant coefficients the bridge of a diffusion with drift is the Brownian bridge, so
`B_k = 1 − exp(−2 (b − r_k)(b − r_{k+1}) / (σ² κ Δ_k))` exactly; with state-dependent `σ` the code uses
`σ(r_k)` (first order in `Δ_k`). This replaced the earlier `S̃(κΔ_k | r_k)` product on 2026-10-20 after the
verifier showed the latter pulls pre-anchor latents away from the boundary (gradient −4.6 at `r = 4.5` vs 0
for the stage term).

**Remark on the clock (resolved 2026-10-20).** The clock multiplies the *generator*: in physiological time the
prior is `dR = μ(R) ds + σ(R) dW_s`, i.e. in calendar time `dR = κ μ(R) dt + √κ σ(R) dW_t`. The implementation
now uses `√κ σ` as the calendar diffusion in the SDE solve and the Girsanov KL (`LatentDynamics._solve_path`),
the synthetic simulator scales its noise the same way (`SyntheticConfig.noise_scaling="generator"`), and the
PDE table is in physiological time — so one table serves every `κ`. The alternative ("clock on the drift
only") would need one PDE solve per distinct `κ`; it is kept only as the simulator's `noise_scaling="drift"`.

**What P1 buys.** The likelihood needs no simulation of the prior beyond `t_now`, no smoothing of an
indicator, and is exact for any `μ, σ` — the only approximation is the PDE discretisation (P4).

## P2 — Properness of the interval-censored log-likelihood

**Setting.** Let the forecast be a distribution `F` over `τ − t_now` and the observation be `(L, R]` with
`L < R ≤ ∞` (right-censored ⇔ `R = ∞`). Assume *independent monitoring*: the visit schedule (hence `L, R`) is
independent of `τ` given the covariates — true by design for the avocado study (fixed daily schedule) and for
the synthetic visits.

**Claim.** `ℓ(F; L, R) = −log[F(R) − F(L)]` is a strictly proper scoring rule for the law of `τ − t_now`
under independent monitoring: its expectation under the true law `G` is minimised by `F = G` only.

**Why.** Conditional on the schedule, the observation is a random *interval* containing `τ`, and
`−log P_F(τ ∈ (L, R])` is the log score of the induced categorical forecast over the schedule's cells, which
is strictly proper for each schedule; integrating over the independent schedule preserves properness. This is
the argument of Yanagisawa & Akiyama (ICML 2026) for Case-k interval censoring; Rindt et al. (AISTATS 2022)
show that Survival-CRPS, IBS and the time-dependent C-index are *not* proper under censoring, which is why
the README's CRPS column is reported for comparability only and the model-selection metric is `nll`.

**Consequence.** Training on the ELBO's event term and reporting `nll` is consistent: the proper score is both
the loss and the metric, and censored items contribute a non-degenerate term.

## P3 — Identifiability of the clock

**Setting.** Groups `g` with rates `κ_g`, shared `μ, σ`, thresholds `θ_1 < … < θ_{K−1}`, stage observations at
visits. The map `(μ, σ, κ, θ) → law of the stage trajectories` is invariant under the joint rescaling
`r → a r + c`, `θ → a θ + c`, `μ → a μ`, `σ → a σ` (a > 0) and under `κ_g → λ κ_g`, `μ → μ/λ`, `σ → σ/√λ`.

**Claim.** Fix the thresholds (`fixed_thresholds=True`) and fix the scale of `μ` (e.g. `μ ≡ 1` for the linear
drift, or one group's rate). Then the ratios `κ_g / κ_{g'}` are identified from the mean stage progression
rates, and with the Arrhenius form `κ(T) = κ_ref exp(−E_a/R (1/T − 1/T_ref))` two distinct temperatures
identify `E_a`.

**Why.** With thresholds fixed the latent is in stage units; the expected progression `E[R_t − R_0]` is
`κ_g ∫ E[μ(R_s)] ds`, so for the linear drift `κ_g` is the slope of the mean stage vs time (the regression the
original paper fits, Table 4), and the ratio of two groups' slopes is `κ_g/κ_{g'}` regardless of `μ`'s scale.
`E_a = R ln(κ_1/κ_2) / (1/T_2 − 1/T_1)`. The synthetic recovery test (`tests/test_recovery.py`) is the
empirical check: rates within 25 %, `σ` within 50 % with 180 fake fruit.

**Caveat — the common factor is confounded with the origin.** The bottom stage is the open set `r < θ_1`,
so the data do not fix where the latent starts; a lower start plus a faster clock reproduces the same
transition times. Consequently only the *ratios* `κ_g / κ_{g'}` (hence `E_a`), `σ` in stage units, and the
forecast CDF are identified; the absolute rate scale is pinned only by the `z₀` prior (learnable mean/variance
in `LatentDynamics`, or an informative fixed prior in stage units). Verified empirically: in 19 synthetic runs
the ratios were within 10 % while the common factor wandered by +15–35 % (`docs/verifier_report.md`). The
recovery tests therefore assert ratios, `σ`, the CDF and `E_a`, not absolute rates. With `μ` an MLP the
same statement holds with `μ`'s scale absorbed into the factor. In the avocado data the ambient group's rate exceeds the 20 °C group's, so no monotone `κ(T)` fits
all three; the paper reports the Arrhenius clock on {T10, T20} + transfer, and the per-group clock as the
unconstrained comparator.

## P4 — The finite-difference solver

**Scheme.** Grid `r_i = r_min + i h`, `i = 0..N−1`, `r_{N−1} = b`; `S_{N−1} = 0` (absorbing), `S_0 = S_1`
(reflecting). First derivative by central differences where the cell Péclet number allows an M-matrix
(`|μ| h ≤ σ²`), upwind elsewhere; centred second derivative. Time stepping: two implicit-Euler steps
(Rannacher start-up, damping the jump in the initial condition at the boundary), then Crank–Nicolson
`(I − ½ ds L) S^{k+1} = (I + ½ ds L) S^k`. The operator is constant in `s`, so both step matrices are formed
once; the result is clipped to `[0, 1]` and made non-increasing in `s` by a running minimum.

**Properties.** (i) The implicit-Euler steps are monotone for any `ds` (`I − ds L` is an M-matrix); CN is
monotone for `ds ≤ 2 h²/σ²_max` and otherwise damped by the Rannacher start, with monotonicity enforced
exactly by the running minimum — `test_table_shape_and_boundaries` checks `0 ≤ S^{k+1} ≤ S^k ≤ 1`.
(ii) Consistency: `O(ds²) + O(h²)` in the central region, `O(h)` where upwinding is active. (iii) Convergence
to the viscosity solution follows from stability + consistency + monotonicity (Barles–Souganidis) for the
implicit-Euler variant; CN inherits it where it is monotone. (iv) For constant `μ, σ` the solution is the
inverse-Gaussian survival: with `N = 160, h ≈ 0.038, ds = 0.02` the maximum absolute error is 0.0011, and
0.0005 at `N = 240, ds = 0.01` (`test_matches_inverse_gaussian`, `test_grid_refinement`).

**Gradient.** `∂S/∂θ` flows through `jnp.linalg.inv(I − ds L)` and the grid (the boundary `b` is a learnable
threshold); `test_gradient_matches_finite_difference` checks it against central differences.

## P5 — Exact inference for the progression coordinate (the grid filter)

**Setting.** Because the event depends on the scalar `R` only and `R` is an autonomous 1-D diffusion under
the prior, the marginal likelihood of everything observed about `R` — stage labels, soft stage emissions from
photos, and the censoring observation — can be computed **exactly** (up to grid discretisation) without a
variational posterior: carry the density of `R` on the grid, propagate it between visits with the forward
Kolmogorov operator `exp(κΔ_k Lᵀ)` (the transpose of the discrete generator; the absorbing node becomes a
cemetery that collects killed mass), multiply by the visit's emission, and accumulate the normalisers
`Z_k`. This is the forward algorithm of a continuous-state HMM with a diffusion transition kernel; the
emission from a photo is the mixture `Σ_k q_k(x) P(stage = k | r)` with `q(x)` a learned soft stage
prediction (the hybrid DNN–HMM construction).

**Claim.** `log p(y_{1:n}, E | θ) = Σ_k log Z_k + log(tail)`, where the tail is the killed mass in `(lo, hi]`
(uncensored) or the surviving mass at `lo` (right-censored) propagated from the last visit's filtered density;
the censoring observations at visits (`known alive` → zero cemetery, `known spoiled` → cemetery only) are
emissions like any other. The forecast from the last visit is `1 − Σ_i p_i S̃(κh | r_i)` with the filtered
live density `p` and the survival table of P4.

**Why it matters.** The latent-SDE ELBO identified only the rate ratios (P3): the amortised posterior left a
gap on the common factor. The filter has no such gap: on the same synthetic data it recovers the absolute
rates within 20 %, `σ` within 40 % and the fresh-fruit CDF within 0.15 (`tests/test_recovery.py::
test_exact_filter_recovers_rates_sigma_and_fpt`), with the start prior `z₀ ~ N(m, s²)` learnable. Validated by
an independent verifier against closed-form (image-method) likelihoods and particle-filter Monte Carlo on
hand-built cases and simulated items: agreement within 0.004–0.07 nats at `ds = 0.02` (`docs/verifier_report.md`
§P5); the likelihood profile is minimised at the true `(κ, σ)`; recovered rates 0.235 / 0.475 / 0.523 vs
0.228 / 0.473 / 0.518.

**Cost.** One `N × N` propagator per batch plus `log₂(K)` matrix–vector products per visit interval (binary
decomposition of the step count, with linear interpolation to the fractional step so the likelihood is
differentiable in the clock). `N = 160`, `ds = 0.02`, `K = 2¹³` steps (164 physiological units; beyond that the
propagation is clipped and its gradient vanishes — keep `κ_max · longest gap / ds` below it).

**Assumption behind the photo emission.** Multiplying the label emission and the photo emission assumes the
label and the photo are conditionally independent given the latent; when raters labelled the same photo that
fails and the photo term double-counts — `image_emission: unlabelled_only` uses the photo only where no label
exists. An emission floor (`emission_floor`, default 1e-4) absorbs label noise and keeps the density from
collapsing to zero.

**Where the latent SDE still matters.** Appearance dimensions `U` and the Gaussian feature decoder are
multi-dimensional; the ELBO (`LatentDynamics`) remains the estimator for that generative model and for
unstructured latents, and the Monte Carlo running-max is the ablation baseline.

## 5. Open items for the verifier

1. ~~Clock/diffusion scaling~~ — resolved (clock on the generator everywhere);
   `test_pde_and_mc_predictions_agree` compares the PDE forecast with simulated paths at `κ = 0.4`.
2. P2: cite the exact theorem number from Yanagisawa & Akiyama once the camera-ready is available; state the
   independent-monitoring assumption in the paper's assumptions box.
3. ~~P3 two-temperature `E_a` recovery~~ — `tests/test_recovery.py::test_arrhenius_clock_recovers_activation_energy`.
4. ~~P4 grid refinement~~ — `tests/test_fpt_pde.py::test_grid_refinement`.
5. ~~Verifier pass on P1 / KL~~ — done 2026-10-20 (`docs/verifier_report.md`): clock scaling, ELBO assembly,
   leakage and metrics confirmed; P1 corrected to bridge factors; origin confound added to P3.
6. ~~Variational gap on the common rate factor~~ — closed by the exact grid filter (P5, `ripen.models.fpt_filter`).
7. ~~Verifier pass on P5~~ — done 2026-10-20 (`docs/verifier_report.md` §P5): operator, gradients, emissions,
   forecasts and brute-force agreement confirmed; the first-visit-spoiled case and the step ceiling were fixed.
