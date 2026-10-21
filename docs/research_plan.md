# Research plan: Neural first-passage processes

*The play, chosen 2026-10-20 after the venue scan in `docs/venue_scan.md` and the review in
`docs/literature_review.md`. Read this before anything else in `docs/`.*

## 1. The one-paragraph pitch

Many things are followed until they stop being usable: a fruit until it spoils, a patient until a disease
declares itself, a battery until it fails. They are observed on visits, never continuously, so the event is
only known to have happened between two visits (interval-censored) or after the last one (right-censored);
and the speed at which they progress depends on a condition — temperature, load, dose — that we would like to
extrapolate. We model the hidden state as a **neural SDE whose clock is a known physical function of the
condition**, define the event as the **first passage** of a learned progression coordinate across a boundary,
and derive the event-time distribution **exactly** from the learned dynamics by solving the backward
Kolmogorov equation once per batch in *physiological time* — no path simulation, no smoothed indicator. The
resulting interval-censored likelihood is a strictly proper score, the clock is identifiable, and forecasting
from any visit is a table look-up. One implementation, `ripen`, runs on produce photos, microwave scans and
clinical records.

## 2. Why this clears the bar (from the venue scan)

| What gets accepted at NeurIPS / ICML / ICLR / AISTATS in this area | What we bring |
|---|---|
| A new loss / scoring rule with a properness proof (SuMo-net AISTATS 22, Yanagisawa ICML 23, SurvivalBoost AISTATS 25, Yanagisawa–Akiyama ICML 26) | The interval-censored first-passage likelihood of a learned diffusion, shown proper under independent monitoring; exact, not Monte Carlo |
| A model class that removes an assumption and carries an identifiability / rate result (DeepEH NeurIPS 21, NFM NeurIPS 23, CDVI ICML 25) | First-passage of a *learned nonlinear* SDE (DeepFHT, Phys Rev E 2026, is closed-form Brownian motion with a network only on its parameters); identifiability of the physical clock; transfer to unseen conditions |
| Neural-SDE papers as training algorithms or constrained classes (Stable Neural SDEs ICLR 24, SDE Matching ICML 25, FDM ICLR 25) | A constrained class — "progression-structured" latent SDE: one autonomous 1-D progression coordinate plus appearance dimensions — for which the event distribution is a 1-D PDE |
| Calibration studied on tabular data; image-sequence survival never reports calibration | Calibration (CiPOT / CSD-style comparators, the ICML 26 interval-censored metric) on image sequences and multimodal sensor sequences |

Open at all five venues per the scan: (i) first passage of a learned latent SDE with interval censoring,
(ii) a physics-constrained time reparametrisation with a transfer test, (iii) calibrated distributional
outputs from image sequences. We do all three; (i) is the headline.

## 3. The model

Latent `Z = (R, U)`: `R` is the scalar **progression** coordinate, `U ∈ ℝ^{d−1}` carries appearance.

```
prior      dR = κ(x)·μ_R(R) ds + σ_R(R) dW_R                     (autonomous in R)
           dU = κ(x)·μ_U(R, U) ds + σ_U(R, U) dW_U
posterior  drift += g_φ(Z, ctx, t)   for t ≤ t_now              (Li et al. 2020 latent SDE)
clock      κ(x) = κ_ref · exp(−E_a/R_gas (1/T − 1/T_ref))       (Arrhenius; or a learned per-group rate)
stages     stage_k ~ Ordinal(R(t_k))        thresholds θ_1 < … < θ_{K−1}
features   y_k ~ N(dec(Z(t_k)), s² I)      (optional; lets unlabeled sequences train the dynamics)
event      τ = inf{ t : R(t) ≥ b },  b = θ_{K−1}
```

**Physiological time.** With `s = κ(x)·t`, the prior for `R` has no `κ`: `dR = μ_R(R) ds + σ_R(R) dW`.
The survival function `S̃(s | r) = P(no crossing of b within physiological time s | R(0) = r)` solves

```
∂_s S̃ = μ_R(r) ∂_r S̃ + ½ σ_R(r)² ∂_rr S̃,     S̃(0, r) = 1 (r < b),   S̃(s, b) = 0,   ∂_r S̃(s, r_min) = 0
```

once per parameter setting, on a grid in `r` and `s`. For item `i` at condition `x_i` and latent `r` at any
time, `P(τ − t > h) = S̃(κ(x_i)·h | r)`. The clock commutes with the PDE, so one solve serves every item in the
batch and every horizon.

**Likelihood of what we observe.** Visits at `t_0 < … < t_n`, posterior latent `r_k = R(t_k)`. By the strong
Markov property the crossing events on disjoint inter-visit intervals are conditionally independent given the
latents at the visits:

```
interval-censored (crossed in (t_{n−1}, t_n]):   Π_{k<n−1} B_k · [1 − S̃(κΔ_{n−1} | r_{n−1})]
right-censored (unspoiled at t_n):               Π_{k<n}   B_k
B_k = P(no crossing in (t_k, t_{k+1}] | r_k, r_{k+1}) = 1 − exp(−2(b − r_k)(b − r_{k+1}) / (σ² κ Δ_k))   (bridge factor)
```

This is exact under the model (Prop. 1), replaces the smoothed running-max indicator of the current code
(which has a biased gradient), and costs one PDE solve plus table look-ups. Training maximises the latent-SDE
ELBO with this term in the likelihood: stage log-lik + feature log-lik + first-passage log-lik − Girsanov KL −
KL(q(z₀)‖p(z₀)). The Monte Carlo running-max estimator stays as a baseline and as the fallback for latents
without the progression structure.

**Forecasting** from the latest visit is simulation-free: sample `r` from the posterior at `t_now`, read
`S̃(κh | r)` for `h` on the day grid, average.

## 4. Claims and what proves them

| # | Claim | Evidence |
|---|---|---|
| P1 | *Exactness.* Under the progression-structured model the product of bridge no-crossing factors times the last-interval survival difference equals the likelihood of the censoring observations. | Markov property of the prior diffusion; proof in `docs/theory.md`; verified numerically (`docs/verifier_report.md`) |
| P2 | *Properness.* Under independent monitoring (visit times ⊥ event given covariates) the interval-censored log-likelihood is a strictly proper scoring rule for the event distribution; Survival-CRPS is not (Rindt et al. 2022). | Follows Yanagisawa & Akiyama (ICML 2026); we state it for our likelihood and use it as both loss and metric |
| P3 | *Identifiability of the clock.* The rate *ratios* `κ(x)/κ(x')` — hence the Arrhenius `E_a` — and the forecast CDF are identified from the stage trajectories; the common factor is confounded with the latent origin and is pinned by the `z₀` prior. | Scale argument on `(μ_R, σ_R, κ, θ, z₀)`; `tests/test_recovery.py` (ratios within 15 %, `E_a` within 15 %, CDF within 0.15) |
| P4 | *Numerics.* The finite-difference scheme converges to `S̃` as the grid is refined, and for constant drift it matches the inverse-Gaussian law. | `tests/test_fpt_pde.py` |
| E1 | Beats the two published point-estimate models on the avocado data on MAE-PO and gives calibrated intervals; keeps the censored fruit they dropped. | Mendeley avocado, all 8 model families, one protocol |
| E2 | The exact PDE likelihood trains better than the smoothed Monte Carlo estimator (lower loss, lower gradient variance, faster). | Synthetic + avocado ablation |
| E3 | Transfer to an unseen temperature via the Arrhenius clock beats a per-group clock and the no-clock baselines on the held-out group. | Leave-one-storage-group-out |
| E4 | Calibration and sharpness degrade gracefully with sparser visits (every 2–3 days); photo count tightens intervals. | Subsampling and prefix ablations |
| E5 | Same model, other domain: PBC2 (longitudinal clinical) and/or C-MAPSS (degradation to failure, artificially inspected every k cycles). | Second-domain runs |
| E6 | Multimodal: photos + microwave grids (A2SL) through the feature decoder; qualitative latent trajectories. | A2SL avocado |

## 5. What is reusable by others (the "others can apply it" part)

`ripen` is organised so the first-passage machinery is independent of produce:

- `ripen.models.fpt_pde` — survival of a 1-D diffusion with learned drift/diffusion on a grid, differentiable,
  plus the interval-censored Markov-interval likelihood. Works with any encoder that outputs a latent.
- `ripen.data.schema.Item` — the manifest format (visits, features, interval); any longitudinal dataset with
  an absorbing event fits it (`scripts/export_manifest.py` shows the CSV).
- The clock module: Arrhenius, per-group, or any user function of covariates.
- The evaluation suite with proper scores and calibration, identical for every model.

## 6. Risks and the honest answers

- **A2SL data has no labels.** It demonstrates multimodality and the generative path, not the headline
  numbers. The headline numbers come from Mendeley (478 labelled fruit) and the second domain.
- **The 1-D progression restriction.** It is an assumption; we show it costs nothing on these data against
  the unrestricted MC model, and it is what makes the likelihood exact. Multi-dimensional boundaries fall back
  to the MC estimator.
- **Tamb faster than T20 in the avocado data.** A mean-temperature Arrhenius law cannot fit all three groups;
  the per-group clock is the comparator and the paper says so.
- **Compute.** PDE solve: a 160-point grid × ~3,000 explicit steps per batch, vectorised; cheaper than 64
  simulated paths per item.

## 7. Order of work from here

1. ✔ `fpt_pde.py` (Crank–Nicolson + Rannacher, hybrid differencing; 0.001 error vs inverse-Gaussian) + tests.
2. ✔ Markov-interval likelihood in `LatentDynamics` (`event_likelihood: pde | mc`), progression-structured prior,
   clock on the generator (`√κ σ` calendar diffusion), learnable ordinal link scale.
3. ✔ Proper metric `nll` (censored log score); MAE-PO still to add; CRPS/MAE kept for comparability.
4. ✔ PBC2 and C-MAPSS loaders, configs, tests (`docs/data_second_domain.md`).
5. ✔ `run_umiacs.sh`.
6. ✔ `docs/theory.md` (P1–P4), `docs/background.md`.
7. ⟳ Verifier passes (first pass running 2026-10-20); fix list to follow.
8. ☐ MAE-PO; CiPOT/CSD-style post-hoc calibration comparator; gradient-method sweep on the real data.
