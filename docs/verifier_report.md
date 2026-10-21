# Verifier report — PDE first-passage likelihood (2026-10-20)

*Independent review by a verifier agent of `fpt_pde.py`, `latent.py`, the metrics and `theory.md`, with its
own numerical experiments. Abridged; the numbers are the agent's. Fixes it prompted are marked ✔.*

## 1. Math

- **`survival_table` — OK.** Coefficients implement `μS' + ½σ²S''` with central/upwind switching at the M-matrix
  condition `|μ|h ≤ σ²`; absorbing and reflecting rows correct; Rannacher start-up then Crank–Nicolson.
  Max |PDE − inverse-Gaussian| = 0.00109 (N=160, ds=0.02), 0.00049 (240, 0.01), 0.00027 (320, 0.01).
- **`lookup` — OK** at all edges. Note: the table spans `pde_steps·ds` physiological units; longer queries take
  the last row (harmless where S ≈ 0). ✔ documented.
- **`markov_interval_log_lik`:** anchor selection, prefix handling and the last-interval term verified against
  hand products in four configurations. Two findings:
  - the inter-visit factor `S̃(κΔ_k | r_k)` was wrong in principle: with both endpoints observed the correct
    factor is the bridge no-crossing probability `1 − exp(−2(b−r_k)(b−r_{k+1})/(σ²κΔ_k))`; the old term pulled
    pre-anchor latents away from the boundary (gradient −4.6 at r = 4.5 vs 0 from the stage term; trained
    stage-4 latents 3.5–3.8 instead of ≈ 4.5). ✔ replaced by bridge factors; `theory.md` P1 rewritten.
  - a zero-width interval (first visit already spoiled) had constant likelihood and zero gradient.
    ✔ widened to `(lo, lo + ds/κ]`.
- **Clock scaling — OK and mutually consistent.** Monte Carlo paths of `dR = κμ dt + √κ σ dW` vs
  `1 − S̃(κt | r₀)`: max abs error 0.0026 (κ = 0.4), 0.0057 (κ = 2.0), at the MC standard error. With
  drift-only noise the discrepancy would be 0.10.

## 2. Statistics

- ELBO assembly correct (signs, sample averaging, masks, KL gated by `t ≤ t_now`, `kl_z0`). The MC branch
  uses `log(mean_M p)` — a looser marginal estimate, fine as a baseline but not the same objective.
- **No leakage**: perturbing every invisible slot leaves loss and `predict_cdf` unchanged to 1e-16; shifting
  `spoil_hi` leaves `predict_cdf` unchanged.
- `predict_cdf` (pde) = `P(τ − t_now ≤ h | history, no crossing before t_now)`. OK.

## 3. The recovery failure

19 training runs (180 synthetic items, 14 epochs). The over-estimate of the rates is a **common factor on all
groups**, present with both likelihoods, unaffected by gradient floors, 16 posterior samples, 3× finer Euler,
30 epochs, or the KL's √κ. **Rate ratios were within 10 % in every run** (T10/T20: −11…+7 %; Tamb/T20:
−5…+8 %). With a hard ordinal link the forecast CDF error is 0.075 while κ is +26 %: the fit is
self-consistent — the ordinal head's bottom stage is open below, `q(z₀)` settles at 0.5–0.9 instead of 1.2,
and a 16–20 % faster clock covers the longer distance `(5−0.6)/(5−1.2) ≈ 1.16`. Pinning the z₀ prior to the
truth restores the latents and σ (0.28) but the amortised GRU posterior still leaves a gap on the common
factor. Conclusion: the identified quantities are the ratios, `E_a`, `σ` and the forecast CDF.
✔ tests assert those; ✔ `theory.md` P3 states the origin confound; the learnable link scale ends at 0.35–0.39
on hard labels, so ✔ the synthetic tests fix it at 0.1.

## 4. Metrics

`cdf_at`, `censored_log_score`, `quantile_from_cdf`, `harrell_cindex`: OK. `reliability_curve` counted
uncensored instances with `lo < h < hi` as "not spoiled by h" — harmless with daily visits, biased for the
sparser-visit ablation. ✔ known-set changed to `(lo ≥ h) | (hi ≤ h)`. ✔ `INF_SENTINEL` guard added.

## 5. Tests

`pytest -m "not slow"`: 60 passed. The committed HEAD also passed the slow recovery test with rates
+18.7 / +4.7 / +18.5 % — the earlier 25 % absolute bound was marginal, not a guarantee.

## Open

- ~~An exact grid-filtering likelihood~~ — implemented the same day as `ripen.models.fpt_filter` (`theory.md`
  P5); it recovers the absolute rates within 20 % where the ELBO could not. Needs its own verifier pass.


---

# Verifier report — the exact grid filter (P5), 2026-10-20

*Second independent pass, on `src/ripen/models/fpt_filter.py`. Abridged; numbers are the agent's.*

1. **Forward operator — OK.** `L·1 = 0` exactly; `P = (I−½ds Lᵀ)⁻¹(I+½ds Lᵀ)` conserves mass (column sums off
   by ≤ 1.6e-5 at 160/0.02 after 11 squarings). Killed mass vs the backward table: 2.7e-3 (ds 0.02), 6.7e-4
   (0.01); vs the inverse-Gaussian truth: 8.9e-3 (160/0.02), 2.2e-3 (320/0.01) — grid error dominates.
   Transport exact to 4 d.p. (mean `μs + r₀`, variance `σ²s + var₀`). CN is formally non-monotone for
   `σ ≥ 0.4` at 160/0.02 but no node went negative on the densities that occur. ✔ comment added.
2. **Clock gradient — OK.** `propagate_time` continuous across integer steps (jump ≤ 8e-7); `jax.grad` of the
   loss w.r.t. the group log-rates matches central differences to 0.5 %. Caveat: steps clipped at `2^n_pow − 1`
   → zero gradient on gaps longer than 41 units at `n_pow = 11`. ✔ `n_pow` raised to 13 (164 units).
3. **Emission semantics — OK except (f).** Nine hand cases vs closed form and particle-filter MC agree within
   0.004–0.03 nats (unlabelled alive visit, spoiled visit visible or hidden, right-censored, prefix before `lo`).
   (f) **CONFIRMED-BUG** first visit already spoiled: `alive` and `dead` both true → emission ≡ 0 →
   `log 1e-30`. ✔ fixed: `alive = … & ~dead`, and a dead visit always gets one propagation step of killing;
   `tests/test_fpt_filter.py::test_zero_width_interval_and_first_visit_spoiled`.
4. **`predict_cdf` — OK.** No leakage (0.0 change when the interval is altered), monotone, fresh-item forecast
   within 0.002–0.006 of the z₀-mixture of inverse-Gaussian CDFs.
5. **Photo emission.** Mixture `q(x) @ e_stage` verified; multiplying with a label emission assumes label ⊥
   photo | latent. ✔ documented; `image_emission: unlabelled_only` option added. The mixture is not a normalised
   likelihood in `x` (fine for end-to-end training); the hybrid-HMM prior division is absorbed by training.
6. **Recovery test** passed in 13 s: rates 0.2345 / 0.4745 / 0.5226 (true 0.228 / 0.473 / 0.518), σ 0.193
   (true 0.25), z₀ 1.19 ± 0.155 (true 1.2 ± 0.15), fresh-CDF error 0.028. Brute-force gap at `ds = 0.02` is
   0.00–0.07 nats (the earlier "1 nat" figure came from an item with the tail bug).

Also found while running the real data end to end (not by the verifier): with monthly units (PBC2) the default
clock was ~20× too fast, the density was entirely absorbed before an "alive" visit, and the backward pass
produced NaN gradients. ✔ emission floor + collapse guard in the filter; ✔ `clock_kwargs.rate_ref: auto`
initialises the clock from the median event time (`DatasetInfo.median_event_time`).
