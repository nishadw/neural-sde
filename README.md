# Neural SDE First-Passage Models

[![tests](https://github.com/nishadw/neural-sde-first-passage/actions/workflows/tests.yml/badge.svg)](https://github.com/nishadw/neural-sde-first-passage/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-1F2328)
![jax](https://img.shields.io/badge/JAX-Diffrax%20%7C%20Equinox-C8102E)
[![license](https://img.shields.io/badge/license-MIT-6B7280)](LICENSE)

**Learn a latent Itô diffusion from irregular, censored observations, and predict a calibrated distribution
over *when* it first hits a boundary. The likelihood is exact: it comes from the Kolmogorov PDEs, not from
path simulation.**

The running example is fruit spoilage from daily photos. The same code runs on liver-disease survival (PBC2)
and jet-engine failure (NASA C-MAPSS). Built in JAX / Equinox / Diffrax.

## Methods

| Piece | Stochastic-calculus tool |
|---|---|
| Latent dynamics `dR = κ(x) μ_θ(R) dt + √κ(x) σ_θ(R) dW` | Neural Itô SDE; the condition `x` (temperature) enters only as a **time change** `s = κ(x) t` (Arrhenius `κ`, learned activation energy) |
| Event `τ = inf{t : R_t ≥ b}` | First-passage time of a learned diffusion |
| Forecast `P(τ − t > h)` | **Backward Kolmogorov PDE** `∂_s S = μ ∂_r S + ½σ² ∂_rr S`, Crank–Nicolson with upwinding; one table serves every condition and horizon |
| Filtering between visits | **Forward (Fokker–Planck) equation** on a grid: an exact continuous-state HMM (`fpt_filter`) |
| Interval-censored likelihood | Markov property + **Brownian-bridge** no-crossing factors `1 − exp(−2(b−r_k)(b−r_{k+1})/(σ²κΔ_k))`; exact under the model ([proof](docs/theory.md)) |
| Multi-dimensional latents | Latent-SDE ELBO with the **Girsanov** KL `½∫‖(f_post − f_prior)/σ‖² dt` (`latent_sde`) |
| Gradients through the solver | Backprop, recursive checkpointing, or the **stochastic adjoint** (one config switch) |

The PDE solver matches the inverse-Gaussian first-passage law to 1e-3. The derivations (exactness, properness
of the censored log score, identifiability of the clock) are in [`docs/theory.md`](docs/theory.md).

## Results

From a single training run on the UMIACS GPU cluster. The full write-up with figures is
[`report/REPORT.md`](report/REPORT.md).

| | Result |
|---|---|
| **Transfer to an unseen storage temperature** | The Arrhenius clock cuts MAE **1.7–2.9×** versus ablated clocks and beats every black-box baseline (0.9 / 1.7 / 4.7 days on T20 / ambient / T10) |
| **SDE vs ODE calibration** | Same ranking (C ≈ 0.93), but 80% intervals cover **93%** for the SDE vs 63% for the deterministic latent ODE: the diffusion term buys calibration |
| **Sparse observations** | With a photo every 3rd day the latent SDE has the best CRPS (0.81 vs 2.10 for GRU-D) and keeps 0.96 coverage |
| **vs the published baseline** | Every distributional model beats classify-then-regress (CRPS 1.82 → 0.69–1.05) |

Still open: the SDE models are near chance on PBC2, where the time scale is months rather than days. The
exact likelihood loses to Monte Carlo on synthetic data. And every number comes from a single seed.

## SDE foundations (`src/nsde`)

- **Gradient methods benchmark.** Backprop is exact but its memory grows with step count. Checkpointing is
  exact at ~1 MB and within 1.5× of backprop's speed. The stochastic adjoint uses constant memory but is 64×
  slower, and its error falls like 1/steps.
  ![Adjoint trade-offs](assets/adjoint_tradeoffs.png)
- **Neural local-volatility calibration.** Fits a neural local-vol SDE to a Heston implied-vol smile by
  differentiating a Monte Carlo pricer end to end. RMSE is 0.26 vol points on held-out strikes, vs 4.2 for the
  best flat Black–Scholes vol.
  ![Local-vol calibration](assets/local_vol_calibration.png)

## Run it

```bash
pip install -e ".[dev]"
make test                      # CPU tests, incl. parameter recovery on a known SDE
make smoke                     # every model: build, loss, gradient, predict
bash run_umiacs.sh pipeline    # full experiment pipeline on SLURM (--local to run without it)
```

Each experiment is one Hydra command, e.g.
`python scripts/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=T10`.

## Layout

```
src/ripen/   data loaders, models (latent SDE/ODE, fpt_pde, clocks, 7 baselines), trainer, metrics
src/nsde/    SDE solver with gradient-method switch; Black–Scholes / Heston pricers
configs/     Hydra configs        scripts/  train, evaluate, summarize, figures
report/      results report       docs/     theory, literature, data notes
lecture/     graduate lecture on neural SDEs (slides, notes, demo)
```

## Data

The avocado photos come from Mendeley ([10.17632/3xd9n945v8.1](https://doi.org/10.17632/3xd9n945v8.1), CC BY 4.0):
478 fruit, 14,710 photos. The other datasets are A2SL microwave scans, PBC2 and NASA C-MAPSS. Raw data is not
committed; see [`data/README.md`](data/README.md).

## Lecture

*Neural SDEs: Learning Random Dynamics.* A graduate lecture for UMD CMSC 838B (Fall 2026). It covers neural
ODEs, the adjoint method, Brownian motion and Itô's lemma, Euler–Maruyama, pathwise SDE training and latent
SDEs. Slides and demo are in [`lecture/`](lecture/).

## Key references

Li et al., *Scalable Gradients for SDEs*, AISTATS 2020 · Kidger et al., *Efficient and Accurate Gradients for
Neural SDEs*, NeurIPS 2021 · Rubanova et al., *Latent ODEs*, NeurIPS 2019 · Rindt et al., *Survival regression
with proper scoring rules*, AISTATS 2022. The full list is in [`docs/literature_review.md`](docs/literature_review.md).
