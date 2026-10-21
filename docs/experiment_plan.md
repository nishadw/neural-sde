# Experiment plan

Every experiment is one Hydra command; `run_umiacs.sh` strings them together with SLURM dependencies.
Everything before §2 runs on a laptop CPU. Results cells in the README stay TBD until the run exists.

Claims (see `docs/research_plan.md`, `docs/theory.md`):

| # | Claim | Experiment |
|---|---|---|
| P1 | Exact interval-censored first-passage likelihood (PDE) vs the smoothed Monte Carlo estimator | E2 |
| P2 | The censored log score is the proper metric; CRPS/MAE reported for comparability | all (metric `nll`) |
| P3 | The Arrhenius clock is identifiable and transfers to an unseen storage temperature | E3 |
| E1 | Beats the two published point-estimate models on the avocado data while keeping censored fruit | E1 |
| E4 | Calibration and sharpness vs visit spacing and photo count | E4 |
| E5 | Same model on PBC2 (clinical) and C-MAPSS (degradation) | E5 |
| E6 | Multimodal photos + microwave (A2SL) | E6 |

## 0. One-time setup

```bash
git clone https://github.com/nishadw/ripen && cd ripen
bash run_umiacs.sh setup                 # venv at ~/envs/ripen, editable install, CPU tests + smoke
```

Data (no new collection; public or provided only):

```bash
# Mendeley avocado set (CC BY 4.0): download https://data.mendeley.com/datasets/3xd9n945v8/1 and unzip into data/avocado
# A2SL microwave drops: place the three raw folders under data/a2sl/ as in docs/data_a2sl.md, then
python scripts/ingest_a2sl.py            # HEIC -> JPEG, manifests, scan caches (a few minutes)
bash run_umiacs.sh data                  # PBC2 + C-MAPSS downloads, split files, ResNet-18 feature caches
```

Expected: avocado 478 fruit (three groups, 9–27 % right-censored per group); A2SL 15 fruit (4 tracked for 7 visits);
PBC2 312 patients; C-MAPSS FD001 100 engines.

## 1. Milestone 1 — fake fruit (no real data)

```bash
make test                                # includes tests/test_recovery.py (rates, sigma, FPT CDF, Arrhenius E_a)
make synthetic-quick                     # results/synthetic/recovery.{json,png}
```

Pass criteria are in the tests: rates within 25 %, sigma within 50 %, max |CDF error| < 0.15, `E_a` within 15 %.

## 2. On UMIACS

```bash
export RIPEN_ACCOUNT=... RIPEN_PARTITION=... RIPEN_QOS=...     # from: sacctmgr show assoc user=$USER
bash run_umiacs.sh all                   # submits synthetic, E1–E6 and the summary job with dependencies
bash run_umiacs.sh e3                    # or any single stage
bash run_umiacs.sh all --local           # same sequence, no SLURM (a workstation)
```

What each stage runs:

| Stage | Jobs | Output |
|---|---|---|
| `synthetic` | `scripts/synthetic_recovery.py` at full scale | `results/synthetic/recovery.{json,png}` |
| `e1` | all 8 models on `data=avocado`; latent SDE labels-only variant | `results/runs/<model>_avocado/` with `metrics_{test,val}.json`, `metrics_by_n_obs.json`, reliability and pred-vs-true figures |
| `e2` | latent SDE with `model.event_likelihood=pde` vs `mc`, on avocado and synthetic | `results/runs/latent_sde_{avocado,synthetic}_{pde,mc}/` |
| `e3` | leave one storage group out × {direct_reg, gru_d, latent_sde}; latent SDE with `clock=group` and `clock=fixed` | `results/runs/*_logo_*` |
| `e4` | every-2/every-3-day training for gru_d / latent_ode / latent_sde; daily-trained models evaluated on sparser test visits | `results/runs/*_every{2,3}`, `metrics_test_every{2,3}.json` |
| `e5` | six models on `data=pbc2` (horizon 180 months) and `data=cmapss` (horizon 40 inspections) | `results/runs/<model>_{pbc2,cmapss}/` |
| `e6` | gru_d and latent SDE on `data=a2sl`; latent SDE with the feature decoder | `results/runs/*_a2sl*` |
| `summary` | `scripts/summarize.py` after everything | `results/summary.md` |

Metrics in every `metrics_*.json`: `nll` (proper censored log score, model selection), `mae`, `crps`,
`crps_censored`, `coverage80`, `width80`, `cindex`, `ece`; overall, per storage group, and fruit-weighted.

Targets to beat on avocado (test split, uncensored instances): Xavier et al. 1.04 d MAE by sample;
Hangud & Hangud 1.126 d fruit-weighted MAE.

## 3. Reading the results

- E1 table: `results/summary.md` → README results table.
- P1: compare `latent_sde_*_pde` vs `_mc` on `nll`, training time per epoch (`history.csv`), and the synthetic
  rate/sigma recovery.
- P3: `metrics_test.json` of the three LOGO folds for `latent_sde` (Arrhenius) vs `_clock_group` vs
  `_clock_fixed`; the fitted `E_a` and `Q10` are in the model (`model.clock.ea`, `model.clock.q10()`).
- E4: `metrics_by_n_obs.json` (photo count) and `metrics_test_every{2,3}.json` (visit spacing).

## 4. Manual follow-ups

- Gradient-method sweep (backprop / checkpoint / adjoint) for the latent SDE: `make gradients` or
  `slurm/gradient_methods.sh`; compare wall-clock, memory and `nll`.
- Re-evaluate any finished run: `python scripts/evaluate.py results/runs/<run> [--every k] [--n-obs 1 3 5]`.
