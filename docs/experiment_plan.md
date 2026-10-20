# Experiment plan

The exact commands, in order. Everything before §3 runs on a laptop CPU; §3 onward is written for UMIACS
but every command is also a plain `python scripts/...` call you can run anywhere.

Novelty claims locked on 2026-10-20 (see `docs/literature_review.md`, §6):

| # | Claim | Experiment |
|---|---|---|
| 1 | Interval-censored first-passage likelihood for a latent neural SDE, with censored fruit kept | E1 (all models), E4 |
| 2 | Shared ripening clock with an Arrhenius temperature factor; transfer to an unseen storage group | E3 |
| 3 | How accuracy and calibration degrade with sparser visits | E2 |
| 4 | How much each extra photo tightens the interval (single model via visible-prefix dropout) | E1 ablation |
| 5 | Next-photo timing | deferred |

## 0. One-time setup

```bash
git clone https://github.com/nishadw/ripen && cd ripen
python -m venv ~/envs/ripen && source ~/envs/ripen/bin/activate
pip install -e ".[dev,features]"          # JAX CPU wheel; on a GPU node: pip install -U "jax[cuda12]"
make test                                  # CPU suite incl. the fake-fruit recovery test (~2 min)
make smoke                                 # every model, one batch
```

Data (no new collection; public only):

```bash
mkdir -p data/avocado && cd data/avocado
# download https://data.mendeley.com/datasets/3xd9n945v8/1 (CC BY 4.0), unzip here, then:
cd ../.. && python -c "from ripen.data.avocado import load_avocado; it=load_avocado('data/avocado'); print(len(it), 'fruit')"
```

Expected: 478 fruit, three groups, 9–27 % right-censored per group.

## 1. Milestone 1 — fake fruit (no real data)

```bash
make synthetic-quick                 # ~1 min: results/synthetic/recovery.{json,png}
make synthetic                       # full: 600 items, 40 epochs
```

Pass criterion (also enforced by `tests/test_recovery.py`): per-group rates within 25 % of
0.228 / 0.473 / 0.518, sigma within 50 % of 0.25, max |CDF error| < 0.15 on fresh T20 fruit.

## 2. Milestone 2 — avocado data organised, baseline scored

```bash
make splits DATA=avocado             # results/splits/avocado_random_seed0.json + three LOGO files (committed)
make features                        # ResNet-18 features -> data/features/avocado_resnet18.npz (GPU: slurm/extract_features.sh)
python scripts/train.py model=stage_linear data=avocado
python scripts/train.py model=direct_reg data=avocado
```

Targets to beat (test split, uncensored instances): Xavier et al. 1.04 d MAE by sample;
Hangud & Hangud 1.126 d fruit-weighted MAE. Our numbers go in `results/summary.md` via `make summary`.

## 3. E1 — every model, one protocol (UMIACS)

```bash
source slurm/common.sh                              # edit RIPEN_ACCOUNT / RIPEN_PARTITION / RIPEN_QOS first
sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/train_all_models.sh avocado
# labels-only variant of the latent models (milestone "first model, using ripeness labels only"):
sbatch ... slurm/train_all_models.sh avocado +experiment=labels_only
```

Each run writes `results/runs/<model>_avocado/{metrics_test.json, metrics_val.json, metrics_by_n_obs.json,
reliability_test.png, pred_vs_true_test.png, history.csv, model.eqx, config.yaml}`.
Metrics: MAE, CRPS, 80 % coverage and width, C-index, ECE; overall, per storage group, and fruit-weighted.
`metrics_by_n_obs.json` is claim 4.

## 4. E2 — sampling frequency (claim 3)

```bash
sbatch ... slurm/sampling_ablation.sh avocado        # gru_d, latent_ode, latent_sde x every 2 / 3 days
# evaluate the daily-trained models on sparser test visits without retraining:
python scripts/evaluate.py results/runs/latent_sde_avocado --every 2
python scripts/evaluate.py results/runs/latent_sde_avocado --every 3
```

## 5. E3 — unseen storage group (claim 2)

```bash
sbatch ... slurm/logo.sh avocado                     # 3 held-out groups x {direct_reg, gru_d, latent_sde}
# clock ablation on the hardest fold:
python scripts/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=T10 model.clock=group
python scripts/train.py model=latent_sde data=avocado +experiment=logo splits.held_out=T10 model.clock=fixed
```

Report the fitted activation energy and Q10 from the Arrhenius clock next to the 50 kJ/mol prior.

## 6. E4 — gradient methods through the SDE solve

```bash
sbatch ... slurm/gradient_methods.sh avocado         # backprop | checkpoint | adjoint, same model and seed
```

Compare wall-clock per epoch, peak memory and final test metrics; cross-reference `experiments/adjoint_tradeoffs.py`.

## 7. Multi-produce (when TriModal / FreshLab / Stevens data are in hand)

```bash
python scripts/export_manifest.py --root data/avocado --out data/stevens/manifest.csv   # schema check
python scripts/train.py model=latent_sde data=stevens
```

Leave-one-produce-out uses the same `splits.mode=logo` machinery with `key="produce"` (see
`ripen.data.splits.leave_one_group_out`).

## 8. Tables and figures

```bash
make summary                         # results/summary.md
```

Every figure in the README is produced by a script in `scripts/` or `experiments/`; nothing is hand-edited.
Cells marked TBD in the README stay TBD until the corresponding run exists.
