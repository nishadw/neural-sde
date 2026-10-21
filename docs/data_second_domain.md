# Second application domains: PBC2 and NASA C-MAPSS

Two public longitudinal time-to-event datasets that fit the `Item` schema without
any change to the models. Both ship a registry event time rather than a labelled
"spoiled" visit, so `stage` is `None` everywhere and the loaders write the spoilage
interval directly. Counts below were computed from the downloaded files on
2026-10-20 with the loaders in `src/ripen/data/pbc2.py` and `src/ripen/data/cmapss.py`;
`tests/test_second_domain.py` re-checks them whenever the files are present.

## Why these two

PBC2 is the standard benchmark for dynamic survival prediction from longitudinal
covariates: it is the dataset Dynamic-DeepHit (Lee, Yoon & van der Schaar, IEEE TBME
2020) ships and evaluates on, and it is the reference example of the joint-modelling
literature (Rizopoulos' `JM`/`JMbayes`). Showing the latent-SDE forecaster on it puts
the method on the same footing as those baselines, with a competing event (transplant)
to exercise the censoring machinery. C-MAPSS is degradation to an absorbing failure:
every training engine runs from healthy to broken, sensor readings drift as damage
accumulates, and the failure time is exact. It is the closest industrial analogue of
produce spoilage (monotone deterioration under a known operating regime, observed at
inspections) and lets the visit schedule, inspection spacing and censoring fraction be
controlled exactly, which the produce data does not allow.

## PBC2

**Source.** `pbc2_cleaned.csv` from the Dynamic-DeepHit repository,
<https://github.com/chl8856/Dynamic-DeepHit/tree/master/data>, raw file
<https://raw.githubusercontent.com/chl8856/Dynamic-DeepHit/master/data/pbc2_cleaned.csv>.
It is the Mayo Clinic primary biliary cirrhosis trial of D-penicillamine
(Murtaugh et al., Hepatology 1994), as distributed in the R package `JM` and cleaned by
the Dynamic-DeepHit authors. Download to `data/pbc2/pbc2_cleaned.csv` (`data/` is gitignored).

**License / access terms as stated by the source.** The GitHub repository has no
`LICENSE` file and the GitHub API reports `license: null`; the README cites the TBME
paper and says nothing about data terms. The CSV itself carries no header comment. The
underlying `pbc2` data are distributed in the GPL-licensed R packages `JM` and
`JMbayes`. Treat it as research-use data from a public repository; nothing more
specific is stated by the source.

**File.** 1945 rows x 19 columns, 312 patients, 1 to 16 visits each (median 5).
Columns: `id, tte, times, label, drug, age, sex, ascites, hepatomegaly, spiders, edema,
serBilir, serChol, albumin, alkaline, SGOT, platelets, prothrombin, histologic`. `tte`
and `label` are constant within a patient; every patient's first `times` is 0. Missing
values: `serChol` 821 rows, `platelets` 73, `hepatomegaly` 61, `ascites` 60,
`alkaline` 60, `spiders` 58.

**Time unit.** `tte` and `times` are integer weeks, not years: the median visit gap is
51 (six-monthly then yearly protocol visits), patient 1's `tte` of 57 matches the 1.095
years of the original record, and the maximum `tte` of 744 is 14.3 years x 52. The
loader converts to months with `months_per_unit = 12 / 52`. Follow-up therefore runs to
171.7 months; `configs/data/pbc2.yaml` suggests `horizon=180`.

**Event definition.** `label` is 0 censored (143 patients), 1 death (140), 2 liver
transplant (29). Death is the event. Transplant is a competing event: with the default
`transplant_as_event=False` it is right-censoring at the transplant time and
`meta["competing"] = True`; with `transplant_as_event=True` it is a composite endpoint
(169 events).

**Counts (loader defaults).** 312 items, 140 events, 172 right-censored (143 + 29),
1937 visits (8 visits that fell on the day of death were dropped, see below), feature
dimension 15, arms `D-penicil` 158 / `placebo` 154. Latest event 166.8 months, latest
censoring 171.7 months. Event intervals are narrow: median width 3.9 months, mean 10.4,
maximum 128.3 (a patient with a long gap between the last visit and death); none is
exactly zero.

**Mapping to `Item`.**

| field | value |
|---|---|
| `item_id` | `pbc2-<id>` |
| `produce` | `"pbc2"` |
| `storage` | arm: `"D-penicil"` (`drug == 1`, 158 patients, which matches the trial's 158 D-penicillamine patients) or `"placebo"` |
| `temp_c`, `rh` | `nan` |
| `visits[k].t` | `times * 12 / 52`, months since the first visit; first visit at 0 |
| `visits[k].stage` | `None` |
| `visits[k].features` | `(1, 15)` float32: 12 longitudinal columns (`serBilir, serChol, albumin, alkaline, SGOT, platelets, prothrombin, ascites, hepatomegaly, spiders, edema, histologic`) then `age, sex, drug`; `serBilir, serChol, alkaline, SGOT` are `log1p`-transformed; missing values are carried forward within the patient and then set to the column mean; every column standardised with the mean/std over all rows of the file (recorded in `meta["feature_mean"]`, `meta["feature_std"]`, names in `meta["feature_names"]`) |
| `spoil_lo`, `spoil_hi` (event) | `(t of the last visit before T, T]`, with `T = tte * 12 / 52`. The event time is exact, so this is a narrow interval that says "alive at the last visit, dead at T". Visits at `t >= T` are dropped (8 death patients had their last visit on the day of death) and counted in `meta["n_visits_dropped_at_event"]`; a patient whose only visit is the event day keeps it and the interval collapses to a point |
| `spoil_lo`, `spoil_hi` (censored) | `(T, inf]` where `T` is the censoring or transplant time, which can be later than the last visit; the last visit time is in `meta["t_last_visit"]` |
| `meta` | `id, label, event, competing, transplant_as_event, t_event, t_last_visit, tte_file_units, months_per_unit, n_visits_dropped_at_event, feature_*` |

Subsampling (`subsample_every > 1`) keeps the registry event time and only moves
`spoil_lo` back to the last kept visit (`ripen.pipeline.subsample_visits_known_event`).

## NASA C-MAPSS

**Source.** `CMAPSSData.zip` from <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
(the data.nasa.gov legacy mirror of the NASA Prognostics Center of Excellence data
repository; the URL redirects to a signed S3 link and downloaded in one try, 12.4 MB).
Unzip under `data/cmapss/`. The archive holds `train_FD00x.txt`, `test_FD00x.txt`,
`RUL_FD00x.txt` for x = 1..4, `readme.txt` and the PHM 2008 paper
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"
(Saxena, Goebel, Simon & Eklund). Only the four training files are used.

**License / access terms as stated by the source.** The `readme.txt` in the archive
contains no license statement. The data.nasa.gov catalog entry for
"CMAPSS Jet Engine Simulated Data" shows "License not specified". The PCoE data
repository page lists a "Data Set Citation" for each set (for this one, the PHM 2008
paper) and no license text. Nothing more specific is stated by the source.

**File format (verified).** Whitespace-separated numeric text, 26 columns, no header,
trailing space on each line, one row per operational cycle: `unit, cycle, setting1,
setting2, setting3, s1 .. s21`. Cycles are contiguous `1..L` for every unit.

| subset | engines (train) | conditions / fault modes | cycles per engine | visits (default schedule) | kept features |
|---|---|---|---|---|---|
| FD001 | 100 | 1 / 1 (HPC) | 128-362 | 2101 (13-36 per engine, median 20) | 17 (drops `setting3, s1, s5, s10, s16, s18, s19`) |
| FD002 | 260 | 6 / 1 | 128-378 | 5469 | 24 |
| FD003 | 100 | 1 / 2 (HPC + fan) | 145-525 | 2505 | 18 (drops `setting3, s1, s5, s16, s18, s19`) |
| FD004 | 249 | 6 / 2 | 128-543 | 6215 | 24 |

The readme says FD004 has 248 training trajectories; the file has 249 units. Loading all
four together gives 709 items with 24 features.

**Event definition.** Failure is exact at the last recorded cycle `L` of each training
trajectory (`meta["failure_cycle"]`). No engine is censored unless `drop_tail > 0`.

**Visit schedule.** Visits are the sensor rows at cycles `c_0 = 1` and
`c_k = 1 + k * inspect_every + U{-jitter..jitter}` (forced strictly increasing), for all
`c_k < L`; defaults `inspect_every = 10`, `jitter = 2`, `seed = 0`. Time is in units of
`inspect_every` cycles: `t = (c - 1) / inspect_every`, so consecutive visits are 0.6-1.4
units apart (mean 1.0) and the first is at 0. The resulting event interval
`(t_last_visit, t_failure]` has mean width 0.52-0.59 units across subsets, so it is a
narrow interval rather than a point. The time horizon is 36.1 units for FD001 and 54.2
for FD004 (`configs/data/cmapss.yaml` suggests `horizon=40` for FD001).

**Censoring ablation.** `drop_tail = f` right-censors `round(f * n_engines)` engines
(FD001, f = 0.3: 30) at a uniform cycle `C` in `[1, L)`: visits after `C` are dropped,
`spoil_lo = t(C)`, `spoil_hi = inf`, and `meta["failure_cycle"]` keeps the truth for
evaluation.

**Mapping to `Item`.**

| field | value |
|---|---|
| `item_id` | `<subset>-<unit>`, e.g. `FD001-042` |
| `produce` | `"turbofan"` |
| `storage` | subset name `FD001`..`FD004` (encodes the operating-condition and fault-mode regime) |
| `temp_c`, `rh` | `nan` |
| `visits[k].t` | `(cycle - 1) / inspect_every`; first visit at 0 |
| `visits[k].stage` | `None` |
| `visits[k].features` | `(1, D)` float32: the three settings and 21 sensors at that cycle, columns that are constant over the loaded rows dropped (`drop_constant=True`), standardised with the mean/std over all rows of the loaded subsets (`meta["feature_names"]`, `meta["dropped_columns"]`, `meta["feature_mean"]`, `meta["feature_std"]`) |
| `spoil_lo`, `spoil_hi` | `(t of last visit, t(L)]`; censored engines `(t(C), inf]` |
| `meta` | `unit, subset, failure_cycle, visit_cycles, censored_at_cycle, inspect_every, jitter, seed, feature_*` |

## Running

```
curl -sSL -o data/pbc2/pbc2_cleaned.csv \
  https://raw.githubusercontent.com/chl8856/Dynamic-DeepHit/master/data/pbc2_cleaned.csv
curl -sSL -o data/cmapss/CMAPSSData.zip https://data.nasa.gov/docs/legacy/CMAPSSData.zip \
  && (cd data/cmapss && unzip -o CMAPSSData.zip)
JAX_PLATFORMS=cpu python -m pytest -q tests/test_second_domain.py
python scripts/train.py data=pbc2 horizon=180
python scripts/train.py data=cmapss horizon=40
```

Both loaders standardise with statistics over the whole file. That leaks nothing about
labels but does use test-fold covariate moments; the means and standard deviations are in
every item's `meta` so a split-aware caller can refit them on the training fold.
