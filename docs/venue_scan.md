# Venue scan: what gets accepted, and where the gaps are

*Compiled 2026-10-20 by a research agent; every venue below was confirmed on a proceedings page, OpenReview forum
or conference poster page, and IDs come from the arXiv API. "UNVERIFIED" = no primary page confirmed.
Target venues: NeurIPS, ICML, ICLR, AISTATS, UAI, 2021–2026.*

## (a) Deep time-to-event / survival

| # | Paper | Venue | ID | Contribution | Shape |
|---|---|---|---|---|---|
| 1 | Inverse-Weighted Survival Games (Han et al.) | NeurIPS 2021 | arXiv:2111.08175 | Failure and censoring models trained as a game of re-weighted proper scores whose stationary point is the true pair | loss + theory |
| 2 | Deep Extended Hazard Models (Zhong et al.) | NeurIPS 2021 | ACM 10.5555/3540261.3541419 | Neural extended-hazard model nesting Cox PH and AFT (a covariate-dependent time rescaling), with consistency and rates | model class + theory |
| 3 | Survival regression with proper scoring rules and monotonic nets, SuMo-net (Rindt et al.) | AISTATS 2022, PMLR 151 | arXiv:2103.14755 | **Proves time-dependent C-index, Survival-CRPS, IBS, IBLL are not proper under right censoring; the censored log-likelihood is** | theory + loss |
| 4 | DeSurv (Danks & Yau) | AISTATS 2022, PMLR 151 | PMLR v151/danks22a | Competing-risk CIFs as neural ODEs, continuous time | model class |
| 5 | Censored Quantile Regression Neural Networks (Pearce et al.) | NeurIPS 2022 | arXiv:2205.13496 | Quantile grid with a censoring-aware pinball loss | loss |
| 6 | Proper Scoring Rules for Survival Analysis (Yanagisawa) | ICML 2023, PMLR 202 | arXiv:2305.00621 | Four strictly proper scores extended to right censoring, properness under discretisation | theory + loss |
| 7 | An Effective Meaningful Way to Evaluate Survival Models, MAE-PO (Qi et al.) | ICML 2023 | arXiv:2306.01196 | Pseudo-observation MAE for censored test sets | evaluation |
| 8 | Neural Frailty Machine (Wu et al.) | NeurIPS 2023 | arXiv:2303.10358 | Multiplicative frailty with nonparametric rates | model class + theory |
| 9 | Diffsurv (Vauvelle et al.) | NeurIPS 2023 | arXiv:2304.13594 | Differentiable sorting consistent with censoring | loss |
| 10 | Copula-Based Deep Survival Models for Dependent Censoring (Foomani et al.) | UAI 2023, PMLR 216 | arXiv:2306.11912 | Copula links event and censoring | model class |
| 11 | MOTOR (Stanford) | ICLR 2024 spotlight | arXiv:2301.03150 | TTE foundation model pretrained on 55M EHR records | pretraining + application |
| 12 | Conformalized Survival Distributions, CSD (Qi et al.) | ICML 2024, PMLR 235 | arXiv:2405.07374 | Conformal recalibration of whole survival curves | post-hoc estimator |
| 13 | CiPOT (Qi et al.) | NeurIPS 2024 | arXiv:2410.20579 | Conformal vertical rescaling with marginal + conditional calibration guarantees | estimator + theory |
| 14 | Conformalized Survival Analysis for General Right-Censored Data (Davidov et al.) | ICLR 2025 | — | Conformal lower bounds beyond type-I censoring | estimator + theory |
| 15 | Boosting for interval-censored data (Bian et al.) | ICLR 2025 | arXiv:2601.17973 | Censoring-unbiased transformation / imputation; tabular, static covariates | loss |
| 16 | Time-to-Event Pretraining for 3D Medical Imaging (Huo et al.) | ICLR 2025 | arXiv:2411.09361 | TTE pretraining of CT encoders; reports AUROC / C-index, no calibration | pretraining + application |
| 17 | VLSA (Liu et al.) | ICLR 2025 | arXiv:2409.09369 | Vision–language survival for WSIs with an ordinal incidence target | application + model |
| 18 | SurvivalBoost (Alberge et al.) | AISTATS 2025, PMLR 258 | arXiv:2410.16765 | Strictly proper, censoring-adjusted, separable score for competing risks | loss + theory |
| 19 | HACSurv (Liu et al.) | AISTATS 2025 | arXiv:2410.15180 | Hierarchical Archimedean copulas for competing risks | model class |
| 20 | Censor-Dependent Variational Inference, CDVI (Liu & Wang) | ICML 2025 | arXiv:2502.09591 | ELBO-optimal posterior for latent-variable survival depends on the censoring indicator | theory + estimator |
| 21 | Asymmetric Laplace survival distributions, ALD (Sheng & Henao) | ICML 2025, PMLR 267 | arXiv:2505.03712 | Closed-form quantiles, calibrated | model class |
| 22 | Doubly Robust Conformalized Survival Analysis (Sesia & Svetnik) | ICML 2025 spotlight | arXiv:2412.09729 | ML-imputed censoring times + weighted conformal | estimator + theory |
| 23 | NeuralSurv (Monod et al.) | NeurIPS 2025 | arXiv:2505.11054 | Bayesian deep survival with linear-cost VI | model class |
| 24 | KSP post-hoc calibration (Park et al.) | NeurIPS 2025 | — | Bin-free, KM-free recalibration | post-hoc estimator |
| 25 | ICALD (Sheng & Henao) | ICLR 2026 | — | PAC individual calibration for ALD survival | model + theory |
| 26 | On the calibration of survival models with competing risks (Alberge et al.) | AISTATS 2026 spotlight | arXiv:2602.00194 | Existing calibration measures fail under competing risks; two new measures | evaluation + estimator |
| 27 | SALaD (Qi et al.) | ICML 2026 | — | Explicitly models censoring via latent decomposition | estimator |
| 28 | **A Strictly Proper Scoring Rule and a Calibration Metric for Interval-Censored Data (Yanagisawa & Akiyama)** | **ICML 2026** | — | First strictly proper score and calibration metric for interval censoring; independent monitoring ≡ non-informative censoring only for Case-1 | theory + loss + evaluation |

## (b) Neural / latent SDEs for irregular series

| # | Paper | Venue | ID | Contribution | Shape |
|---|---|---|---|---|---|
| 29 | Neural SDEs as Infinite-Dimensional GANs (Kidger et al.) | ICML 2021 | arXiv:2102.03657 | SDE generator + CDE discriminator | model class |
| 30 | Efficient and Accurate Gradients for Neural SDEs (Kidger et al.) | NeurIPS 2021 | arXiv:2105.13493 | Reversible Heun, Brownian Interval | training |
| 31 | Learning Fractional White Noises in Neural SDEs (Tong et al.) | NeurIPS 2022 | — | fBM-driven neural SDEs with learned Hurst | model class |
| 32 | Latent SDEs on Homogeneous Spaces (Zeng et al.) | NeurIPS 2023 | arXiv:2306.16248 | Latent SDE on the sphere, closed-form KL | model class |
| 33 | ARCTA (Course & Nair) | NeurIPS 2023 | arXiv:2312.10550 | Solver-free latent-SDE inference | inference |
| 34 | Signature-kernel scores for neural SDEs (Issa et al.) | NeurIPS 2023 | arXiv:2305.16274 | Proper path-space scoring rules, non-adversarial | loss + theory |
| 35 | Stable Neural SDEs for irregular time series (Oh et al.) | ICLR 2024 spotlight | arXiv:2402.14989 | Three constrained SDE classes with well-posedness | model class |
| 36 | Trajectory Flow Matching (Zhang et al.) | NeurIPS 2024 spotlight | arXiv:2410.21154 | Simulation-free neural-SDE training for clinical series | training |
| 37 | Matching finite-dimensional distributions, FDM (Zhang et al.) | ICLR 2025 | arXiv:2410.03973 | Proper scores for Markov processes, O(D) training | loss + theory |
| 38 | SDE Matching (Bartosh et al.) | ICML 2025, PMLR 267 | arXiv:2502.02472 | O(1) memory latent-SDE training without simulation | training |
| — | Improving the noise estimation of latent SDEs (Heck et al.) | Chaos 35(6):063139, 2025 (not an ML venue) | arXiv:2412.17499 | Noise regulariser against diffusion under-estimation | loss |

## (c) First-passage / hitting time / degradation

| # | Paper | Venue | ID | Contribution |
|---|---|---|---|---|
| 39 | DeepFHT (Cristofoletto et al.) | **Phys. Rev. E 113:055303, May 2026 — not an ML venue** | arXiv:2510.00733; doi:10.1103/r9jt-f69r | Network maps static covariates to the parameters of closed-form Brownian first-hitting-time laws |
| 40 | SurvSurf (Chen et al.) | UNVERIFIED (submitted to RSS Data Science & AI) | arXiv:2504.04997 | Monotone net for first hitting times under intermittent observation; no process model |
| 41 | H-CDE degradation inference (Zhao & Fink) | Reliability Eng. & System Safety 2026 (UNVERIFIED) | arXiv:2509.00639 | Hierarchical neural CDE with a monotone degradation latent |
| 42 | SurvLatent ODE (Moon et al.) | MLHC 2022, PMLR 182 | arXiv:2204.09633 | Latent ODE over longitudinal covariates → competing-risk hazards |

## (d) Time reparametrisation / AFT

| # | Paper | Venue | ID | Contribution |
|---|---|---|---|---|
| 43 | Deep kernel AFT (Wu et al.) | MLHC 2021 | arXiv:2107.12250 | RNN + sparse-GP AFT with uncertainty |
| 44 | Time-warping RNNs for transfer (Hirschi) | UNVERIFIED preprint | arXiv:2604.02474 | Rescales RNN time to transfer across characteristic time scales |
| — | No accepted paper combining "Arrhenius" with neural models at the five venues. DeepEH (#2) is the only accepted one with an explicit covariate-dependent time scale. | | | |

## Synthesis

1. Survival work is accepted as (a) a loss or scoring rule with a properness proof, (b) a post-hoc calibration or
   conformal procedure with guarantees, or (c) a model class that removes an assumption and carries a rate or
   identifiability result. Pure architectures are rare unless they bring pretraining scale.
2. Evaluation papers get in when they show an existing metric is wrong (MAE-PO; competing-risk calibration).
3. Neural-SDE papers get in as training algorithms or constrained classes with well-posedness proofs; none
   derives event-time distributions from the learned process.
4. Interval censoring reached these venues only in 2025–26 (#15, #28), both tabular with static covariates.
5. **Gap (i)**: no accepted paper computes first-passage times of a learned nonlinear latent SDE. DeepFHT is
   closed-form Brownian motion (Phys Rev E); SurvSurf has no process; SurvLatent ODE is a latent ODE with a
   parametric head. Learned diffusion + boundary-crossing likelihood + interval censoring is open.
6. **Gap (ii)**: covariate-dependent clocks appear only as AFT scalings (#2) or a transfer preprint (#44); no
   accepted paper learns a physics-constrained time reparametrisation and tests transfer to unseen conditions.
7. **Gap (iii)**: calibration is studied on tabular data; image-sequence survival (#16, #17) reports
   discrimination only. The calibration metrics of #12/#13/#24/#28 transfer directly.
8. Positioning that fits: a new model class (first passage of a progression-structured latent SDE) with a
   properness statement for its interval-censored likelihood and an identifiability statement for the clock,
   trained with accepted latent-SDE machinery, evaluated with the 2024–26 calibration comparators.
