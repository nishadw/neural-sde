# Literature review: distributional shelf-life prediction from produce images

*Written 2026-10-20 for the CMSC 838B project. Every entry below was opened during the review (full text, abstract, or Crossref/arXiv record). The access level is marked per entry: **[full]** full text read, **[abs]** abstract read, **[meta]** only the bibliographic record verified (title, authors, venue, DOI) — content claims for [meta] entries are limited to what the title states. Nothing is cited from memory.*

## 0. The problem in one paragraph

A piece of produce is photographed on a schedule (daily, or every other day) until it spoils. From the photos so far, predict a *distribution* over the spoilage time τ — not a point estimate. Three facts shape the modelling: (i) spoilage is only ever observed between two visits, so τ is **interval-censored**; (ii) items still unspoiled when the study ends are **right-censored**; (iii) the rate of ripening depends on storage temperature through well-studied **Arrhenius kinetics**. The field that studies (i)–(ii) is survival analysis; the field that studies (iii) is food-science kinetics; the field that models an evolving hidden state from irregular observations is continuous-time deep learning. No paper we found sits at the intersection of all three.

## 1. Produce shelf-life and ripeness from images

### 1.1 The avocado dataset and its two published models

**Xavier, Rodrigues & Silva (2024).** *Shelf-Life Management and Ripening Assessment of 'Hass' Avocado (Persea americana) Using Deep Learning Approaches.* Foods 13(8):1150. doi:10.3390/foods13081150. **[full]**
478 fruits, 14,710 photos, two sides per fruit per day, three storage groups (T10: 192 fruits at 10 °C/85 % RH; T20: 143 at 20 °C/85 % RH; Tamb: 143 at 15.8–21.7 °C, mean 18.7 °C, RH unlogged). Two trained raters assigned a 5-stage and a 10-stage Ripening Index to every photo. "Days Left" = days until the fruit reached stage 5 (or stage 9 of 10), added after the fact. Models: AlexNet and ResNet-18 transfer learning as per-photo stage classifiers (MATLAB), 70/15/15 split by fruit. Shelf life is then a deterministic linear rule fitted per group by OLS: Days Left = α·(stage − 5), with α = −4.390 / −2.116 / −1.929 days per stage for T10 / T20 / Tamb (Table 4; R² 0.953–0.963). Table 3: 5-stage accuracy 78.4 % (ResNet-18), 99.7 % within one stage. Table 6: mean absolute shelf-life loss 1.04 days overall by sample (T10 1.82, T20 0.75, Tamb 0.54); human raters using the same rule score 0.99 days.
What it does not do: no uncertainty, no censoring (fruits that never reached the endpoint simply have no Days Left), every photo treated as i.i.d. — the per-fruit trajectory is discarded — and the storage group is an oracle label chosen at prediction time. The authors note that T20 and Tamb ripened about twice as fast as T10, that errors between the two CNNs overlapped less than half the time, and that models trained on T10 alone lose accuracy at late stages because cold-stored fruit soften without darkening.

Rates implied by Table 4: v = 1/|α| = 0.228 / 0.473 / 0.518 stages per day for T10 / T20 / Tamb. An Arrhenius fit to T10 vs T20 gives Eₐ ≈ 50 kJ/mol. **But Tamb (mean 18.7 °C) ripened *faster* than T20 (20 °C)**, which a mean-temperature Arrhenius law cannot produce. Candidate explanations: temperature fluctuation (15.8–21.7 °C), unlogged humidity, or rater drift. This matters for direction 2 (§6).

**Hangud & Hangud (2026).** *Storage Conditions Improve Deep Learning Prediction of Remaining Avocado Shelf-Life.* National High School Journal of Science, 26 Sept 2026. No DOI. https://nhsjs.com/2026/storage-conditions-improve-deep-learning-prediction-of-remaining-avocado-shelf-life/ **[full]**
Same Mendeley dataset. ResNet-18 with late fusion of the image branch and a storage-condition branch, direct regression of days remaining (Huber loss, δ = 2), single photo per prediction, repeated grouped 5-fold CV (×2) split at the fruit level. Fruit-weighted MAE: image-only 1.658 d → image + storage label 1.126 d (T10 1.640, T20 0.810, Tamb 0.702). A categorical storage label performed the same as the continuous temperature. **Right-censored fruits were excluded** ("assigning them an imputed value would have fabricated labels"): 401 of 478 fruits retained, i.e. 16 % discarded, unevenly across groups (9.1 %–26.6 %). Uncertainty is reported only as bootstrap CIs on the MAE, not per prediction. Venue is a high-school journal; the methodology is sound and it is the strongest published number on this data, so it is the baseline to beat.

### 1.2 Other avocado work

**Nuanmeesri (2025).** *Enhanced hybrid attention deep learning for avocado ripeness classification on resource constrained devices.* Sci Rep 15:3719. doi:10.1038/s41598-025-87173-7. **[full, via PMC]** Own dataset of 200 Peruvian Hass (1,000 raw images, 14,000 after augmentation), five classes incl. Rotten. 19 CNN backbones plus a hybrid-attention head; best test accuracy 91.25 %. Pure classification, no time axis, no shelf life.

**(2025) *Explainable AI and mobile imaging for non-destructive avocado ripeness and internal quality assessment to reduce food waste.*** Curr Res Food Sci, doi:10.1016/j.crfs.2025.101196. **[abs + partial text]** 1,400 smartphone images over 8 days at room temperature; firmness from a texture analyser as the regression target (CNN ResNet regressor, R² 0.919); shelf life then read off firmness via industry guidelines. Point estimates only, one storage condition. It cites Sierra et al. (2019) as having fitted an Arrhenius-based logistic model to avocado colour and firmness with R² > 0.95.

**Sierra, Londoño, Gómez & Herrera (2019).** *Evaluation and modeling of changes in shelf life, firmness and color of 'Hass' avocado depending on storage temperature.* Food Sci Technol Int. doi:10.1177/1082013219826825. **[meta]** The food-science precedent for temperature-dependent avocado shelf-life kinetics.

**Sibeko, Shikwambana & Mathaba (2024).** *Effect of ripening temperature on early-season 'Hass' avocado fruit exocarp colour development and pigmentation during ripening.* Hortic Sci. doi:10.17221/72/2023-hortsci. **[meta]** Supports the Xavier observation that skin colour at a given softness depends on temperature — i.e. the image→stage map itself is temperature-dependent, not only the clock.

### 1.3 Other produce, imaging, and time

**Kamat, Chandekar, Dlima, Gite, Pradhan & Alamri (2025).** *Comprehensive dataset on ripening stages of strawberries and avocados: From unripe to rotten.* Data in Brief, doi:10.1016/j.dib.2025.111663. **[full, via PMC]** 1,333 original images (14,630 with augmentation), four classes, photographed daily over 25 days. Per-item identifiers are not described; treat as snapshot data until the files are inspected.

**Unnikrishnan, Deepak, Aishwaryaa & Bagyammal (2026).** *Multimodal perishable fruits and vegetables dataset* ("TriModal Ripeness 6"). Data in Brief, doi:10.1016/j.dib.2026.112545; data doi:10.6084/m9.figshare.30783827. **[full, via PMC + figshare record]** Six produce types followed three times daily until spoilage with RGB, thermal and methane readings; temperature and humidity were *not* logged. See the dataset survey.

**(2021) *Predicting sensitivity of recently harvested tomatoes and tomato sepals to future fungal infections.*** Sci Rep, doi:10.1038/s41598-021-02302-2. **[abs]** NIR hyperspectral image at harvest → susceptibility to fungal infection four days later. A forecasting (not monitoring) framing, but a fixed horizon and a binary target, no distribution over time.

**(2023) *Development of a longevity prediction model for cut roses using hyperspectral imaging and a convolutional neural network.*** Front Plant Sci, doi:10.3389/fpls.2023.1296473. **[meta]** Title-level: longevity as a regression target from HSI.

Reviews, title-level only: *Artificial Vision Systems for Fruit Inspection and Classification: Systematic Literature Review*, Sensors 2025, doi:10.3390/s25051524 **[meta]**; *Overview of Deep Learning and Nondestructive Detection Technology for Quality Assessment of Tomatoes*, Foods 2025, doi:10.3390/foods14020286 **[meta]**.

**Summary of §1.** Every produce-imaging paper we opened predicts either a discrete ripeness class or a point estimate of firmness / days left from a single image. None models the per-item trajectory, none handles censoring (the one paper that mentions it discards censored items), and none outputs a predictive distribution over spoilage time.

## 2. Deep time-to-event (survival) models

**Katzman et al. (2016/2018), DeepSurv.** arXiv:1606.00931. **[abs]** Cox partial likelihood with a neural risk score; proportional hazards assumed.

**Kvamme, Borgan & Scheel (2019), Cox-Time.** arXiv:1907.00825. **[abs]** Drops proportional hazards by letting the risk score depend on time; introduces the `pycox` losses.

**Lee, Zame, Yoon & van der Schaar (2018), DeepHit.** AAAI 2018, doi:10.1609/aaai.v32i1.11842. **[meta]** Discrete-time, nonparametric PMF over time bins with a ranking loss; competing risks.

**Lee, Yoon & van der Schaar (2020), Dynamic-DeepHit.** IEEE TBME, doi:10.1109/tbme.2019.2909027. **[meta]** DeepHit with an RNN over longitudinal covariates — the standard "sequence → survival distribution" baseline.

**Luck, Sylvain, Cardinal, Lodi & Bengio (2017).** arXiv:1705.10245. **[abs]** Directly models the survival function rather than the hazard.

**Avati, Duan, Zhou, Jung, Shah & Ng (2018), Countdown Regression.** arXiv:1806.08324. **[abs]** Defines **Survival-CRPS with right- and interval-censored variants** and shows training on it gives sharper, still-calibrated distributions than MLE. This is the evaluation metric the project uses and it can also be a training loss for any model in the zoo.

**Haider, Hoehn, Davis & Greiner (2018).** *Effective Ways to Build and Evaluate Individual Survival Distributions.* arXiv:1811.11347. **[meta]** Source of D-calibration for individual survival curves.

**Jarrett, Yoon & van der Schaar (2018), MATCH-Net.** arXiv:1811.10746. **[abs]** Convolutional dynamic survival prediction from longitudinal data.

**Mesinovic, Watkinson & Zhu (2023), DySurv.** arXiv:2310.18681. **[abs]** Conditional VAE for dynamic survival from longitudinal EHR.

**Shahin et al. (2023), CenTime.** arXiv:2309.03851. **[abs]** Event-conditional censoring model; imaging (CT) inputs.

**Wiegrebe, Kopper, Sonabend, Bischl & Bender (2023).** *Deep Learning for Survival Analysis: A Review.* arXiv:2305.14961. **[abs]** **Chen (2024).** *An Introduction to Deep Survival Analysis Models.* arXiv:2410.01086. **[abs]** The two surveys to point readers at.

**Metrics.** Harrell-style concordance in its time-dependent form: Antolini, Boracchi & Biganzoli (2005), doi:10.1002/sim.2427 **[meta]**. Integrated Brier score: Graf, Schmoor, Sauerbrei & Schumacher (1999), doi:10.1002/(sici)1097-0258(19990915/30)18:17/18<2529::aid-sim274>3.0.co;2-5 **[meta]**. Proper scoring rules and CRPS: Gneiting & Raftery (2007), JASA, doi:10.1198/016214506000001437 **[meta]**.

**Summary of §2.** Dynamic survival models exist and are the right baselines (Dynamic-DeepHit, DySurv, SurvLatent ODE in §3). They model the hazard or the PMF directly; none derives the event from a learned latent process hitting a boundary, and apart from Avati et al. none handles interval censoring.

## 3. Continuous-time sequence models

**Chen, Rubanova, Bettencourt & Duvenaud (2018), Neural ODEs.** arXiv:1806.07366. **[abs]**
**Rubanova, Chen & Duvenaud (2019), Latent ODEs / ODE-RNN.** arXiv:1907.03907. **[abs]** The encoder family for irregular visits.
**Che, Purushotham, Cho, Sontag & Liu (2016), GRU-D.** arXiv:1606.01865. **[meta]** Decay-to-mean GRU with masks and time gaps.
**Kidger, Morrill, Foster & Lyons (2020), Neural CDEs.** arXiv:2005.08926. **[meta]** Continuous-time RNN driven by an interpolated path.
**Li, Wong, Chen & Duvenaud (2020), Scalable Gradients for SDEs / latent SDE.** arXiv:2001.01328. **[meta]** Stochastic adjoint and the latent-SDE ELBO via Girsanov; the training objective for the main model.
**Kidger, Foster, Li & Lyons (2021), Efficient and Accurate Gradients for Neural SDEs.** arXiv:2105.13493. **[abs]** Reversible Heun and the exact discrete adjoint.
**Kidger, Foster, Li, Oberhauser & Lyons (2021), Neural SDEs as Infinite-Dimensional GANs.** arXiv:2102.03657. **[meta]**
**Tzen & Raginsky (2019).** arXiv:1905.09883. **[abs]** Neural SDEs as the diffusion limit of deep latent Gaussian models.
**Heck, Gelbrecht, Schaub & Boers (2024).** *Improving the Noise Estimation of Latent Neural SDEs.* arXiv:2412.17499. **[abs]** Latent SDEs trained by the ELBO **systematically underestimate the diffusion**; an explicit noise regulariser fixes it. Directly relevant: our first-passage spread depends on σ, so the synthetic harness must test σ recovery, and this regulariser is a config option.
**Moon, Groha & Gusev (2022), SurvLatent ODE.** arXiv:2204.09633. **[abs]** ODE-RNN encoder over irregular longitudinal data → latent embedding → nonparametric hazards with competing risks. The closest existing "continuous-time encoder + survival head" model; it is a baseline here. It does not model the event as a first passage, has no stochastic latent dynamics, and does not address interval censoring.
**Laurie & Lu (2023), TDNODE.** arXiv:2308.01362. **[abs]** Neural-ODE tumour dynamics feeding overall-survival prediction; mechanistic flavour, deterministic.

## 4. First-passage-time and interval-censored models

**Cristofoletto, Rollo, Birolo & Fariselli (2025), DeepFHT.** arXiv:2510.00733. **[abs]** A network maps *baseline* covariates to the initial condition, drift and diffusion of a *closed-form* first-hitting-time process (Brownian motion with or without drift), giving analytic survival and hazard. Matches Cox on accuracy with a physical parameterisation. It is the nearest neighbour to direction 1: same "event = boundary hit" idea. Differences from what we propose: the latent process is linear with closed-form FHT (no learned nonlinear drift/diffusion), there is no sequence of observations updating the latent state, no images, and only right censoring.

**Si, Wang, Hu & Zhou (2012).** *Remaining Useful Life Estimation Based on a Nonlinear Diffusion Degradation Process.* IEEE Trans Reliab, doi:10.1109/tr.2011.2182221. **[meta]** The reliability-engineering lineage of first-passage RUL from a diffusion degradation model; the analogy (produce ripening ≈ degradation to a failure threshold) is worth stating in the paper.

**Wang, Zeng, Sun, Cho & Ding (2026), ICODEN.** arXiv:2602.10303. **[abs]** Neural hazard, cumulative hazard obtained by solving an ODE, interval-censored likelihood, baseline covariates (SNPs). Assumption-lean survival for interval censoring — but static covariates, no latent state, no first passage.

**García Meixide, Matabuena, Abraham & Kosorok (2022).** *Neural interval-censored survival regression with feature selection.* arXiv:2206.06885. **[meta]**
**Yuan, Zhao, Li, Song & Chen (2025).** *Interpretable Deep Regression Models with Interval-Censored Failure Time Data.* arXiv:2503.19763. **[meta]**
**Cho, Jewell & Kosorok (2019), ICRF.** arXiv:1912.09983. **[meta]** **Fu & Simonoff (2017).** Survival trees for interval-censored data, arXiv:1702.07763. **[meta]** Tree ensembles for interval censoring — a non-deep baseline if wanted.

**Ordinal heads.** Cao, Mirjalili & Raschka (2019), CORAL, arXiv:1901.07884 **[meta]**; Shi, Cao & Raschka (2021), CORN, arXiv:2111.08851 **[meta]**. Rank-consistent ordinal regression for the 5-stage observation head.

**Active sensing (direction 5).** Yoon, Jordon & van der Schaar (2019), ASAC, arXiv:1906.06796. **[abs]** Learns when to make costly observations in a sequential medical setting. General machinery exists; no produce or first-passage application.

## 5. Arrhenius kinetics in food science

**Van Boekel (2008).** *Kinetic Modeling of Food Quality: A Critical Review.* Compr Rev Food Sci Food Saf, doi:10.1111/j.1541-4337.2007.00036.x. **[meta]** The standard review: quality attributes follow zero/first-order kinetics with Arrhenius temperature dependence; discusses variability between items as a modelling problem.
**Taoukis & Labuza (1989).** *Applicability of Time-Temperature Indicators as Shelf Life Monitors of Food Products.* J Food Sci, doi:10.1111/j.1365-2621.1989.tb07882.x. **[meta]** Time–temperature integration: shelf life consumed is ∫k(T(t))dt — the "physiological time" idea behind κ(T).
**Labuza, Saguy & Taoukis (1997).** *Kinetics of Food Deterioration and Shelf-Life Prediction.* In Handbook of Food Engineering Practice, doi:10.1201/9781420049077.ch9. **[meta]**
**Sierra et al. (2019)** — see §1.2: Arrhenius-based shelf-life model for Hass specifically.

**Summary of §5.** The food-science view is that ripening progresses on a clock whose rate is an Arrhenius function of temperature, integrated over the temperature history. This is a prior, not a learned quantity, in all the above; it has never been combined with a learned latent state from images.

## 6. Gap table

| # | Direction | Closest prior work | What exists | What does not | Verdict |
|---|---|---|---|---|---|
| 1 | Interval-censored first-passage likelihood for a latent neural SDE | DeepFHT (2510.00733); ICODEN (2602.10303); Avati (1806.08324); Si 2012; SurvLatent ODE (2204.09633) | Event-as-first-passage with closed-form BM and static covariates (DeepFHT). Interval-censored neural hazards with static covariates (ICODEN). Interval-censored CRPS as a loss (Avati). ODE-RNN encoder → hazard head (SurvLatent ODE). | A *learned nonlinear* latent SDE whose first passage to an ordinal boundary is fit to *interval-censored* visit data, with the latent state updated from an image sequence. Also unaddressed: the pathwise gradient of a first-passage probability is zero (lecture, slide 14), so the estimator itself is a contribution (smoothed boundary / score-function / likelihood via the discretised Fokker–Planck). | **Open. Headline claim.** |
| 2 | Shared ripening clock with produce-specific Arrhenius κ(T); transfer to unseen temperature / produce | Sierra 2019; Van Boekel 2008; Taoukis & Labuza 1989; Hangud 2026; Sibeko 2024 | Arrhenius shelf-life models fit per attribute with hand-measured quality; storage label as a late-fusion feature (Hangud: categorical ≈ continuous T with only 3 levels). | Learning κ(T) jointly with latent dynamics from images; cross-produce transfer; extrapolation to an unseen temperature. **Caveat found:** Tamb ripened faster than T20 in the avocado data despite lower mean temperature, so κ(mean T) alone will not fit all three groups — κ must take the temperature *history* (Taoukis-style integral) or a group embedding, and the paper should say so. Sibeko 2024 implies the image→stage map is also temperature-dependent. | **Open, with the caveat.** Needs FreshLab-Tomato or TriModal for cross-produce; FreshLab logs T. |
| 3 | Uncertainty vs sampling interval (subsample daily data to 2–3 days) | none found | — | Any quantification of how calibration and CRPS degrade with visit spacing for produce. | **Open but incremental.** Run it as an ablation; not a headline claim. |
| 4 | Single- vs multi-photo prediction; photo dropout at training | Hangud 2026 (single photo); Xavier 2024 (per-picture vs per-sample); Dynamic-DeepHit / DySurv (sequence inputs) | Single-image regression; sequence survival models in EHR. | Interval-width-per-added-photo curves for produce; training with random visit dropout so one-photo and n-photo predictions come from one model. | **Open, moderate novelty.** Natural second experiment. |
| 5 | When to take the next photo | ASAC (1906.06796) | Active sensing with actor–critic in medicine. | Next-observation timing chosen to minimise first-passage uncertainty. | **Open but optional.** Defer unless 1–2 are done early. |
| — | Using right-censored fruits at all | Hangud 2026 dropped 77/478; Xavier 2024 never labelled them | — | Both published avocado models throw censored fruit away; T10 has 26.6 % censoring and the worst MAE (1.82 d). A censoring-aware likelihood is a free, defensible gain on this data. | **Open. Fold into 1.** |

**Claims that do not survive:** "first neural first-passage survival model" (DeepFHT, Oct 2025); "first interval-censored neural survival model" (ICODEN, Avati, Meixide); "first continuous-time encoder for survival" (SurvLatent ODE). The defensible headline is the *combination* in row 1 plus the physical clock in row 2, demonstrated on image sequences with the censoring both prior avocado papers discarded.

## 7. Not verified / not found

- No paper was found that models per-item produce ripening as a stochastic process from images. Searches: arXiv (cs.LG, stat.ML, q-bio), PMC (190+ hits on avocado + shelf-life + deep learning; 310 on shelf-life + image + CNN), Crossref.
- Labuza's 1982 book *Shelf-Life Dating of Foods* could not be resolved to a DOI; the 1997 chapter is cited instead.
- Author names for doi:10.1016/j.crfs.2025.101196 and doi:10.1038/s41598-021-02302-2 were not extracted from the PMC XML; the DOIs are verified.
- The NHSJS paper has no DOI; cited by URL and date.
- Firecrawl was out of credits for the whole review; sources were opened through PMC E-utilities, the arXiv MCP, the Crossref API, the figshare API, and direct page fetches.
