# The A2SL microwave-imaging data

*What arrived on 2026-10-20, what it contains, what it can and cannot be used for.*

The three folders came from the **A2SL group, Department of Electrical and Computer Engineering, University
of Alabama** (title slide of every deck in `avocado_jun2023/03_presentations/`). They are microwave-imaging
experiments: an antenna array around the fruit, a reconstruction (delay-and-sum beamforming; files are
labelled `DMAS` — delay-multiply-and-sum — and `CDAS` — coherent delay-and-sum — with a reconstruction
parameter `RP`) that yields a cross-sectional intensity map of the fruit's interior, repeated at several
rotation angles. Photos accompany every scan. **No ripeness, firmness or spoilage label was recorded**; the only
ground truth is the day an avocado was cut open and the cut photo.

Raw files live under `data/a2sl/` (never committed, ~0.5 GB). `python scripts/ingest_a2sl.py` converts the
HEIC photos to 1024-px JPEG and writes the manifests under `data/manifests/` (committed) and two `.npz` scan
caches (not committed).

## 1. `avocado_jun2023` — the longitudinal set

| | |
|---|---|
| Fruit | 15 Hass avocados, Mexico, bought 2023-06-22 at Sam's; mass 205–257 g, length 82–111 mm, diameter 66–75 mm (`Avocado_information.xlsx`) |
| Visits | study days 1, 3, 5, 7, 9, 11, 13 (Jun 22 – Jul 4), i.e. **every other day** |
| Design | avocados **1, 2, 3, 11** scanned at all 7 visits; avocados **4–10** scanned once each and **cut open** that day (4 → day 1, 5 → day 3, …, 10 → day 13): a staggered destructive design |
| Per visit × fruit | one side photo (HEIC, 3024²), six 75×75 reconstructions at rotations 0/60/…/300° (`02_raw_csv_files/Day_*/avo_{id}_{angle}.xlsx`, NaN outside the imaged disk, 3,209 finite pixels), the same six as PNG renders; a cut photo for the fruit cut that day |
| Extra | `DMAS_RP9.3/` and `raw_data_CDAS/` hold alternative reconstructions of angle 0 for fruit 1, 2, 3, 11 on every visit (correlation with the day folder's grid ≈ 0.96) |
| Totals | 35 (fruit, visit) pairs, 266 scan grids, 68 photos (7 of them cut photos) |

Storage temperature was not logged (lab ambient).

## 2. `pilot_may2023` — the daily pilot

6 avocados photographed daily for 13 days (May 2023) from four views (`NN_01..04`), top/bottom views from
day 8, cut photos on the cut day. 12-angle reconstruction PNGs per fruit per day (`scans/NN_avo_{angle}.png`,
angles 0–330°) but **no numeric grids**. Cut days: fruit 3 → day 1, 4 → day 4, 5 → day 7, 6 → day 10,
1 and 2 → day 13. Day 1 photos are JPEGs with a different naming scheme (`NN_x_neg_1.JPEG` …); an iCloud zip
holds 30 unassigned originals (`01_day_01/icloud_unassigned/`). 220 photos, 15 cut photos, 576 scan PNGs.

## 3. `watermelon_2022` — single-visit scans with a grower's ripeness label

7 watermelons (samples 3–9; samples 1–2 are in the log but have no files) bought June–July 2022. Each was
scanned once at up to 5 heights (low … high) × 5 rotations (0/90/162/234/306°) × 5 frequencies
(3, 3.5, 4, 4.5, 5 GHz), 96×96 grids (`ImagePoints{GHz}.csv`). Samples 3 and 5 have one position only,
sample 6 two. The log gives purchase place, type (Red Seedless / Crimson Sweet / Charleston Gray), size,
dimensions and a **ripeness note from the grower**: "Ripe", "1.5 weeks out", "1.5–2 weeks out", "3 weeks out".
471 grids. The `Other` sheet lists calibration objects with permittivity and the 2–4 GHz optimal band.

## 4. What this data supports

| Use | avocado_jun2023 | pilot_may2023 | watermelon_2022 |
|---|---|---|---|
| Trajectory model of a fruit over time | **yes** (4 fruit × 7 visits) | yes (6 × up to 13) | no (one visit) |
| Spoilage / stage supervision | **no** — nothing labelled | no | weak: grower's weeks-to-ripe |
| Second sensing modality (microwave) | **yes**, numeric | PNG renders only | yes, numeric, 5 frequencies |
| Destructive ground truth | cut photos (7) | cut photos (15) | — |
| Storage temperature | not logged | not logged | — |

So in `ripen` these sets are loaded with `stage=None` and right-censored at the last visit: they exercise the
**generative** side of the latent SDE (the feature decoder, `decode_features: true`) and the multimodal
encoder, and they are the demonstration that the same model consumes photos *and* microwave grids. The
spoilage supervision still comes from the Mendeley avocado set (478 fruit, labelled, three temperatures).

Possible label sources if they are ever wanted: rating the 22 cut photos for ripeness by eye, or asking A2SL
for their firmness notes. Both are outside the "no new data collection" rule for now.

## 5. Loader

```python
from ripen.data.a2sl import load_a2sl_avocado, load_a2sl_pilot, load_a2sl_watermelon, load_scan_cache
scans = load_scan_cache("data/a2sl/scans_avocado_jun2023.npz")
items = load_a2sl_avocado(features=photo_feature_cache, scans=scans)   # 15 Items; visit features = [photo | scan summary]
```

Scan features are fixed summaries (`ripen.data.a2sl.scan_summary`: log-intensity mean/std/max, centre-to-rim
contrast, an 8-bin radial profile; mean and spread over rotation angles → 24 dims). The raw grids stay in the
`.npz` cache for a learned scan encoder.
