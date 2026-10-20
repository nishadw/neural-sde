# Dataset survey: produce followed over time until spoilage

*2026-10-20. Rule: public or provided data only; no new collection. Only datasets that track the **same item** across visits can train the time model. Single-snapshot sets are labelled **encoder pretraining only**.*

## A. Longitudinal (same item tracked) — usable for the time model

| Dataset | Link | License | Items | Same item tracked? | Interval | Storage recorded? | Labels | Notes |
|---|---|---|---|---|---|---|---|---|
| **'Hass' Avocado Ripening Photographic Dataset** (Xavier, Rodrigues, Silva 2024) | doi:10.17632/3xd9n945v8.1 — https://data.mendeley.com/datasets/3xd9n945v8/1 | **CC BY 4.0** | 478 avocados, 14,710 photos (800×800 JPEG) | **Yes** — sample ID, side a/b | Daily, two sides | **Yes**: T10 10 °C/85 % RH (192), T20 20 °C/85 % RH (143), Tamb 15.8–21.7 °C mean 18.7 °C, RH unlogged (143) | 5-stage and 10-stage Ripening Index per photo; Days Left derivable; spreadsheet columns: filename, sample, group, stage, day, side | **Primary.** Right-censoring 9.1–26.6 % per group (Hangud 2026). Single origin, single harvest; fruit was over-mature (DM 31–39 %) with wind-rub and sunburn marks. |
| **FreshLab-Tomato: A Longitudinal Shelf-Life Dataset** (Njoku et al., DaintyMindz Laboratory) | doi:10.21227/wvqh-5c80 — https://ieee-dataport.org/documents/freshlab-tomato-longitudinal-shelf-life-dataset | **Not shown on page** — IEEE DataPort; **login required**, listed as subscription access | 12 tomatoes (4 per condition), 1,080 images | **Yes** — sample ID | 3 capture periods per day × 3 views (top, left, right), 10 days | **Yes**: ambient / refrigerated / outdoor; **temperature and RH in metadata** | 5-point ordinal spoilage score (1 fresh → 5 severe) per observation | The only other set with logged temperature. Tiny (12 items) — enough for a κ(T) transfer *test*, not for training. Needs access confirmation and license before use. |
| **TriModal Ripeness 6** (Unnikrishnan, Deepak, Aishwaryaa, Bagyammal 2026; Data in Brief doi:10.1016/j.dib.2026.112545) | doi:10.6084/m9.figshare.30783827 — https://figshare.com/articles/dataset/30783827 | **CC BY 4.0** | 6 types — banana, carrot, guava, Indian gooseberry, mango, tomato; "10,000+" RGB and "10,000+" thermal images; item counts not stated | **Stated** as monitored daily until spoilage with 17 thermal views per fruit per day; per-item folder structure **unverified** (single 20.4 GB zip, not downloaded) | 3 sessions per day (morning/afternoon/evening), 25 Jun–3 Oct 2025 | **No** — ambient T/RH not logged | Normal vs Classified (Spoiled / Not_spoiled); methane readings | Best candidate for **multi-produce**. Must download and inspect folder naming to confirm item IDs and session timestamps before counting on it. 3000×4000 sRGB, 160×120 thermal. |
| **banana_ripening_dataset_day0_to_day7** (Sangolgi, Narode, Atre; SPPU Pune) | doi:10.17632/d5tczj7fs7.2 — https://data.mendeley.com/datasets/d5tczj7fs7/2 | **CC BY 4.0** | ~30 bananas, ~240 images | **Yes** — `Banana_ID_xxx/day0..day7.jpg` | Daily, 8 days | "Controlled and consistent" — **values not stated** | None per image; authors suggest day-based classes | Small, clean, per-item. No spoilage endpoint (stops at day 7, "ripe"); usable for the latent clock but gives no first-passage label unless day 7 is taken as the boundary. |

## B. Snapshot — encoder pretraining only

| Dataset | Link | License | Size | Why not for the time model |
|---|---|---|---|---|
| Comprehensive dataset on ripening stages of strawberries and avocados (Kamat et al. 2025, Data in Brief doi:10.1016/j.dib.2025.111663) | Mendeley (DOI in article; not opened) | CC BY 4.0 (article) | 1,333 original / 14,630 augmented, 4 classes, 25 days of daily photos | Per-item IDs not described in the article. Re-classify to group A if the files carry item IDs. |
| Fruits-360 (Muresan & Oltean 2018) | https://github.com/Horea94/Fruit-Images-Dataset | MIT | 90,483 images, 131 classes, 100×100 | Rotation snapshots of one fruit; no time axis. |
| Fresh and Stale Images of Fruits and Vegetables (Kaggle, raghavrpotdar) | https://www.kaggle.com/datasets/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables | not checked | — | Binary fresh/stale snapshots. |
| Fruit and vegetable dataset for Shelf life (Kaggle, soorajkavumpadi) | https://www.kaggle.com/datasets/soorajkavumpadi/fruit-and-vegetable-dataset-for-shelf-life | not checked | apple, tomato, banana, carrot — page description only | Kaggle page not readable without login; no evidence of per-item tracking. |
| Nuanmeesri 2025 avocado set (Sci Rep doi:10.1038/s41598-025-87173-7) | not located as a public download | — | 1,000 raw images, 5 classes | Classification only; availability unconfirmed. |

## C. Not found

- **Stevens Institute dataset** (items photographed about every other day until spoilage): no public listing found under "Stevens Institute" + produce spoilage. Treat as *provided* data; the loader interface in `src/ripen/data/stevens.py` will be a stub against a manifest schema (item_id, produce_type, visit_time, image_path, storage fields, spoiled flag) until the files arrive.
- No public longitudinal set was found for apple, mango-alone, or leafy greens with per-item tracking beyond TriModal.

## D. Recommendation

1. **Avocado (Mendeley)** is the training set. Download now: CC BY 4.0, 478 items, three temperatures, daily, per-item.
2. **TriModal Ripeness 6** is the multi-produce extension — but it is a single 20 GB zip and the per-item structure is unverified. First action on a machine with disk: download, list the tree, and confirm `item → session → image` naming. If confirmed, it supports leave-one-produce-out (direction 2) without temperature; if not, it drops to group B.
3. **FreshLab-Tomato** is the κ(T) *test* set (logged T/RH, three regimes) if the license and access clear. 12 items is a test, not a training set.
4. **Banana day0–7** is a cheap sanity set for the latent clock on a second produce type.
5. The **Stevens** set gets the same manifest schema so it plugs in when it arrives.

Every loader emits the same record: `(item_id, produce, t_visit [days], image paths, T [°C] or NaN, RH or NaN, stage or NaN, spoiled_by_t, censor_type ∈ {exact, interval, right})`.
