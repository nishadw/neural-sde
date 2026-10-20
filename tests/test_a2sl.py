"""A2SL loaders: a tiny synthetic manifest (so CI needs no raw data), plus the committed manifests when present."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ripen.data import collate, dataset_info
from ripen.data.a2sl import SCAN_RADIAL_BINS, load_a2sl_avocado, load_a2sl_pilot, load_a2sl_watermelon, scan_summary


def _disk_grid(seed: int, n: int = 75) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    r = np.sqrt((yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    g = np.exp(-((r / 12) ** 2)) * 1e4 + rng.uniform(50, 500, (n, n))
    g[r > n / 2.2] = np.nan
    return g.astype(np.float32)


def test_scan_summary_shape_and_contrast():
    f = scan_summary(_disk_grid(0))
    assert f.shape == (4 + SCAN_RADIAL_BINS,)
    assert np.isfinite(f).all() and f[3] > 0  # bright centre -> positive centre-to-rim contrast
    assert scan_summary(np.full((5, 5), np.nan)).sum() == 0


def _write_fake_manifests(tmp: Path):
    man = tmp / "manifests"
    man.mkdir()
    pd.DataFrame(
        [
            {
                "item_id": "avo01",
                "avocado": 1,
                "cut_day": np.nan,
                "tracked_to_end": True,
                "mass_g": 230,
                "length_mm": 100,
                "diameter_mm": 70,
            },
            {
                "item_id": "avo04",
                "avocado": 4,
                "cut_day": 1,
                "tracked_to_end": False,
                "mass_g": 238,
                "length_mm": 89,
                "diameter_mm": 72,
            },
        ]
    ).to_csv(man / "a2sl_avocado_items.csv", index=False)
    rows = []
    for day in (1, 3, 5):
        rows.append(
            {
                "item_id": "avo01",
                "avocado": 1,
                "day": day,
                "date": "",
                "visit_index": 1,
                "side_photo": f"p/day{day}/01_side.jpg",
                "side_photo_heic": "",
                "cut_photo": "",
                "cut_today": False,
                "scan_keys": f"a1_d{day}_r0;a1_d{day}_r60",
                "scan_pngs": "",
            }
        )
    rows.append(
        {
            "item_id": "avo04",
            "avocado": 4,
            "day": 1,
            "date": "",
            "visit_index": 1,
            "side_photo": "p/day1/04_side.jpg",
            "side_photo_heic": "",
            "cut_photo": "p/day1/04_cut.jpg",
            "cut_today": True,
            "scan_keys": "a4_d1_r0",
            "scan_pngs": "",
        }
    )
    pd.DataFrame(rows).to_csv(man / "a2sl_avocado_visits.csv", index=False)
    prow = []
    for day in (1, 2, 3):
        for v in ("01", "02"):
            prow.append(
                {
                    "item_id": "pilot01",
                    "avocado": 1,
                    "day": day,
                    "view": v,
                    "kind": "photo",
                    "photo": f"q/{day}_{v}.jpg",
                    "cut_day": 13,
                }
            )
    prow.append(
        {
            "item_id": "pilot01",
            "avocado": 1,
            "day": 3,
            "view": "scan_r0",
            "kind": "scan_png",
            "photo": "q/s.png",
            "cut_day": 13,
        }
    )
    pd.DataFrame(prow).to_csv(man / "a2sl_pilot_visits.csv", index=False)
    wrow = [
        {
            "item_id": "wm03",
            "sample": 3,
            "height": "single",
            "rotation_deg": 0,
            "freq_ghz": f,
            "scan_key": f"s3_single_r0_f{f:g}",
            "csv": "",
            "place": "x",
            "ripeness": "1.5-2 weeks out",
            "purchase_date": "",
            "length_cm": "",
            "width_cm": "",
            "origin": "",
            "type": "Crimson",
            "size": "Small",
            "weight_kg": "",
        }
        for f in (3.0, 3.5)
    ]
    pd.DataFrame(wrow).to_csv(man / "a2sl_watermelon.csv", index=False)
    scans = {
        k: _disk_grid(i)
        for i, k in enumerate(
            [
                "a1_d1_r0",
                "a1_d1_r60",
                "a1_d3_r0",
                "a1_d3_r60",
                "a1_d5_r0",
                "a1_d5_r60",
                "a4_d1_r0",
                "s3_single_r0_f3",
                "s3_single_r0_f3.5",
            ]
        )
    }
    feats = {
        "01_side.jpg": np.ones(8, np.float32),
        "04_side.jpg": np.zeros(8, np.float32),
        "1_01.jpg": np.ones(8, np.float32),
    }
    return man, scans, feats


def test_loaders_on_fake_manifests(tmp_path):
    man, scans, feats = _write_fake_manifests(tmp_path)
    items = load_a2sl_avocado(man, features=feats, scans=scans, photo_dim=8)
    assert [it.item_id for it in items] == ["avo01", "avo04"]
    a1, a4 = items
    assert a1.n_visits == 3 and a1.right_censored and a1.spoil_lo == 4.0 and a1.meta["tracked_to_end"]
    assert a4.n_visits == 1 and a4.meta["cut_day"] == 1 and a4.meta["cut_photo"] == ["p/day1/04_cut.jpg"]
    assert a1.visits[0].features.shape == (1, 8 + 2 * (4 + SCAN_RADIAL_BINS))
    info = dataset_info(items)
    b = collate(items, info, n_obs="all")
    assert b.feat.shape[-1] == info.feat_dim and bool(b.right_censored.all())
    pilot = load_a2sl_pilot(man, features=feats)
    assert pilot[0].n_visits == 3 and pilot[0].meta["cut_day"] == 13 and 3 in pilot[0].meta["scan_pngs"]
    wm = load_a2sl_watermelon(man, scans=scans)
    assert wm[0].produce == "watermelon" and wm[0].meta["weeks_out"] == 1.75 and wm[0].meta["n_grids"] == 2


@pytest.mark.skipif(not Path("data/manifests/a2sl_avocado_visits.csv").exists(), reason="A2SL manifests not present")
def test_committed_manifests_parse():
    items = load_a2sl_avocado()
    assert len(items) == 15
    assert sum(it.meta["tracked_to_end"] for it in items) == 4
    assert sorted(it.meta["cut_day"] for it in items if it.meta["cut_day"]) == [1, 3, 5, 7, 9, 11, 13]
    assert all(it.n_visits == 7 for it in items if it.meta["tracked_to_end"])
    pilot = load_a2sl_pilot()
    assert len(pilot) == 6
    wm = load_a2sl_watermelon()
    assert len(wm) == 7 and all(not np.isnan(it.meta["weeks_out"]) for it in wm)
