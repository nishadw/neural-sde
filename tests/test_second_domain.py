"""PBC2 and C-MAPSS loaders on synthetic fixtures in the real file formats, plus optional real-data checks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from ripen.data import collate, dataset_info, load_cmapss, load_pbc2
from ripen.data.cmapss import FEATURE_COLUMNS, visit_cycles
from ripen.data.pbc2 import BASELINE, LONGITUDINAL, MONTHS_PER_UNIT
from ripen.pipeline import load_items, subsample_visits_known_event

REPO = Path(__file__).resolve().parents[1]
PBC2_CSV = REPO / "data" / "pbc2" / "pbc2_cleaned.csv"
CMAPSS_ROOT = REPO / "data" / "cmapss"


# --------------------------------------------------------------------------- fixtures in the real formats


def _pbc2_csv(tmp_path: Path) -> Path:
    """Six patients covering every event/visit pattern. Times are in the file's unit (weeks)."""
    rng = np.random.default_rng(0)
    patients = [
        # id, tte, label, visit times (weeks)
        (1, 57, 1, [0, 28]),  # death, visits strictly before T
        (2, 100, 1, [0, 50, 100]),  # death with the last visit on the event day -> that visit is dropped
        (3, 300, 2, [0, 52, 104, 200]),  # transplant: censored at T (or the event with transplant_as_event)
        (4, 500, 0, [0, 52, 104, 156]),  # censored at tte, later than the last visit
        (5, 20, 1, [0]),  # single-visit death
        (6, 400, 0, [0, 100, 200, 300, 400]),  # censored on the last visit day
    ]
    rows = []
    for pid, tte, label, times in patients:
        drug, sex, age = pid % 2, (pid == 3) * 1, 40.0 + pid
        for t in times:
            r = {
                "id": pid,
                "tte": float(tte),
                "times": float(t),
                "label": float(label),
                "drug": drug,
                "age": age,
                "sex": sex,
            }
            for c in LONGITUDINAL:
                r[c] = float(rng.uniform(0.5, 5.0))
            r["edema"] = int(rng.integers(0, 3))
            r["histologic"] = int(rng.integers(1, 5))
            if t > 0 and rng.uniform() < 0.3:
                r["serChol"] = np.nan  # the real file has many missing cholesterol values
            rows.append(r)
    rows[0]["ascites"] = np.nan  # a missing value at a patient's first visit (no LOCF source)
    cols = ["id", "tte", "times", "label", "drug", "age", "sex", *LONGITUDINAL]
    cols = list(dict.fromkeys(cols))
    df = pd.DataFrame(rows)[cols]
    df = df.sample(frac=1.0, random_state=0)  # the loader must sort
    path = tmp_path / "pbc2_cleaned.csv"
    df.to_csv(path, index=False)
    return path


def _cmapss_root(tmp_path: Path, lengths: dict[str, list[int]]) -> Path:
    """Write ``train_<subset>.txt`` files with the real 26-column whitespace layout."""
    rng = np.random.default_rng(1)
    for subset, lens in lengths.items():
        lines = []
        for unit, L in enumerate(lens, start=1):
            for c in range(1, L + 1):
                settings = [rng.normal(0, 0.002), rng.normal(0, 0.0003), 100.0]  # setting3 constant like FD001
                sensors = [518.67] + list(641 + 0.05 * c + rng.normal(0, 0.5, 20))  # s1 constant like FD001
                vals = [unit, c, *settings, *sensors]
                lines.append(" ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals) + " ")
        (tmp_path / f"train_{subset}.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


# --------------------------------------------------------------------------- PBC2


def test_pbc2_items_intervals_and_features(tmp_path):
    items = load_pbc2(_pbc2_csv(tmp_path))
    assert [it.item_id for it in items] == [f"pbc2-{i:03d}" for i in range(1, 7)]
    by = {it.meta["id"]: it for it in items}
    D = len(LONGITUDINAL) + len(BASELINE)
    for it in items:
        assert it.produce == "pbc2" and math.isnan(it.temp_c)
        assert it.storage in ("placebo", "D-penicil")
        ts = [v.t for v in it.visits]
        assert ts[0] == 0.0 and ts == sorted(ts)
        assert all(v.stage is None and v.features.shape == (1, D) and v.features.dtype == np.float32 for v in it.visits)
        assert all(np.isfinite(v.features).all() for v in it.visits)
        assert it.spoil_lo <= it.spoil_hi
        assert it.meta["feature_names"] == [*LONGITUDINAL, *BASELINE]
        assert len(it.meta["feature_mean"]) == D == len(it.meta["feature_std"])
    m = MONTHS_PER_UNIT
    # death, visits before T: narrow interval (last visit, T]
    it = by[1]
    assert not it.right_censored and it.censor_type == "interval"
    assert it.spoil_lo == pytest.approx(28 * m) and it.spoil_hi == pytest.approx(57 * m) and it.n_visits == 2
    # death with a visit on the event day: that visit is dropped, lo is the previous one
    it = by[2]
    assert it.n_visits == 2 and it.meta["n_visits_dropped_at_event"] == 1
    assert it.spoil_lo == pytest.approx(50 * m) and it.spoil_hi == pytest.approx(100 * m)
    # transplant: right-censored at the transplant time, flagged competing
    it = by[3]
    assert it.right_censored and it.meta["competing"] and it.meta["event"] == "transplant"
    assert it.spoil_lo == pytest.approx(300 * m) and it.n_visits == 4
    # censored later than the last visit: lo is the censoring time, not the last visit
    it = by[4]
    assert it.right_censored and not it.meta["competing"]
    assert it.spoil_lo == pytest.approx(500 * m) and it.meta["t_last_visit"] == pytest.approx(156 * m)
    # single-visit death: lo = 0 (the only visit), hi = T
    it = by[5]
    assert it.n_visits == 1 and it.spoil_lo == 0.0 and it.spoil_hi == pytest.approx(20 * m)
    # censored on the last visit day
    it = by[6]
    assert it.right_censored and it.spoil_lo == pytest.approx(400 * m) == it.t_last
    # the arm is read from the drug column
    assert by[1].storage == "D-penicil" and by[2].storage == "placebo"


def test_pbc2_transplant_as_event(tmp_path):
    items = load_pbc2(_pbc2_csv(tmp_path), transplant_as_event=True)
    it = next(x for x in items if x.meta["id"] == 3)
    assert not it.right_censored and it.meta["competing"] and it.meta["transplant_as_event"]
    assert it.spoil_lo == pytest.approx(200 * MONTHS_PER_UNIT) and it.spoil_hi == pytest.approx(300 * MONTHS_PER_UNIT)
    assert sum(not x.right_censored for x in items) == 4


def test_pbc2_collate_and_pipeline(tmp_path):
    path = _pbc2_csv(tmp_path)
    items = load_pbc2(path)
    info = dataset_info(items)
    assert info.feat_dim == len(LONGITUDINAL) + len(BASELINE) and info.produce == ("pbc2",)
    assert set(info.groups) == {"placebo", "D-penicil"}
    b = collate(items, info, n_obs="all")
    assert b.feat.shape == (6, info.max_visits, info.feat_dim)
    assert int(b.right_censored.sum()) == 3
    assert np.all(
        np.asarray(b.lo_rel)[~np.asarray(b.right_censored)] <= np.asarray(b.hi_rel)[~np.asarray(b.right_censored)]
    )
    cfg = OmegaConf.create({"name": "pbc2", "path": str(path), "subsample_every": 2, "subsample_offset": 0})
    sub = load_items(cfg)
    full = {it.item_id: it for it in items}
    for it in sub:
        f = full[it.item_id]
        assert it.n_visits == len([i for i in range(f.n_visits) if i == 0 or i % 2 == 0])
        assert it.spoil_hi == f.spoil_hi  # the registry event time never moves
        if not it.right_censored:
            assert it.spoil_lo <= f.spoil_lo and it.spoil_lo < it.spoil_hi
        else:
            assert it.spoil_lo == f.spoil_lo
    # the dropped-at-event rule: patient 2 keeps visits [0, 50] -> after subsampling every 2 keeps [0] only
    p2 = next(it for it in sub if it.meta["id"] == 2)
    assert p2.n_visits == 1 and p2.spoil_lo == 0.0


def test_pbc2_rejects_bad_files(tmp_path):
    df = pd.read_csv(_pbc2_csv(tmp_path))
    bad = tmp_path / "bad.csv"
    df.drop(columns=["serBilir"]).to_csv(bad, index=False)
    with pytest.raises(KeyError):
        load_pbc2(bad)
    df2 = df.copy()
    df2.loc[df2["id"] == 1, "times"] += 5.0  # first visit no longer at 0
    df2.to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_pbc2(bad)


# --------------------------------------------------------------------------- C-MAPSS


def test_visit_cycles_schedule():
    rng = np.random.default_rng(0)
    for L in (31, 128, 362):
        vc = visit_cycles(L, 10, 2, rng)
        assert vc[0] == 1 and all(b > a for a, b in zip(vc, vc[1:], strict=False)) and vc[-1] < L
        gaps = np.diff(vc)
        assert gaps.min() >= 6 and gaps.max() <= 14  # 10 +- 2*jitter
        assert abs(len(vc) - L / 10) <= 2
    assert visit_cycles(50, 10, 0, rng) == [1, 11, 21, 31, 41]


def test_cmapss_items_from_text_fixture(tmp_path):
    root = _cmapss_root(tmp_path, {"FD001": [40, 55, 63], "FD002": [45, 80]})
    items = load_cmapss(root, subsets=("FD001",), inspect_every=10, jitter=2, seed=0)
    assert [it.item_id for it in items] == ["FD001-001", "FD001-002", "FD001-003"]
    for it, L in zip(items, [40, 55, 63], strict=True):
        assert it.produce == "turbofan" and it.storage == "FD001" and math.isnan(it.temp_c)
        ts = [v.t for v in it.visits]
        assert ts[0] == 0.0 and ts == sorted(ts)
        assert it.meta["failure_cycle"] == L and it.meta["visit_cycles"][-1] < L
        assert not it.right_censored and it.censor_type == "interval"
        assert it.spoil_lo == it.t_last and it.spoil_hi == pytest.approx((L - 1) / 10)
        assert it.spoil_lo < it.spoil_hi
        assert all(v.stage is None and v.features.shape == (1, len(it.meta["feature_names"])) for v in it.visits)
        assert all(v.features.dtype == np.float32 for v in it.visits)
        # visits are ~1 unit apart
        assert np.all(np.diff(ts) >= 0.6) and np.all(np.diff(ts) <= 1.4)
    # setting3 and s1 are constant in the fixture (as in the real FD001) and must be dropped
    assert set(items[0].meta["dropped_columns"]) == {"setting3", "s1"}
    assert len(items[0].meta["feature_names"]) == len(FEATURE_COLUMNS) - 2
    # features at a visit are the standardised row of that cycle
    feats = np.concatenate([v.features for it in items for v in it.visits])
    assert np.isfinite(feats).all() and feats.std() > 0
    # two subsets -> two storage groups, and nothing is constant across the mix when drop_constant=False
    both = load_cmapss(root, subsets=("FD001", "FD002"), drop_constant=False)
    assert len(both) == 5 and {it.storage for it in both} == {"FD001", "FD002"}
    assert len(both[0].meta["feature_names"]) == len(FEATURE_COLUMNS)
    info = dataset_info(both)
    b = collate(both, info, n_obs="all")
    assert b.feat.shape == (5, info.max_visits, len(FEATURE_COLUMNS)) and info.groups == ("FD001", "FD002")


def test_cmapss_seed_and_drop_tail(tmp_path):
    root = _cmapss_root(tmp_path, {"FD001": [40, 55, 63, 70, 90, 100, 120, 130, 140, 150]})
    a = load_cmapss(root, seed=3)
    b = load_cmapss(root, seed=3)
    c = load_cmapss(root, seed=4)
    assert [it.meta["visit_cycles"] for it in a] == [it.meta["visit_cycles"] for it in b]
    assert [it.meta["visit_cycles"] for it in a] != [it.meta["visit_cycles"] for it in c]
    cens = load_cmapss(root, seed=0, drop_tail=0.3)
    rc = [it for it in cens if it.right_censored]
    assert len(rc) == 3 and len(cens) == 10
    for it in rc:
        C = it.meta["censored_at_cycle"]
        assert 1 <= C < it.meta["failure_cycle"]
        assert it.spoil_lo == pytest.approx((C - 1) / 10) and math.isinf(it.spoil_hi)
        assert it.meta["visit_cycles"][-1] <= C and it.t_last <= it.spoil_lo
    for it in cens:
        if not it.right_censored:
            assert it.meta["censored_at_cycle"] is None
    with pytest.raises(ValueError):
        load_cmapss(root, jitter=10)
    with pytest.raises(ValueError):
        load_cmapss(root, subsets=("FD009",))


def test_cmapss_pipeline_and_subsample(tmp_path):
    root = _cmapss_root(tmp_path, {"FD003": [60, 85]})
    cfg = OmegaConf.create(
        {"name": "cmapss", "root": str(root), "subsets": ["FD003"], "inspect_every": 5, "jitter": 1, "seed": 1}
    )
    items = load_items(cfg)
    assert len(items) == 2 and all(it.storage == "FD003" for it in items)
    assert all(abs(np.diff([v.t for v in it.visits]).mean() - 1.0) < 0.3 for it in items)
    cfg.subsample_every = 3
    sub = load_items(cfg)
    for s, f in zip(sub, items, strict=True):
        assert s.spoil_hi == f.spoil_hi and s.spoil_lo <= f.spoil_lo and s.n_visits < f.n_visits
        assert s.visits[0].t == 0.0
    # the helper itself on a right-censored item leaves the censoring time alone
    cen = load_cmapss(root, subsets=("FD003",), drop_tail=1.0, inspect_every=5, jitter=1)
    for it in cen:
        s = subsample_visits_known_event(it, 2)
        assert s.right_censored and s.spoil_lo == it.spoil_lo and s.n_visits <= it.n_visits


# --------------------------------------------------------------------------- real files, if present


@pytest.mark.skipif(not PBC2_CSV.exists(), reason="data/pbc2/pbc2_cleaned.csv not downloaded")
def test_pbc2_real_counts():
    items = load_pbc2(PBC2_CSV)
    assert len(items) == 312
    events = sum(not it.right_censored for it in items)
    competing = sum(it.meta["competing"] for it in items)
    assert events == 140 and competing == 29 and len(items) - events == 172
    assert sum(it.n_visits for it in items) == 1945 - sum(it.meta["n_visits_dropped_at_event"] for it in items)
    assert all(it.visits[0].t == 0.0 for it in items)
    assert max(it.spoil_hi for it in items if not it.right_censored) < 180
    assert max(it.spoil_lo for it in items) < 180
    print(
        f"PBC2: {len(items)} patients, {events} deaths, {competing} transplants (censored), "
        f"{sum(it.n_visits for it in items)} visits, D={items[0].visits[0].features.shape[1]}"
    )
    comp = load_pbc2(PBC2_CSV, transplant_as_event=True)
    assert sum(not it.right_censored for it in comp) == 169


@pytest.mark.skipif(not (CMAPSS_ROOT / "train_FD001.txt").exists(), reason="data/cmapss not downloaded")
def test_cmapss_real_counts():
    items = load_cmapss(CMAPSS_ROOT, subsets=("FD001",))
    assert len(items) == 100 and all(not it.right_censored for it in items)
    assert set(items[0].meta["dropped_columns"]) == {"setting3", "s1", "s5", "s10", "s16", "s18", "s19"}
    assert len(items[0].meta["feature_names"]) == 17
    L = [it.meta["failure_cycle"] for it in items]
    assert min(L) == 128 and max(L) == 362
    assert all(it.spoil_lo < it.spoil_hi and it.visits[0].t == 0.0 for it in items)
    print(f"C-MAPSS FD001: {len(items)} engines, {sum(it.n_visits for it in items)} visits, cycles {min(L)}-{max(L)}")
    for subset, n in (("FD002", 260), ("FD003", 100), ("FD004", 249)):
        if (CMAPSS_ROOT / f"train_{subset}.txt").exists():
            assert len(load_cmapss(CMAPSS_ROOT, subsets=(subset,))) == n
