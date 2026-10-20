import math

import numpy as np
import pandas as pd
import pytest

from ripen.data import collate, dataset_info, leave_one_group_out, load_splits, make_splits, save_splits, select
from ripen.data.avocado import GROUP_TEMPS_C, find_columns, load_avocado
from ripen.data.features import HashFeaturizer
from ripen.data.stevens import export_manifest, load_manifest


def _fake_avocado(tmp_path, n_fruit=6):
    rows = []
    rng = np.random.default_rng(0)
    for s in range(n_fruit):
        group = ["T10", "T20", "Tamb"][s % 3]
        days = 15 if group == "T10" else 6  # T10 reaches stage 5 on day 12, the others on day 4
        for d in range(days):
            stage = min(5, 1 + d // (3 if group == "T10" else 1))
            if s == 0 and stage == 5:
                stage = 4  # fruit 0 never spoils -> right-censored
            for side in "ab":
                rows.append(
                    {
                        "Photo": f"{group}_{s}_{d}_{side}.jpg",
                        "Sample": s,
                        "Storage Group": group,
                        "Ripening Index": stage,
                        "Day": d,
                        "Side": side,
                    }
                )
            if stage == 5:
                break
    df = pd.DataFrame(rows)
    rng.shuffle(df.values)
    path = tmp_path / "labels.xlsx"
    df.to_excel(path, index=False)
    return path, df


def test_find_columns_by_keyword():
    cols = find_columns(["Photo name", "Sample Number", "Storage Group", "Ripening Index (5)", "Day", "Side"])
    assert set(cols) == {"filename", "sample", "group", "stage", "day", "side"}
    with pytest.raises(KeyError):
        find_columns(["a", "b"])


def test_avocado_loader_builds_items_with_intervals(tmp_path):
    path, df = _fake_avocado(tmp_path)
    feats = HashFeaturizer(16)(list(df["Photo"]))
    items = load_avocado(tmp_path, path, features=feats)
    assert len(items) == 6
    for it in items:
        assert it.produce == "avocado" and it.temp_c == GROUP_TEMPS_C[it.storage]
        assert it.visits[0].t == 0.0
        assert all(v.features is not None and v.features.shape == (2, 16) for v in it.visits)
        if it.item_id.endswith("-0"):
            assert it.right_censored and math.isinf(it.spoil_hi)
        else:
            assert it.visits[-1].stage == 5 and it.spoil_hi == it.t_last and it.spoil_lo == it.visits[-2].t
    info = dataset_info(items)
    assert info.feat_dim == 16 and info.groups == ("T10", "T20", "Tamb")
    b = collate(items, info, n_obs="all")
    assert b.feat.shape == (6, info.max_visits, 16)


def test_stevens_manifest_roundtrip(tmp_path):
    path, _ = _fake_avocado(tmp_path)
    items = load_avocado(tmp_path, path)
    export_manifest(items, tmp_path / "manifest.csv")
    back = load_manifest(tmp_path / "manifest.csv")
    assert [it.item_id for it in back] == sorted(it.item_id for it in items)
    for a in items:
        b = next(x for x in back if x.item_id == a.item_id)
        assert b.n_visits == a.n_visits and b.spoil_lo == a.spoil_lo and b.spoil_hi == a.spoil_hi
        assert b.storage == a.storage and b.temp_c == a.temp_c


def test_splits_are_by_item_and_disjoint(synthetic_items, tmp_path):
    sp = make_splits(synthetic_items, seed=0)
    all_ids = sp["train"] + sp["val"] + sp["test"]
    assert len(all_ids) == len(set(all_ids)) == len(synthetic_items)
    assert len(sp["train"]) > len(sp["val"]) > 0 and len(sp["test"]) > 0
    assert sp == make_splits(synthetic_items, seed=0)  # deterministic
    save_splits(sp, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == sp
    logo = leave_one_group_out(synthetic_items, "T10")
    assert all(it.storage == "T10" for it in select(synthetic_items, logo["test"]))
    assert all(it.storage != "T10" for it in select(synthetic_items, logo["train"] + logo["val"]))
