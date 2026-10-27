"""Loader for the 'Hass' Avocado Ripening Photographic Dataset (Mendeley, doi:10.17632/3xd9n945v8.1, CC BY 4.0).

Expected layout under ``root``::

    root/
      <labels>.xlsx        one row per photo: filename, sample, storage group, stage (1-5), day, side (a/b)
      images/...           the 800x800 JPEGs named as in the spreadsheet (``root`` itself if no ``images/``)

Column names are matched by keyword (see ``COLUMN_KEYS``) so other exports still load. The released
spreadsheet (sheet ``DATABASE``) has ``File Name, Time Stamp, Storage Group, Sample, Day of Experiment,
Ripening Index Classification``: file names carry no ``.jpg`` and encode the side (``T20_d01_001_a_1``),
and the ambient group is spelled ``Tam``. 12 of its 14,722 rows have no photo in the release; those
visits keep their labels and get features only from the photos that exist. Two sides are photographed
per fruit per day; the stage is a per-sample rating, so a visit's stage is
the max over its sides when they disagree (recorded in ``meta``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Item, Visit, spoil_interval_from_visits

GROUP_TEMPS_C = {"T10": 10.0, "T20": 20.0, "Tamb": 18.7}
GROUP_RH = {"T10": 85.0, "T20": 85.0, "Tamb": math.nan}

COLUMN_KEYS = {
    "filename": ("file", "photo", "image", "name"),
    "sample": ("sample", "fruit", "id"),
    "group": ("group", "storage", "condition"),
    "stage": ("stage", "index", "ripen", "class"),
    "day": ("day", "date"),
    "side": ("side", "position"),
}


def find_columns(columns: list[str]) -> dict[str, str]:
    """Map our field names to spreadsheet columns by keyword; raise if a required one is missing."""
    lowered = {c: c.lower() for c in columns}
    out: dict[str, str] = {}
    for field, keys in COLUMN_KEYS.items():
        for c, lc in lowered.items():
            if any(k in lc for k in keys) and c not in out.values():
                out[field] = c
                break
    missing = [f for f in ("filename", "sample", "group", "stage", "day") if f not in out]
    if missing:
        raise KeyError(f"could not find columns for {missing} among {columns}")
    return out


def normalise_group(g: str) -> str:
    s = str(g).strip().replace(" ", "")
    for name in GROUP_TEMPS_C:
        if s.lower() == name.lower():
            return name
    if "10" in s:
        return "T10"
    if "20" in s:
        return "T20"
    if s.lower().startswith("tam") or "amb" in s.lower():
        return "Tamb"
    raise ValueError(f"unknown storage group {g!r}")


def read_labels(path: str | Path) -> pd.DataFrame:
    """Read the spreadsheet (xlsx or csv) into a frame with canonical columns."""
    path = Path(path)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    cols = find_columns(list(df.columns))
    names = df[cols["filename"]].astype(str).str.strip()
    names = names.where(names.str.contains(r"\.\w{3,4}$"), names + ".jpg")
    if "side" in cols:
        side = df[cols["side"]].astype(str).str.strip()
    else:  # side from file names like T20_d01_001_a_1
        side = names.str.extract(r"_([ab])_\d+\.\w+$", expand=False).fillna("a")
    out = pd.DataFrame(
        {
            "filename": names,
            "sample": df[cols["sample"]].astype(str).str.strip(),
            "group": df[cols["group"]].map(normalise_group),
            "stage": df[cols["stage"]].astype(int),
            "day": df[cols["day"]],
            "side": side,
        }
    )
    if np.issubdtype(out["day"].dtype, np.datetime64):
        out["day"] = (out["day"] - out.groupby("sample")["day"].transform("min")).dt.days
    out["day"] = out["day"].astype(float)
    return out


def load_avocado(
    root: str | Path,
    labels_file: str | Path | None = None,
    images_dir: str | Path | None = None,
    features: dict[str, np.ndarray] | None = None,
) -> list[Item]:
    """Build one :class:`Item` per fruit.

    Args:
        root: dataset directory.
        labels_file: spreadsheet path; default is the first ``*.xlsx`` under ``root``.
        images_dir: where the JPEGs are; default ``root/images`` if it exists, else ``root``. Paths are recorded, not opened.
        features: optional ``{filename: (D,) array}`` from :func:`ripen.data.features.load_feature_cache`.
    """
    root = Path(root)
    if labels_file is None:
        found = sorted(root.rglob("*.xlsx"))
        if not found:
            raise FileNotFoundError(f"no .xlsx under {root}")
        labels_file = found[0]
    if images_dir is None:
        images_dir = root / "images" if (root / "images").is_dir() else root
    images_dir = Path(images_dir)
    df = read_labels(labels_file)
    items: list[Item] = []
    for (sample, group), rows in df.groupby(["sample", "group"], sort=True):
        rows = rows.sort_values(["day", "side"])
        day0 = float(rows["day"].min())
        visits: list[Visit] = []
        side_stages: dict[float, list[int]] = defaultdict(list)
        by_day = defaultdict(list)
        for r in rows.itertuples(index=False):
            by_day[float(r.day)].append(r)
            side_stages[float(r.day)].append(int(r.stage))
        for day in sorted(by_day):
            rs = by_day[day]
            paths = tuple(str(images_dir / r.filename) for r in rs)
            feats = None
            if features is not None:
                vecs = [features[r.filename] for r in rs if r.filename in features]
                feats = np.stack(vecs).astype(np.float32) if vecs else None
            visits.append(Visit(t=day - day0, stage=max(side_stages[day]), image_paths=paths, features=feats))
            if visits[-1].stage is not None and visits[-1].stage >= 5:
                break
        lo, hi = spoil_interval_from_visits(visits)
        disagreements = sum(1 for d, s in side_stages.items() if len(set(s)) > 1)
        items.append(
            Item(
                item_id=f"{group}-{sample}",
                produce="avocado",
                storage=group,
                temp_c=GROUP_TEMPS_C[group],
                rh=GROUP_RH[group],
                visits=visits,
                spoil_lo=lo,
                spoil_hi=hi,
                meta={"sample": str(sample), "side_disagreements": disagreements, "labels_file": str(labels_file)},
            )
        )
    return items
