"""Stub loader for the Stevens Institute produce dataset (not yet in hand).

What we know: each item is photographed about every other day until it
spoils. Produce types, counts and storage conditions are unknown. Until the
files arrive, this loader reads a **manifest CSV** with one row per photo and
the columns below, so the rest of the pipeline can be exercised on any data
that fits the schema (the avocado set can be exported to it).

Manifest columns::

    item_id, produce, visit_time_days, image_path, storage [optional], temp_c [optional],
    rh [optional], stage [optional, 1-5], spoiled [0/1]

``spoiled`` is per visit; the spoilage interval is derived from it (first
visit flagged spoiled ends the interval; otherwise right-censored at the last visit).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import SPOILED_STAGE, Item, Visit

REQUIRED = ("item_id", "produce", "visit_time_days", "image_path", "spoiled")


def load_manifest(path: str | Path, features: dict[str, np.ndarray] | None = None) -> list[Item]:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"manifest {path} is missing columns {missing}")
    items: list[Item] = []
    for item_id, rows in df.groupby("item_id", sort=True):
        rows = rows.sort_values("visit_time_days")
        t0 = float(rows["visit_time_days"].min())
        by_t: dict[float, list] = defaultdict(list)
        for r in rows.itertuples(index=False):
            by_t[float(r.visit_time_days)].append(r)
        visits: list[Visit] = []
        lo, hi = 0.0, math.inf
        for t in sorted(by_t):
            rs = by_t[t]
            spoiled = any(int(r.spoiled) == 1 for r in rs)
            stage_vals = [int(r.stage) for r in rs if "stage" in df.columns and not pd.isna(r.stage)]
            stage = SPOILED_STAGE if spoiled else (max(stage_vals) if stage_vals else None)
            feats = None
            if features is not None:
                vecs = [features[Path(r.image_path).name] for r in rs if Path(r.image_path).name in features]
                feats = np.stack(vecs).astype(np.float32) if vecs else None
            visits.append(
                Visit(t=t - t0, stage=stage, image_paths=tuple(str(r.image_path) for r in rs), features=feats)
            )
            if spoiled:
                hi = t - t0
                break
            lo = t - t0
        first = rows.iloc[0]
        items.append(
            Item(
                item_id=str(item_id),
                produce=str(first["produce"]),
                storage=str(first["storage"])
                if "storage" in df.columns and not pd.isna(first["storage"])
                else "unknown",
                temp_c=float(first["temp_c"]) if "temp_c" in df.columns and not pd.isna(first["temp_c"]) else math.nan,
                rh=float(first["rh"]) if "rh" in df.columns and not pd.isna(first["rh"]) else math.nan,
                visits=visits,
                spoil_lo=lo,
                spoil_hi=hi,
            )
        )
    return items


def export_manifest(items: list[Item], path: str | Path) -> None:
    """Write items in the manifest schema (used to round-trip the avocado set through this loader)."""
    rows = []
    for it in items:
        for v in it.visits:
            for p in v.image_paths or ("",):
                rows.append(
                    {
                        "item_id": it.item_id,
                        "produce": it.produce,
                        "visit_time_days": v.t,
                        "image_path": p,
                        "storage": it.storage,
                        "temp_c": it.temp_c,
                        "rh": it.rh,
                        "stage": v.stage,
                        "spoiled": int(v.stage is not None and v.stage >= SPOILED_STAGE),
                    }
                )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
