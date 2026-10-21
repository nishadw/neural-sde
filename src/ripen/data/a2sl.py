"""Loaders for the A2SL (University of Alabama ECE) microwave-imaging drops, read from the committed manifests.

``scripts/ingest_a2sl.py`` turns the raw folders into ``data/manifests/a2sl_*.csv`` plus two ``.npz`` scan
caches. These loaders build :class:`Item` records from the manifests:

* :func:`load_a2sl_avocado` — 15 avocados scanned every other day. Four were followed for all seven visits;
  seven were cut open at one visit each (destructive check, not spoilage). No ripeness stage was recorded, so
  every item is **right-censored at its last visit** and carries ``stage=None``: the set trains the generative
  (feature-decoding) part of the latent SDE and the multimodal encoder, not the spoilage head.
* :func:`load_a2sl_pilot` — 6 avocados photographed daily for 13 days (4 views), cut on staggered days.
* :func:`load_a2sl_watermelon` — 7 watermelons, one visit each, text ripeness label; snapshot data.

Per-visit features are ``[photo features | scan features]``. Photo features come from the usual backbone cache
(``features={filename: vec}``); scan features are the fixed summaries in :func:`scan_summary` (no learning), so
that the same JAX models run unchanged. ``meta`` keeps the raw scan keys for a learned scan encoder later.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Item, Visit, spoil_interval_from_visits

MAN = Path("data/manifests")
SCAN_RADIAL_BINS = 8

# Ripeness classes of Wu et al., IEEE J. Microwaves 6(1):273-284, 2026 (doi:10.1109/JMW.2025.3636171), Fig. 4:
# every image of a given study day carries the same class (0 unripe, 1 barely ripe, 2 ripe, 3 almost overripe,
# 4 overripe), validated by colour, firmness and the fruit cut open that day. Mapped onto the 5-stage index used
# here (1..5), with stage 5 = end of shelf life, so every fruit's spoilage interval is (day 7, day 9].
JMW2026_DAY_STAGE = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 5, 13: 5}


def scan_summary(grid: np.ndarray, n_bins: int = SCAN_RADIAL_BINS) -> np.ndarray:
    """Fixed features of one reconstruction grid (``H x W`` with NaN outside the imaged disk).

    Log-intensity statistics (mean, std, max, centre-to-rim ratio) plus a ``n_bins`` radial profile of the log
    intensity from the centre of mass of the finite region. Returns ``(4 + n_bins,)`` float32.
    """
    g = np.asarray(grid, np.float64)
    m = np.isfinite(g) & (g > 0)
    if m.sum() < 10:
        return np.zeros(4 + n_bins, np.float32)
    lg = np.full_like(g, np.nan)
    lg[m] = np.log(g[m])
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r = r / max(r.max(), 1e-6)
    v = lg[m]
    prof = np.array(
        [
            v[(r >= i / n_bins) & (r < (i + 1) / n_bins)].mean()
            if np.any((r >= i / n_bins) & (r < (i + 1) / n_bins))
            else v.mean()
            for i in range(n_bins)
        ]
    )
    centre = v[r < 0.3].mean() if np.any(r < 0.3) else v.mean()
    rim = v[r > 0.7].mean() if np.any(r > 0.7) else v.mean()
    return np.concatenate([[v.mean(), v.std(), v.max(), centre - rim], prof]).astype(np.float32)


def _scan_features(keys: list[str], scans: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if scans is None or not keys:
        return None
    vecs = [scan_summary(scans[k]) for k in keys if k in scans]
    if not vecs:
        return None
    arr = np.stack(vecs)
    return np.concatenate([arr.mean(0), arr.std(0)]).astype(np.float32)  # mean and spread over rotation angles


def _photo_feature(paths: list[str], features: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if features is None:
        return None
    vecs = [features[Path(p).name] for p in paths if Path(p).name in features]
    return np.stack(vecs).mean(0).astype(np.float32) if vecs else None


def _combine(photo: np.ndarray | None, scan: np.ndarray | None, photo_dim: int, scan_dim: int) -> np.ndarray | None:
    if photo is None and scan is None:
        return None
    p = photo if photo is not None else np.zeros(photo_dim, np.float32)
    s = scan if scan is not None else np.zeros(scan_dim, np.float32)
    return np.concatenate([p, s])[None]


def load_scan_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_a2sl_avocado(
    manifests: str | Path = MAN,
    features: dict[str, np.ndarray] | None = None,
    scans: dict[str, np.ndarray] | None = None,
    photo_dim: int = 512,
    labels: str | None = "jmw2026",
) -> list[Item]:
    """Items from ``a2sl_avocado_visits.csv``; ``features`` keyed by JPEG filename, ``scans`` by scan key.

    ``labels="jmw2026"`` assigns the published day-determined ripeness class to every visit (see
    ``JMW2026_DAY_STAGE``) and derives the spoilage interval from the first stage-5 visit; ``labels=None``
    leaves every visit unlabelled and right-censored at its last visit.
    """
    man = Path(manifests)
    visits = pd.read_csv(man / "a2sl_avocado_visits.csv")
    props = pd.read_csv(man / "a2sl_avocado_items.csv").set_index("item_id")
    scan_dim = 2 * (4 + SCAN_RADIAL_BINS)
    items: list[Item] = []
    for item_id, rows in visits.groupby("item_id", sort=True):
        rows = rows.sort_values("day")
        day0 = float(rows.day.min())
        vs: list[Visit] = []
        for r in rows.itertuples(index=False):
            keys = str(r.scan_keys).split(";") if isinstance(r.scan_keys, str) else []
            photo = _photo_feature([r.side_photo], features) if isinstance(r.side_photo, str) else None
            feat = _combine(photo, _scan_features(keys, scans), photo_dim, scan_dim)
            stage = JMW2026_DAY_STAGE.get(int(r.day)) if labels == "jmw2026" else None
            vs.append(
                Visit(
                    t=float(r.day) - day0,
                    stage=stage,
                    image_paths=(r.side_photo,) if isinstance(r.side_photo, str) else (),
                    features=feat,
                )
            )
            if stage is not None and stage >= 5:
                break
        lo, hi = spoil_interval_from_visits(vs)
        p = props.loc[item_id]
        items.append(
            Item(
                item_id=str(item_id),
                produce="avocado",
                storage="ambient",
                temp_c=math.nan,
                visits=vs,
                spoil_lo=lo,
                spoil_hi=hi,
                meta={
                    "source": "a2sl_avocado_jun2023",
                    "label_source": labels,
                    "cut_day": None if pd.isna(p.cut_day) else int(p.cut_day),
                    "tracked_to_end": bool(p.tracked_to_end),
                    "mass_g": float(p.mass_g),
                    "length_mm": float(p.length_mm),
                    "diameter_mm": float(p.diameter_mm),
                    "scan_keys": [str(r.scan_keys).split(";") for r in rows.itertuples(index=False)],
                    "cut_photo": [r.cut_photo for r in rows.itertuples(index=False) if isinstance(r.cut_photo, str)],
                },
            )
        )
    return items


def load_a2sl_pilot(manifests: str | Path = MAN, features: dict[str, np.ndarray] | None = None) -> list[Item]:
    """Items from ``a2sl_pilot_visits.csv`` (photos only; scan PNGs are kept in ``meta``)."""
    df = pd.read_csv(Path(manifests) / "a2sl_pilot_visits.csv")
    items: list[Item] = []
    for item_id, rows in df.groupby("item_id", sort=True):
        photos = rows[rows.kind == "photo"]
        if photos.empty:
            continue
        day0 = float(photos.day.min())
        vs = []
        for day, g in photos.groupby("day", sort=True):
            paths = tuple(g.photo.tolist())
            vs.append(
                Visit(
                    t=float(day) - day0,
                    stage=None,
                    image_paths=paths,
                    features=_photo_feature(list(paths), features)[None]
                    if _photo_feature(list(paths), features) is not None
                    else None,
                )
            )
        cut_day = rows.cut_day.dropna()
        items.append(
            Item(
                item_id=str(item_id),
                produce="avocado",
                storage="ambient",
                temp_c=math.nan,
                visits=vs,
                spoil_lo=vs[-1].t,
                spoil_hi=math.inf,
                meta={
                    "source": "a2sl_pilot_may2023",
                    "cut_day": int(cut_day.iloc[0]) if len(cut_day) else None,
                    "scan_pngs": rows[rows.kind == "scan_png"].groupby("day").photo.apply(list).to_dict(),
                    "cut_photos": rows[rows.kind == "cut"].photo.tolist(),
                },
            )
        )
    return items


RIPENESS_TEXT_TO_WEEKS_OUT = {
    "ripe": 0.0,
    "ripe*": 0.0,
    "1.5 weeks out*": 1.5,
    "1.5-2 weeks out": 1.75,
    "3 weeks out": 3.0,
}


def load_a2sl_watermelon(manifests: str | Path = MAN, scans: dict[str, np.ndarray] | None = None) -> list[Item]:
    """One single-visit item per watermelon; ``meta['weeks_out']`` is the grower's ripeness estimate (0 = ripe)."""
    df = pd.read_csv(Path(manifests) / "a2sl_watermelon.csv")
    scan_dim = 2 * (4 + SCAN_RADIAL_BINS)
    items: list[Item] = []
    for item_id, rows in df.groupby("item_id", sort=True):
        keys = rows.scan_key.tolist()
        feat = _scan_features(keys, scans)
        rip = str(rows.ripeness.iloc[0]).strip().lower() if isinstance(rows.ripeness.iloc[0], str) else ""
        items.append(
            Item(
                item_id=str(item_id),
                produce="watermelon",
                storage="ambient",
                temp_c=math.nan,
                visits=[
                    Visit(
                        t=0.0,
                        stage=None,
                        features=feat[None] if feat is not None else np.zeros((1, scan_dim), np.float32),
                    )
                ],
                spoil_lo=0.0,
                spoil_hi=math.inf,
                meta={
                    "source": "a2sl_watermelon_2022",
                    "ripeness_text": rows.ripeness.iloc[0],
                    "weeks_out": RIPENESS_TEXT_TO_WEEKS_OUT.get(rip, math.nan),
                    "type": rows.type.iloc[0],
                    "size": rows["size"].iloc[0],
                    "origin": rows.origin.iloc[0],
                    "scan_keys": keys,
                    "n_grids": len(keys),
                },
            )
        )
    return items
