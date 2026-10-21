"""Loader for PBC2, the Mayo Clinic primary biliary cirrhosis trial (Murtaugh et al. 1994), as cleaned for
Dynamic-DeepHit (Lee, Yoon & van der Schaar, IEEE TBME 2020).

Source file: ``pbc2_cleaned.csv`` from https://github.com/chl8856/Dynamic-DeepHit/tree/master/data
(raw: https://raw.githubusercontent.com/chl8856/Dynamic-DeepHit/master/data/pbc2_cleaned.csv).

Columns as inspected on 2026-10-20 (1945 rows, 19 columns, 312 patients, 1-16 visits each, median 5)::

    id            patient id, 1..312
    tte           time from the first visit to death / transplant / censoring. Integer, constant per patient.
    times         visit time from the first visit (always 0 at the first row of a patient). Integer.
    label         0 = censored (143 patients), 1 = death (140), 2 = liver transplant (29). Constant per patient.
    drug          0/1 trial arm, constant per patient. The file does not say which is D-penicillamine; the
                  R ``JM::pbc2`` factor order is placebo, D-penicil, and 158 of 312 here have drug == 1, which
                  matches the 158 D-penicillamine patients of the trial, so 1 is taken as "D-penicil".
    age           years at baseline, constant per patient
    sex           0/1, constant per patient; 36 of 312 are 1 (the trial is 88% female, so 1 is read as male)
    ascites, hepatomegaly, spiders
                  0/1 per visit; ~60 missing values each
    edema         0 / 1 / 2 per visit (none / edema without diuretics / edema despite diuretics)
    serBilir      serum bilirubin (mg/dl), per visit
    serChol       serum cholesterol (mg/dl), per visit, 821 of 1945 missing
    albumin       g/dl, per visit
    alkaline      alkaline phosphatase (U/l), per visit, 60 missing
    SGOT          U/ml, per visit
    platelets     per cubic ml / 1000, per visit, 73 missing
    prothrombin   seconds, per visit
    histologic    stage 1..4, per visit (changes during follow-up for 149 of 312 patients)

**Time unit of the file.** ``tte`` and ``times`` are integer **weeks**: the median visit gap is 51 (the
protocol is six-monthly then yearly visits), patient 1's ``tte`` of 57 matches the 1.095 years of the
original record, and the maximum ``tte`` of 744 is 14.3 years x 52. The loader converts to **months** with
``months_per_unit = 12 / 52``; every ``t`` and the spoilage interval are in months since the first visit.

**Event definition.** Death (``label == 1``) is the event. Transplant (``label == 2``) is a competing event
and is treated as right-censoring at the transplant time unless ``transplant_as_event=True``; either way
``meta["competing"]`` flags transplanted patients. ``label == 0`` is right-censoring at ``tte``.

**Interval coding.** The event time ``T`` is exact (from the registry, not from a visit), so an event item
gets the narrow interval ``spoil_lo = t_last_visit_before_T``, ``spoil_hi = T``: the model is only told that
the patient was alive at the last visit before ``T`` and dead at ``T``. Visits at ``t >= T`` (eight death
patients have their last visit on the day of death) are dropped from the visit list and counted in
``meta["n_visits_dropped_at_event"]``. A right-censored patient gets ``spoil_lo = censoring time``
(which can be later than the last visit; the last visit time is in ``meta["t_last_visit"]``) and
``spoil_hi = inf``.

**Features.** One ``(1, D)`` float32 vector per visit: the 12 longitudinal columns followed by the 3
baseline columns (``age``, ``sex``, ``drug``), ``D = 15``. ``serBilir``, ``serChol``, ``alkaline`` and
``SGOT`` are ``log1p``-transformed when ``log_skewed=True`` (they are heavily right-skewed). Missing values
are last-observation-carried-forward within a patient and then filled with the column mean (so they become
0 after standardisation). Every column is standardised to zero mean / unit variance with statistics computed
over all rows of the file; the names, means and standard deviations are recorded in every item's ``meta``
(``feature_names``, ``feature_mean``, ``feature_std``) so a split-aware caller can undo and redo them.

``storage`` is the arm (``"placebo"`` / ``"D-penicil"``), ``produce`` is ``"pbc2"``, ``temp_c`` is ``nan``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Item, Visit

LABEL_CENSORED, LABEL_DEATH, LABEL_TRANSPLANT = 0, 1, 2
EVENT_NAMES = {LABEL_CENSORED: "censored", LABEL_DEATH: "death", LABEL_TRANSPLANT: "transplant"}
ARM_NAMES = {0: "placebo", 1: "D-penicil"}
MONTHS_PER_UNIT = 12.0 / 52.0  # the file's time unit is weeks

LONGITUDINAL = (
    "serBilir",
    "serChol",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "histologic",
)
BASELINE = ("age", "sex", "drug")
LOG_COLUMNS = ("serBilir", "serChol", "alkaline", "SGOT")
REQUIRED = ("id", "tte", "times", "label", *LONGITUDINAL, *BASELINE)


def read_pbc2(path: str | Path) -> pd.DataFrame:
    """Read the CSV, check the columns, and sort by patient and visit time."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}")
    df = df.sort_values(["id", "times"], kind="stable").reset_index(drop=True)
    for col in ("tte", "label"):
        if (df.groupby("id")[col].nunique() > 1).any():
            raise ValueError(f"{col} must be constant within a patient")
    if (df.groupby("id")["times"].min() != 0).any():
        raise ValueError("each patient's first visit must be at times == 0")
    return df


def feature_matrix(df: pd.DataFrame, log_skewed: bool = True) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Impute and standardise the covariates. Returns ``(X [n_rows, D], names, mean, std)``."""
    names = [*LONGITUDINAL, *BASELINE]
    raw = df[names].astype(float).copy()
    if log_skewed:
        for c in LOG_COLUMNS:
            raw[c] = np.log1p(raw[c].clip(lower=0))
    raw = raw.groupby(df["id"]).ffill()  # last observation carried forward within a patient
    mean = raw.mean(axis=0, skipna=True).to_numpy()
    std = raw.std(axis=0, skipna=True, ddof=0).to_numpy()
    std = np.where(std > 1e-8, std, 1.0)
    filled = raw.fillna(pd.Series(mean, index=names))
    X = ((filled.to_numpy() - mean) / std).astype(np.float32)
    return X, names, mean.astype(np.float64), std.astype(np.float64)


def load_pbc2(
    path: str | Path,
    transplant_as_event: bool = False,
    log_skewed: bool = True,
    months_per_unit: float = MONTHS_PER_UNIT,
) -> list[Item]:
    """Build one :class:`Item` per patient; see the module docstring for the conventions.

    Args:
        path: ``pbc2_cleaned.csv``.
        transplant_as_event: treat liver transplant as the event (composite endpoint) instead of censoring.
        log_skewed: ``log1p`` the four right-skewed lab columns before standardising.
        months_per_unit: months per file time unit (the file is in weeks).
    """
    df = read_pbc2(path)
    X, names, mean, std = feature_matrix(df, log_skewed=log_skewed)
    stats = {"feature_names": names, "feature_mean": mean.tolist(), "feature_std": std.tolist()}
    items: list[Item] = []
    for pid, idx in df.groupby("id", sort=True).indices.items():
        rows = df.iloc[idx]
        label = int(rows["label"].iloc[0])
        T = float(rows["tte"].iloc[0]) * months_per_unit
        ts = rows["times"].to_numpy(dtype=float) * months_per_unit
        is_event = label == LABEL_DEATH or (label == LABEL_TRANSPLANT and transplant_as_event)
        keep = ts < T if is_event else np.ones_like(ts, dtype=bool)
        if not keep.any():
            keep[0] = True  # a patient whose only visit is on the event day keeps it (interval collapses to exact)
        visits = [
            Visit(t=float(ts[i]), stage=None, features=X[idx[i]][None, :].astype(np.float32))
            for i in range(len(ts))
            if keep[i]
        ]
        t_last = visits[-1].t
        if is_event:
            lo, hi = (t_last if t_last < T else T), T
        else:
            lo, hi = T, math.inf
        drug = int(rows["drug"].iloc[0])
        items.append(
            Item(
                item_id=f"pbc2-{int(pid):03d}",
                produce="pbc2",
                storage=ARM_NAMES.get(drug, str(drug)),
                temp_c=math.nan,
                visits=visits,
                spoil_lo=lo,
                spoil_hi=hi,
                meta={
                    "id": int(pid),
                    "label": label,
                    "event": EVENT_NAMES.get(label, str(label)),
                    "competing": label == LABEL_TRANSPLANT,
                    "transplant_as_event": transplant_as_event,
                    "t_event": T,
                    "t_last_visit": t_last,
                    "tte_file_units": float(rows["tte"].iloc[0]),
                    "months_per_unit": months_per_unit,
                    "n_visits_dropped_at_event": int(len(ts) - len(visits)),
                    **stats,
                },
            )
        )
    return items
