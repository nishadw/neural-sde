"""Loader for the NASA C-MAPSS turbofan run-to-failure data (Saxena, Goebel, Simon & Eklund, PHM 2008).

Source: ``CMAPSSData.zip`` from https://data.nasa.gov/docs/legacy/CMAPSSData.zip (NASA Prognostics Center
of Excellence data repository), unzipped under ``root``. Only the ``train_FD00x.txt`` files are used: every
training trajectory runs from a healthy state to failure, so the failure cycle is observed exactly.

File format, verified on the downloaded files: whitespace-separated text, 26 numeric columns, one row per
operational cycle, no header. Columns are ``unit, cycle, setting1, setting2, setting3, s1 .. s21``. Cycles
are contiguous ``1..L`` for every unit. Trajectory counts: FD001 100 units, FD002 260, FD003 100, FD004 249
(the readme says 248 for FD004; the file has 249). Cycle counts per unit range from 128 to 543.

**Event definition.** Failure is exact at the last recorded cycle ``L`` of the unit (``meta["failure_cycle"]``).

**Visit schedule.** A real inspection regime is simulated: visits are the sensor readings at cycles
``c_0 = 1``, then ``c_k = 1 + k * inspect_every + U{-jitter, ..., jitter}`` (made strictly increasing), for
all ``c_k < L``. Time is measured in units of ``inspect_every`` cycles, ``t = (c - 1) / inspect_every``, so
consecutive visits are about one time unit apart and the first visit is at ``t = 0``. The interval is
``spoil_lo = t(last visit)``, ``spoil_hi = t(L)``. The visit cycles are in ``meta["visit_cycles"]``.

**Censoring ablation.** ``drop_tail > 0`` right-censors that fraction of the units (chosen with the same seed)
at a uniformly random cycle ``C`` in ``[1, L)``: visits after ``C`` are dropped, ``spoil_lo = t(C)``,
``spoil_hi = inf``; the true failure cycle stays in ``meta["failure_cycle"]`` and ``meta["censored_at_cycle"]``
records ``C``.

**Features.** One ``(1, D)`` float32 vector per visit: the three settings and 21 sensors at that cycle,
standardised to zero mean / unit variance over all rows of the loaded subsets. Columns that are constant over
the loaded rows are dropped when ``drop_constant=True`` (for FD001 alone: ``setting3, s1, s5, s10, s16, s18,
s19``, so ``D = 17``; for FD003 alone: ``setting3, s1, s5, s16, s18, s19``, ``D = 18``; nothing is constant in
FD002/FD004 or any mix with them, ``D = 24``). Kept names, dropped names and the mean/std used are in every
item's ``meta``.

``storage`` is the subset name (``"FD001"``..``"FD004"``, which also encodes the operating-condition and fault-mode
regime), ``produce`` is ``"turbofan"``, ``temp_c`` is ``nan``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Item, Visit

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTINGS = ("setting1", "setting2", "setting3")
SENSORS = tuple(f"s{i}" for i in range(1, 22))
COLUMNS = ("unit", "cycle", *SETTINGS, *SENSORS)
FEATURE_COLUMNS = (*SETTINGS, *SENSORS)


def read_subset(root: str | Path, subset: str, split: str = "train") -> pd.DataFrame:
    """Read ``<root>/<split>_<subset>.txt`` into a frame with the canonical 26 columns."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
    path = Path(root) / f"{split}_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = list(COLUMNS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df["subset"] = subset
    return df.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)


def visit_cycles(failure_cycle: int, inspect_every: int, jitter: int, rng: np.random.Generator) -> list[int]:
    """Inspection cycles ``1 = c_0 < c_1 < ... < failure_cycle`` with jittered spacing."""
    cycles = [1]
    k = 1
    while True:
        c = 1 + k * inspect_every + int(rng.integers(-jitter, jitter + 1))
        c = max(c, cycles[-1] + 1)
        if c >= failure_cycle:
            break
        cycles.append(c)
        k += 1
    return cycles


def load_cmapss(
    root: str | Path,
    subsets: tuple[str, ...] = ("FD001",),
    inspect_every: int = 10,
    jitter: int = 2,
    seed: int = 0,
    drop_tail: float = 0.0,
    drop_constant: bool = True,
) -> list[Item]:
    """Build one :class:`Item` per engine of the requested training subsets; see the module docstring.

    Args:
        root: directory holding ``train_FD00x.txt``.
        subsets: which of ``FD001``..``FD004`` to load.
        inspect_every: nominal cycles between visits; also the time unit.
        jitter: visits are offset by a uniform integer in ``[-jitter, jitter]`` cycles; must be ``< inspect_every``.
        seed: seeds the jitter and the ``drop_tail`` censoring.
        drop_tail: fraction of engines to right-censor at a random cycle before failure.
        drop_constant: drop feature columns that are constant over the loaded rows.
    """
    if inspect_every < 1:
        raise ValueError("inspect_every must be >= 1")
    if not 0 <= jitter < inspect_every:
        raise ValueError("need 0 <= jitter < inspect_every")
    if not 0.0 <= drop_tail <= 1.0:
        raise ValueError("drop_tail must be in [0, 1]")
    subsets = tuple(subsets) if not isinstance(subsets, str) else (subsets,)
    df = pd.concat([read_subset(root, s) for s in subsets], ignore_index=True)

    raw = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    mean, std = raw.mean(0), raw.std(0)
    keep_cols = std > 1e-12 if drop_constant else np.ones(len(FEATURE_COLUMNS), dtype=bool)
    names = [c for c, k in zip(FEATURE_COLUMNS, keep_cols, strict=True) if k]
    dropped = [c for c, k in zip(FEATURE_COLUMNS, keep_cols, strict=True) if not k]
    safe_std = np.where(std > 1e-12, std, 1.0)
    X = ((raw - mean) / safe_std)[:, keep_cols].astype(np.float32)
    stats = {
        "feature_names": names,
        "dropped_columns": dropped,
        "feature_mean": mean[keep_cols].tolist(),
        "feature_std": safe_std[keep_cols].tolist(),
    }

    rng = np.random.default_rng(seed)
    engines = list(df.groupby(["subset", "unit"], sort=True).indices.items())
    n_censor = int(round(drop_tail * len(engines)))
    censored = set(rng.choice(len(engines), size=n_censor, replace=False).tolist()) if n_censor else set()

    items: list[Item] = []
    for e, ((subset, unit), idx) in enumerate(engines):
        cycles = df["cycle"].to_numpy()[idx]
        L = int(cycles[-1])
        if not np.array_equal(cycles, np.arange(1, L + 1)):
            raise ValueError(f"{subset} unit {unit}: cycles are not contiguous 1..{L}")
        vc = visit_cycles(L, inspect_every, jitter, rng)
        censor_at: int | None = None
        if e in censored:
            censor_at = int(rng.integers(1, L))
            vc = [c for c in vc if c <= censor_at]
        visits = [
            Visit(t=(c - 1) / inspect_every, stage=None, features=X[idx[c - 1]][None, :].astype(np.float32)) for c in vc
        ]
        if censor_at is None:
            lo, hi = visits[-1].t, (L - 1) / inspect_every
        else:
            lo, hi = (censor_at - 1) / inspect_every, math.inf
        items.append(
            Item(
                item_id=f"{subset}-{int(unit):03d}",
                produce="turbofan",
                storage=subset,
                temp_c=math.nan,
                visits=visits,
                spoil_lo=lo,
                spoil_hi=hi,
                meta={
                    "unit": int(unit),
                    "subset": subset,
                    "failure_cycle": L,
                    "visit_cycles": vc,
                    "censored_at_cycle": censor_at,
                    "inspect_every": inspect_every,
                    "jitter": jitter,
                    "seed": seed,
                    **stats,
                },
            )
        )
    return items
