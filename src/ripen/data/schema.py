"""The one record every loader emits.

An :class:`Item` is one piece of produce followed over time. Its spoilage time
is never observed exactly: it lies in ``(spoil_lo, spoil_hi]`` where
``spoil_lo`` is the last visit at which the item was seen unspoiled and
``spoil_hi`` the first visit at which it was seen spoiled. If it was still
unspoiled at the last visit, ``spoil_hi`` is ``inf`` (right-censored).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

N_STAGES = 5
SPOILED_STAGE = 5  # end of shelf life in the 5-stage Ripening Index of Xavier et al. (2024)


@dataclass(frozen=True)
class Visit:
    """One observation of an item.

    Args:
        t: days since the item's first visit.
        stage: ripeness stage in ``1..N_STAGES``, or ``None`` if unlabelled.
        image_paths: paths of the photos taken at this visit (any number of views).
        features: cached encoder features, shape ``(n_images, D)``, or ``None``.
    """

    t: float
    stage: int | None = None
    image_paths: tuple[str, ...] = ()
    features: np.ndarray | None = None

    @property
    def feature_vector(self) -> np.ndarray | None:
        """Mean over views, shape ``(D,)``; ``None`` if no features are cached."""
        if self.features is None:
            return None
        return np.asarray(self.features, dtype=np.float32).reshape(-1, self.features.shape[-1]).mean(0)


@dataclass
class Item:
    """One piece of produce and its visit history.

    Args:
        item_id: unique identifier.
        produce: produce type, e.g. ``"avocado"``.
        storage: storage group label, e.g. ``"T10"``.
        temp_c: storage temperature in Celsius (``nan`` if unknown).
        visits: visits sorted by time; the first has ``t == 0``.
        spoil_lo: last time the item was known unspoiled.
        spoil_hi: first time the item was known spoiled; ``inf`` if right-censored.
        rh: relative humidity in percent (``nan`` if unknown).
        meta: free-form extras (ground truth for synthetic items, source rows, ...).
    """

    item_id: str
    produce: str
    storage: str
    temp_c: float
    visits: list[Visit]
    spoil_lo: float
    spoil_hi: float
    rh: float = math.nan
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.visits:
            raise ValueError(f"{self.item_id}: an item needs at least one visit")
        ts = [v.t for v in self.visits]
        if ts != sorted(ts):
            raise ValueError(f"{self.item_id}: visits must be sorted by time")
        if not (self.spoil_lo <= self.spoil_hi):
            raise ValueError(f"{self.item_id}: need spoil_lo <= spoil_hi, got {self.spoil_lo} > {self.spoil_hi}")

    @property
    def right_censored(self) -> bool:
        return math.isinf(self.spoil_hi)

    @property
    def censor_type(self) -> str:
        """``"right"``, ``"exact"`` (interval of zero width) or ``"interval"``."""
        if self.right_censored:
            return "right"
        return "exact" if self.spoil_hi - self.spoil_lo < 1e-9 else "interval"

    @property
    def n_visits(self) -> int:
        return len(self.visits)

    @property
    def t_last(self) -> float:
        return self.visits[-1].t

    def prefix(self, n_visits: int) -> Item:
        """The same item with only the first ``n_visits`` visits visible. The spoilage interval is unchanged."""
        n = max(1, min(n_visits, len(self.visits)))
        return replace(self, visits=self.visits[:n])


def spoil_interval_from_visits(visits: list[Visit]) -> tuple[float, float]:
    """Derive ``(spoil_lo, spoil_hi)`` from labelled visits: first visit at ``SPOILED_STAGE`` ends the interval."""
    lo = visits[0].t
    for v in visits:
        if v.stage is not None and v.stage >= SPOILED_STAGE:
            return lo, v.t
        lo = v.t
    return lo, math.inf


def subsample_visits(item: Item, every: int, offset: int = 0) -> Item:
    """Keep every ``every``-th visit (by index, starting at ``offset``), then recompute the spoilage interval.

    This is how "photos every other day" is simulated from daily data: the
    censoring interval widens because spoilage is only seen at a kept visit.
    Visits after the first spoiled visit are dropped. The first visit is always kept.
    """
    if every < 1:
        raise ValueError("every must be >= 1")
    kept = [v for i, v in enumerate(item.visits) if i == 0 or (i - offset) % every == 0]
    out: list[Visit] = []
    for v in kept:
        out.append(v)
        if v.stage is not None and v.stage >= SPOILED_STAGE:
            break
    lo, hi = spoil_interval_from_visits(out)
    if math.isinf(hi) and not item.right_censored:
        # spoilage happened after the last kept visit but inside the original window:
        # still only known to be after `lo`, so it is right-censored at the kept resolution.
        pass
    return replace(item, visits=out, spoil_lo=lo, spoil_hi=hi)
