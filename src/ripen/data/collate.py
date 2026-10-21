"""Turn a list of items into padded arrays, with one prediction instance per item.

A prediction instance is (item, ``n_obs``): the first ``n_obs`` visits are
visible, "now" is the last visible visit, and the target is the spoilage
interval expressed in days from now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .schema import SPOILED_STAGE, Item

INF_SENTINEL = 1e6  # stands in for inf inside arrays


@dataclass(frozen=True)
class DatasetInfo:
    """What the models need to know about a dataset before seeing a batch."""

    groups: tuple[str, ...]
    produce: tuple[str, ...]
    feat_dim: int
    max_visits: int
    t_max: float
    median_event_time: float = float("nan")  # median spoil_hi over uncensored items (t_max if none)

    def group_index(self, storage: str) -> int:
        return self.groups.index(storage)

    def produce_index(self, produce: str) -> int:
        return self.produce.index(produce)

    def to_dict(self) -> dict:
        return {
            "groups": list(self.groups),
            "produce": list(self.produce),
            "feat_dim": self.feat_dim,
            "max_visits": self.max_visits,
            "t_max": self.t_max,
            "median_event_time": self.median_event_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatasetInfo:
        return cls(
            tuple(d["groups"]),
            tuple(d["produce"]),
            int(d["feat_dim"]),
            int(d["max_visits"]),
            float(d["t_max"]),
            float(d.get("median_event_time", d["t_max"])),
        )


def dataset_info(items: list[Item]) -> DatasetInfo:
    feat_dim = 0
    for it in items:
        for v in it.visits:
            if v.features is not None:
                feat_dim = int(v.features.shape[-1])
                break
        if feat_dim:
            break
    events = [it.spoil_hi for it in items if not it.right_censored]
    t_max = max(it.t_last for it in items)
    return DatasetInfo(
        groups=tuple(sorted({it.storage for it in items})),
        produce=tuple(sorted({it.produce for it in items})),
        feat_dim=feat_dim,
        max_visits=max(it.n_visits for it in items),
        t_max=t_max,
        median_event_time=float(np.median(events)) if events else float(t_max),
    )


class Batch(eqx.Module):
    """Padded arrays for ``B`` prediction instances. All times are in days.

    Shapes: ``t, stage, mask, visible`` are ``[B, V]``; ``feat`` is ``[B, V, D]``;
    everything else is ``[B]``. ``stage`` is 0 where unlabelled or padded.
    ``mask`` marks real visits; ``visible`` marks the first ``n_obs`` of them.
    ``lo_rel``/``hi_rel`` are the spoilage interval in days from ``t_now``;
    ``hi_rel`` holds ``INF_SENTINEL`` where ``right_censored``.
    """

    t: Array
    stage: Array
    feat: Array
    mask: Array
    visible: Array
    n_obs: Array
    t_now: Array
    stage_now: Array
    temp_k: Array
    group: Array
    produce: Array
    spoil_lo: Array
    spoil_hi: Array
    lo_rel: Array
    hi_rel: Array
    right_censored: Array
    item_index: Array

    @property
    def batch_size(self) -> int:
        return int(self.t.shape[0])


def choose_n_obs(item: Item, mode: str | int, rng: np.random.Generator | None) -> int:
    """How many visits are visible. ``"all"``: every visit. ``"random"``: uniform over prefixes ending before
    the spoiled visit (so the instance is a genuine forecast). An int: that many (clipped)."""
    n = item.n_visits
    last_unspoiled = n
    for i, v in enumerate(item.visits):
        if v.stage is not None and v.stage >= SPOILED_STAGE:
            last_unspoiled = i
            break
    if mode == "all":
        return n
    if mode == "random":
        if rng is None:
            raise ValueError("random n_obs needs an rng")
        return int(rng.integers(1, max(1, last_unspoiled) + 1))
    return max(1, min(int(mode), n))


def collate(
    items: list[Item],
    info: DatasetInfo,
    n_obs: str | int | list[int] = "all",
    rng: np.random.Generator | None = None,
    item_index: list[int] | None = None,
) -> Batch:
    """Pad ``items`` to ``info.max_visits`` visits and ``info.feat_dim`` features."""
    B, V, D = len(items), info.max_visits, max(info.feat_dim, 1)
    t = np.zeros((B, V), np.float32)
    stage = np.zeros((B, V), np.int32)
    feat = np.zeros((B, V, D), np.float32)
    mask = np.zeros((B, V), bool)
    visible = np.zeros((B, V), bool)
    n_obs_arr = np.zeros(B, np.int32)
    temp_k = np.zeros(B, np.float32)
    group = np.zeros(B, np.int32)
    produce = np.zeros(B, np.int32)
    lo = np.zeros(B, np.float32)
    hi = np.zeros(B, np.float32)
    rc = np.zeros(B, bool)
    for i, it in enumerate(items):
        n = it.n_visits
        if n > V:
            raise ValueError(f"{it.item_id} has {n} visits > max_visits={V}")
        k = choose_n_obs(it, n_obs[i] if isinstance(n_obs, list) else n_obs, rng)
        n_obs_arr[i] = k
        for j, v in enumerate(it.visits):
            t[i, j] = v.t
            stage[i, j] = 0 if v.stage is None else v.stage
            fv = v.feature_vector
            if fv is not None:
                feat[i, j, : fv.shape[0]] = fv
        mask[i, :n] = True
        visible[i, :k] = True
        temp_k[i] = (it.temp_c if not math.isnan(it.temp_c) else 20.0) + 273.15
        group[i] = info.group_index(it.storage)
        produce[i] = info.produce_index(it.produce)
        lo[i] = it.spoil_lo
        hi[i] = INF_SENTINEL if it.right_censored else it.spoil_hi
        rc[i] = it.right_censored
    t_now = t[np.arange(B), n_obs_arr - 1]
    stage_now = stage[np.arange(B), n_obs_arr - 1]
    idx = np.asarray(item_index if item_index is not None else np.arange(B), np.int32)
    return Batch(
        t=jnp.asarray(t),
        stage=jnp.asarray(stage),
        feat=jnp.asarray(feat),
        mask=jnp.asarray(mask),
        visible=jnp.asarray(visible),
        n_obs=jnp.asarray(n_obs_arr),
        t_now=jnp.asarray(t_now),
        stage_now=jnp.asarray(stage_now),
        temp_k=jnp.asarray(temp_k),
        group=jnp.asarray(group),
        produce=jnp.asarray(produce),
        spoil_lo=jnp.asarray(lo),
        spoil_hi=jnp.asarray(hi),
        lo_rel=jnp.asarray(lo - t_now),
        hi_rel=jnp.asarray(np.where(rc, INF_SENTINEL, hi - t_now)),
        right_censored=jnp.asarray(rc),
        item_index=jnp.asarray(idx),
    )
