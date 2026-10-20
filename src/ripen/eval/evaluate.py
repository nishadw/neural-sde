"""Build prediction instances from items and run a model over them."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import numpy as np

from ..data.collate import DatasetInfo, collate
from ..data.schema import SPOILED_STAGE, Item


@dataclass
class PredictionSet:
    F: np.ndarray  # [N, H+1]
    lo: np.ndarray
    hi: np.ndarray  # inf where right-censored
    group: np.ndarray
    item_ids: np.ndarray
    n_obs: np.ndarray
    t_now: np.ndarray
    stage_now: np.ndarray


def prediction_instances(items: list[Item], n_obs_list: list[int] | str = "all_unspoiled") -> list[tuple[int, int]]:
    """``(item index, n_obs)`` pairs. ``"all_unspoiled"``: every prefix whose last visit is not yet spoiled.
    A list of ints: those prefix lengths (clipped to the unspoiled prefixes)."""
    out = []
    for i, it in enumerate(items):
        last = it.n_visits
        for j, v in enumerate(it.visits):
            if v.stage is not None and v.stage >= SPOILED_STAGE:
                last = j  # number of unspoiled visits
                break
        last = max(last, 1)
        ks = range(1, last + 1) if n_obs_list == "all_unspoiled" else [k for k in n_obs_list if k <= last]
        out += [(i, k) for k in ks]
    return out


def predict_all(
    model, items: list[Item], info: DatasetInfo, key, n_obs_list="all_unspoiled", batch_size=64
) -> PredictionSet:
    inst = prediction_instances(items, n_obs_list)
    pred = eqx.filter_jit(lambda m, b, k: m.predict_cdf(b, k))
    Fs, lo, hi, grp, ids, nobs, tnow, snow = [], [], [], [], [], [], [], []
    for s in range(0, len(inst), batch_size):
        chunk = inst[s : s + batch_size]
        batch = collate([items[i] for i, _ in chunk], info, n_obs=[k for _, k in chunk])
        key, sub = jax.random.split(key)
        F = np.asarray(pred(model, batch, sub))
        Fs.append(F)
        lo.append(np.asarray(batch.lo_rel))
        hi.append(np.where(np.asarray(batch.right_censored), np.inf, np.asarray(batch.hi_rel)))
        grp.append(np.asarray(batch.group))
        ids += [items[i].item_id for i, _ in chunk]
        nobs.append(np.asarray(batch.n_obs))
        tnow.append(np.asarray(batch.t_now))
        snow.append(np.asarray(batch.stage_now))
    return PredictionSet(
        np.concatenate(Fs),
        np.concatenate(lo),
        np.concatenate(hi),
        np.concatenate(grp),
        np.asarray(ids),
        np.concatenate(nobs),
        np.concatenate(tnow),
        np.concatenate(snow),
    )
