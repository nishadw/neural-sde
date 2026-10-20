"""The shared interface.

``predict_cdf`` returns ``F[b, h] = P(tau - t_now <= h)`` for ``h = 0, 1, ..., horizon``
days from the last visible visit. Point-prediction models return a step
function, so CRPS reduces to MAE for them and the baselines are comparable
to the distributional models under one metric.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array

from ..data.collate import Batch


@runtime_checkable
class ShelfLifeModel(Protocol):
    horizon: int

    def loss(self, batch: Batch, key: Array) -> Array:
        """Mean training loss over the batch (a scalar)."""
        ...

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        """``[B, horizon + 1]`` CDF of days-left on the integer grid."""
        ...


def horizon_grid(horizon: int) -> Array:
    return jnp.arange(horizon + 1, dtype=jnp.float32)


def step_cdf(days_left: Array, horizon: int) -> Array:
    """CDF of a point prediction: ``F[h] = 1[h >= days_left]``. ``days_left`` is ``[B]``."""
    h = horizon_grid(horizon)[None, :]
    return (h >= jnp.clip(days_left, 0.0, None)[:, None]).astype(jnp.float32)


def pmf_to_cdf(logits: Array) -> Array:
    """Logits over ``horizon + 2`` bins (days 0..H, then "after H") -> CDF on days 0..H."""
    p = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    return jnp.cumsum(p, axis=-1)[..., :-1]


def interval_log_prob_from_logits(logits: Array, lo_rel: Array, hi_rel: Array, right_censored: Array) -> Array:
    """log P(tau_rel in (lo_rel, hi_rel]) under a PMF on integer days 0..H plus an "after H" bin.

    Right-censored: log P(tau_rel > lo_rel). Shapes: logits ``[B, H+2]``, others ``[B]``.
    """
    logp = logits - jnp.max(logits, axis=-1, keepdims=True)
    logp = logp - jnp.log(jnp.sum(jnp.exp(logp), axis=-1, keepdims=True))
    H = logits.shape[-1] - 2
    days = jnp.arange(H + 2, dtype=jnp.float32)  # last bin stands for any day > H
    days = days.at[-1].set(jnp.inf)
    # an interval of zero width (spoiled at the very first visit) means "in the bin at hi"
    lo_rel = jnp.where(hi_rel - lo_rel < 1e-6, lo_rel - 1.0, lo_rel)
    after_lo = days[None, :] > lo_rel[:, None]
    upto_hi = days[None, :] <= hi_rel[:, None]
    sel = jnp.where(right_censored[:, None], after_lo, after_lo & upto_hi)
    masked = jnp.where(sel, logp, -jnp.inf)
    return jax_logsumexp(masked)


def jax_logsumexp(x: Array) -> Array:
    m = jnp.max(x, axis=-1)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    out = m_safe + jnp.log(jnp.sum(jnp.exp(x - m_safe[..., None]), axis=-1))
    return jnp.where(jnp.isfinite(m), out, -1e4)
