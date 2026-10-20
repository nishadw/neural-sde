"""First-passage utilities.

``first_passage_cdf`` turns sampled ripeness paths into ``P(tau <= t)``. The
hard indicator has zero pathwise gradient (the crossing time moves but the
indicator's value does not), so training uses a sigmoid of the running
maximum's distance to the boundary with temperature ``beta``; evaluation uses
the hard version.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.stats import norm


def running_max_from(r: Array, start_idx: Array) -> Array:
    """Running max of ``r[t]`` over ``t >= start_idx``; ``-inf`` before. ``r`` is ``[T]``."""
    t = jnp.arange(r.shape[0])
    masked = jnp.where(t >= start_idx, r, -jnp.inf)
    return jax.lax.cummax(masked, axis=0)


def crossing_prob(running_max: Array, boundary: Array, beta: float | None) -> Array:
    """Smoothed (``beta``) or hard (``beta=None``) indicator that the running max has reached ``boundary``."""
    if beta is None:
        return (running_max >= boundary).astype(jnp.float32)
    return jax.nn.sigmoid((running_max - boundary) / beta)


def first_passage_cdf(r_paths: Array, boundary: Array, start_idx: Array, query_idx: Array, beta: float | None) -> Array:
    """``r_paths``: ``[M, T]`` sampled ripeness on a time grid. Returns ``[len(query_idx)]`` = P(crossed by query time),
    averaged over the ``M`` paths, conditioning on no crossing before ``start_idx`` by ignoring that segment."""
    rm = jax.vmap(lambda r: running_max_from(r, start_idx))(r_paths)  # [M, T]
    crossed = crossing_prob(rm, boundary, beta)
    return jnp.mean(crossed[:, query_idx], axis=0)


def inverse_gaussian_cdf(t: np.ndarray, distance: float, drift: float, sigma: float) -> np.ndarray:
    """CDF of the first time Brownian motion with drift ``drift`` and volatility ``sigma`` travels ``distance`` > 0.

    Classical formula (e.g. Chhikara & Folks): with mu = d/v and lambda = d^2/sigma^2,
    F(t) = Phi(sqrt(l/t)(t/mu - 1)) + exp(2 l / mu) Phi(-sqrt(l/t)(t/mu + 1)).
    """
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    pos = t > 0
    tt = t[pos]
    mu = distance / drift
    lam = distance**2 / sigma**2
    a = np.sqrt(lam / tt)
    out[pos] = norm.cdf(a * (tt / mu - 1.0)) + np.exp(2.0 * lam / mu) * norm.cdf(-a * (tt / mu + 1.0))
    return np.clip(out, 0.0, 1.0)
