"""Exact first-passage survival of a 1-D diffusion with learned drift and diffusion, by solving the backward
Kolmogorov equation on a grid. Differentiable, simulation-free, one solve per parameter setting.

For the progression coordinate ``dR = mu(R) ds + sigma(R) dW`` in physiological time ``s`` and an absorbing
boundary ``b``, the survival function ``S(s, r) = P(no crossing of b in [0, s] | R(0) = r)`` solves

    d/ds S = mu(r) d/dr S + 1/2 sigma(r)^2 d2/dr2 S,     S(0, r) = 1 for r < b,   S(s, b) = 0,   d/dr S(s, r_min) = 0.

Because the clock ``kappa`` multiplies the whole generator, an item at condition ``x`` with latent ``r`` at
time ``t`` has ``P(tau - t > h) = S(kappa(x) h | r)``: the clock commutes with the PDE, so one table serves
every item, condition and horizon. The scheme is Crank-Nicolson in ``s`` with two implicit-Euler start-up steps
(Rannacher), a hybrid central/upwind first derivative and a centred second derivative; the operator is
constant in ``s``, so its inverse is formed once and each step is a matrix-vector product.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def generator_matrix(
    mu: Callable[[Array], Array], sigma: Callable[[Array], Array], boundary: Array, r_min: float, n_grid: int
) -> tuple[Array, Array, Array]:
    """Discrete backward generator ``L`` of ``mu S' + 1/2 sigma^2 S''`` on ``r = linspace(r_min, boundary, n_grid)``.

    Central differences for ``S'`` where the cell Peclet number allows an M-matrix (``|mu| h <= sigma^2``),
    upwind elsewhere; centred ``S''``; reflecting at ``r_min`` (``S_0 = S_1``), absorbing at the boundary (row
    zeroed, so the last node is a cemetery for the transposed, forward operator). Returns ``(L, r, h)``.
    """
    r = jnp.linspace(r_min, boundary, n_grid)
    h = (boundary - r_min) / (n_grid - 1)
    m = jax.vmap(mu)(r)
    s2 = 0.5 * jax.vmap(sigma)(r) ** 2
    central = jnp.abs(m) * h <= 2.0 * s2
    up = jnp.where(central, m / (2 * h), jnp.where(m >= 0, m / h, 0.0)) + s2 / h**2  # coefficient of S_{i+1}
    lo = jnp.where(central, -m / (2 * h), jnp.where(m < 0, -m / h, 0.0)) + s2 / h**2  # coefficient of S_{i-1}
    diag = -(up + lo)
    L = jnp.diag(diag) + jnp.diag(up[:-1], 1) + jnp.diag(lo[1:], -1)
    L = L.at[0, :].set(0.0).at[0, 0].set(-(up[0])).at[0, 1].set(up[0])
    L = L.at[-1, :].set(0.0)
    return L, r, h


def survival_table(
    mu: Callable[[Array], Array],
    sigma: Callable[[Array], Array],
    boundary: Array,
    r_min: float,
    n_grid: int,
    ds: float,
    n_steps: int,
) -> tuple[Array, Array]:
    """Survival ``S[k, i] = S(k ds, r_i)`` on ``r_i = linspace(r_min, boundary, n_grid)``. Returns ``(S, r_grid)``.

    ``mu``/``sigma`` map a scalar ``r`` to scalars (vectorised with ``vmap``). ``S`` has shape
    ``(n_steps + 1, n_grid)``; ``S[:, -1] = 0`` (absorbing), ``S[0, :-1] = 1``.
    """
    L, r, _ = generator_matrix(mu, sigma, boundary, r_min, n_grid)
    n_grid = L.shape[0]
    ds = float(ds)
    eye = jnp.eye(n_grid)
    # theta scheme: Crank-Nicolson (second order in ds) after two implicit-Euler start-up steps (Rannacher), which
    # damp the oscillations CN would otherwise create from the jump in the initial condition at the boundary
    M_ie = jnp.linalg.inv(eye - ds * L).at[-1, :].set(0.0)
    M_cn = (jnp.linalg.inv(eye - 0.5 * ds * L) @ (eye + 0.5 * ds * L)).at[-1, :].set(0.0)
    s0 = jnp.ones(n_grid).at[-1].set(0.0)

    def step(s, k):
        M = jnp.where(k < 2, M_ie, M_cn)
        s_new = jnp.clip(M @ s, 0.0, 1.0)
        return s_new, s_new

    _, S = jax.lax.scan(step, s0, jnp.arange(n_steps))
    S = jnp.minimum.accumulate(jnp.concatenate([s0[None], S], axis=0), axis=0)  # enforce monotone decay exactly
    return S, r


def lookup(S: Array, r_grid: Array, ds: float, s_query: Array, r_query: Array) -> Array:
    """Bilinear interpolation of ``S`` at physiological time ``s_query`` and latent ``r_query`` (any matching shapes).

    Queries beyond the table in ``s`` take the last row (so ``pde_steps * ds`` should exceed ``kappa_max`` times the
    longest horizon); ``r`` above the boundary gives 0, below ``r_min`` the edge.
    """
    n_steps = S.shape[0] - 1
    ks = jnp.clip(s_query / ds, 0.0, n_steps)
    k0 = jnp.floor(ks).astype(jnp.int32)
    k1 = jnp.minimum(k0 + 1, n_steps)
    w = ks - k0

    v0 = jax.vmap(lambda k, rq: jnp.interp(rq, r_grid, S[k]))(k0.ravel(), r_query.ravel()).reshape(k0.shape)
    v1 = jax.vmap(lambda k, rq: jnp.interp(rq, r_grid, S[k]))(k1.ravel(), r_query.ravel()).reshape(k1.shape)
    out = (1.0 - w) * v0 + w * v1
    return jnp.where(r_query >= r_grid[-1], 0.0, out)


def markov_interval_log_lik(
    S: Array,
    r_grid: Array,
    ds: float,
    kappa: Array,
    t: Array,
    r_at: Array,
    visible: Array,
    spoil_lo: Array,
    spoil_hi: Array,
    right_censored: Array,
    sigma_at: Array | None = None,
) -> Array:
    """Log-likelihood of the censoring observations given the latent at the visits (one item).

    ``t``: ``[V]`` visit times, ``r_at``: ``[V]`` progression latent at the visits, ``visible``: ``[V]`` bool,
    ``sigma_at``: ``[V]`` physiological diffusion at the visits (``None`` -> the product term is dropped).
    The anchor is the last visible visit with ``t <= spoil_lo``. Between two visible visits both endpoints of
    the latent are known, so the no-crossing probability is the *bridge* factor (exact for constant
    coefficients, first order otherwise):

        P(no crossing in (t_k, t_{k+1}] | r_k, r_{k+1}) = 1 - exp(-2 (b - r_k)(b - r_{k+1}) / (sigma_k^2 kappa dt_k))

    and the last interval, whose far end is unobserved, uses the survival table:

        log L = sum_{k < anchor} log bridge_k + log[ S(kappa (lo - t_a) | r_a) - S(kappa (hi - t_a) | r_a) ]

    with the second term inside the bracket dropped when right-censored. A zero-width interval (the first
    visit already spoiled) is widened to ``(lo, lo + ds / kappa]`` so it keeps a gradient toward the boundary.
    """
    V = t.shape[0]
    b = r_grid[-1]
    is_anchor_cand = visible & (t <= spoil_lo + 1e-6)
    anchor = jnp.max(jnp.where(is_anchor_cand, jnp.arange(V), -1))
    anchor = jnp.maximum(anchor, 0)
    t_a = t[anchor]
    r_a = r_at[anchor]
    log_prod = jnp.asarray(0.0)
    if sigma_at is not None:
        dt = jnp.clip(jnp.diff(t, append=t[-1:]), 1e-6, None)
        r_next = jnp.concatenate([r_at[1:], r_at[-1:]])
        gap = jnp.clip(b - r_at, 0.0, None) * jnp.clip(b - r_next, 0.0, None)
        bridge = 1.0 - jnp.exp(-2.0 * gap / (sigma_at**2 * kappa * dt + 1e-9))
        use = visible & (jnp.arange(V) < anchor)
        log_prod = jnp.sum(jnp.where(use, jnp.log(jnp.clip(bridge, 1e-8, 1.0)), 0.0))
    width = jnp.where(
        spoil_hi - spoil_lo < 1e-6, ds / jnp.maximum(kappa, 1e-3), 0.0
    )  # zero-width -> (lo, lo + ds/kappa]
    s_lo = lookup(S, r_grid, ds, kappa * jnp.clip(spoil_lo - t_a, 0.0, None), r_a)
    s_hi = lookup(S, r_grid, ds, kappa * jnp.clip(spoil_hi + width - t_a, 0.0, None), r_a)
    last = jnp.where(right_censored, s_lo, s_lo - s_hi)
    return log_prod + jnp.log(jnp.clip(last, 1e-8, 1.0))
