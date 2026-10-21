"""Observation heads: an ordinal stage head and a discrete-time survival head."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..data.schema import N_STAGES


class OrdinalHead(eqx.Module):
    """Cumulative-link ordinal model on a scalar ripeness ``r``: ``P(stage <= k) = sigmoid(theta_k - r)``.

    The ``N_STAGES - 1`` thresholds are increasing by construction. The last
    threshold is the spoilage boundary: ``stage == N_STAGES`` iff ``r > theta_{K-1}``.
    With ``fixed=True`` the thresholds are pinned to ``2, 3, ..., N_STAGES`` so the
    latent is in stage units (used by the synthetic recovery test; it removes the
    scale non-identifiability between latent and thresholds).
    """

    cut0: Array
    log_gaps: Array
    log_scale: Array
    fixed: bool = eqx.field(static=True)
    fixed_scale: float | None = eqx.field(static=True)

    def __init__(self, fixed: bool = False, init_cut0: float = 2.0, link_scale: float = 0.5, learn_scale: bool = True):
        """``link_scale`` = logistic width of the link in latent units: ``P(stage <= k) = sigmoid((theta_k - r)/s)``.
        Small ``s`` means stages are a near-deterministic function of the latent (simulated hard labels); larger
        ``s`` absorbs rater noise. Learnable unless ``learn_scale=False``."""
        self.cut0 = jnp.asarray(init_cut0, jnp.float32)
        self.log_gaps = jnp.zeros(N_STAGES - 2, jnp.float32)  # softplus(0) ~ 0.69; fixed mode ignores this
        self.log_scale = jnp.log(jnp.asarray(link_scale, jnp.float32))
        self.fixed = fixed
        self.fixed_scale = None if learn_scale else float(link_scale)

    @property
    def scale(self) -> Array:
        if self.fixed_scale is not None:
            return jnp.asarray(self.fixed_scale, jnp.float32)
        return jnp.clip(jnp.exp(self.log_scale), 0.02, 3.0)

    def thresholds(self) -> Array:
        if self.fixed:
            return jnp.arange(2, N_STAGES + 1, dtype=jnp.float32)
        gaps = jax.nn.softplus(self.log_gaps)
        return self.cut0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(gaps)])

    @property
    def boundary(self) -> Array:
        return self.thresholds()[-1]

    def log_probs(self, r: Array) -> Array:
        """``[..., N_STAGES]`` log P(stage = k | r) for ``r`` of any shape."""
        th = self.thresholds()
        cdf = jax.nn.sigmoid((th - r[..., None]) / self.scale)  # P(stage <= k), k = 1..K-1
        cdf = jnp.concatenate([jnp.zeros_like(r)[..., None], cdf, jnp.ones_like(r)[..., None]], axis=-1)
        p = jnp.clip(cdf[..., 1:] - cdf[..., :-1], 1e-6, 1.0)
        return jnp.log(p)

    def log_prob(self, r: Array, stage: Array) -> Array:
        """log P(stage | r); ``stage`` in 1..N_STAGES (0 is treated as missing -> 0)."""
        lp = self.log_probs(r)
        idx = jnp.clip(stage - 1, 0, N_STAGES - 1)
        out = jnp.take_along_axis(lp, idx[..., None], axis=-1)[..., 0]
        return jnp.where(stage > 0, out, 0.0)

    def expected_stage(self, r: Array) -> Array:
        return jnp.sum(jnp.exp(self.log_probs(r)) * jnp.arange(1, N_STAGES + 1), axis=-1)


class DiscreteSurvivalHead(eqx.Module):
    """MLP -> logits over ``horizon + 2`` bins: days-left 0..H and "after H"."""

    net: eqx.nn.MLP
    horizon: int = eqx.field(static=True)

    def __init__(self, in_dim: int, horizon: int, width: int = 64, depth: int = 2, *, key: Array):
        self.horizon = horizon
        self.net = eqx.nn.MLP(in_dim, horizon + 2, width, depth, activation=jax.nn.gelu, key=key)

    def __call__(self, x: Array) -> Array:
        return self.net(x)


def group_embedding(n_groups: int, dim: int, key: Array) -> eqx.nn.Embedding:
    return eqx.nn.Embedding(n_groups, dim, key=key)


def context_features(batch_temp_k: Array, group_emb: eqx.nn.Embedding | None, group: Array) -> Array:
    """Per-item context: scaled temperature plus an optional learned group embedding. Vectorised over the batch."""
    temp = ((batch_temp_k - 273.15) - 15.0) / 10.0
    parts = [temp[:, None]]
    if group_emb is not None:
        parts.append(jax.vmap(group_emb)(group))
    return jnp.concatenate(parts, axis=-1)
