"""Per-photo baselines.

* :class:`StageLinearRule` — the published method (Xavier et al. 2024): classify the current photo's stage,
  then ``days_left = alpha_g * (stage - 5)`` with ``alpha_g`` refit by OLS on the training fruit.
* :class:`DirectRegressor` — one photo plus storage context -> days left (Huber loss), as in Hangud & Hangud (2026).
* :class:`DiscreteSurvival` — one photo plus context -> PMF over days left, trained with the interval-censored
  likelihood, so censored fruit contribute.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from ..data.collate import Batch
from ..data.schema import SPOILED_STAGE
from .base import interval_log_prob_from_logits, pmf_to_cdf, step_cdf
from .heads import DiscreteSurvivalHead, OrdinalHead, context_features, group_embedding


def _now_features(batch: Batch) -> Array:
    idx = batch.n_obs - 1
    return batch.feat[jnp.arange(batch.batch_size), idx]


class StageLinearRule(eqx.Module):
    net: eqx.nn.MLP
    head: OrdinalHead
    alpha: Array  # [n_groups] days per stage, negative
    horizon: int = eqx.field(static=True)

    def __init__(self, feat_dim: int, n_groups: int, horizon: int, width: int = 64, depth: int = 2, *, key):
        self.net = eqx.nn.MLP(feat_dim, "scalar", width, depth, activation=jax.nn.gelu, key=key)
        self.head = OrdinalHead()
        self.alpha = jnp.full((n_groups,), -3.0, jnp.float32)
        self.horizon = horizon

    def ripeness(self, feat: Array) -> Array:
        return jax.vmap(jax.vmap(self.net))(feat)  # [B, V]

    def loss(self, batch: Batch, key: Array) -> Array:
        """Ordinal NLL over every visible labelled photo (the classifier never sees days-left)."""
        r = self.ripeness(batch.feat)
        lp = self.head.log_prob(r, batch.stage)
        w = batch.visible & (batch.stage > 0)
        return -jnp.sum(jnp.where(w, lp, 0.0)) / jnp.maximum(jnp.sum(w), 1)

    def fit_closed_form(self, batch: Batch) -> StageLinearRule:
        """OLS of days-left on (stage - 5) through the origin, per group, on labelled uncensored visits."""
        stage = np.asarray(batch.stage, float)
        t = np.asarray(batch.t)
        hi = np.asarray(batch.spoil_hi)[:, None]
        rc = np.asarray(batch.right_censored)[:, None]
        valid = np.asarray(batch.mask) & (stage > 0) & ~rc
        x = stage - SPOILED_STAGE
        y = hi - t
        alpha = np.asarray(self.alpha).copy()
        group = np.asarray(batch.group)
        for g in range(alpha.shape[0]):
            m = valid & (group == g)[:, None]
            if m.sum() >= 2 and np.sum(x[m] ** 2) > 0:
                alpha[g] = float(np.sum(x[m] * y[m]) / np.sum(x[m] ** 2))
        return eqx.tree_at(lambda m: m.alpha, self, jnp.asarray(alpha, jnp.float32))

    def predict_days_left(self, batch: Batch) -> Array:
        s_hat = self.head.expected_stage(jax.vmap(self.net)(_now_features(batch)))
        return jnp.clip(self.alpha[batch.group] * (s_hat - SPOILED_STAGE), 0.0, None)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        return step_cdf(self.predict_days_left(batch), self.horizon)


class DirectRegressor(eqx.Module):
    net: eqx.nn.MLP
    group_emb: eqx.nn.Embedding | None
    horizon: int = eqx.field(static=True)
    huber_delta: float = eqx.field(static=True)

    def __init__(
        self, feat_dim: int, n_groups: int, horizon: int, width=64, depth=2, group_dim=4, huber_delta=2.0, *, key
    ):
        k1, k2 = jax.random.split(key)
        self.group_emb = group_embedding(n_groups, group_dim, k2) if group_dim > 0 else None
        self.net = eqx.nn.MLP(feat_dim + 1 + group_dim, "scalar", width, depth, activation=jax.nn.gelu, key=k1)
        self.horizon, self.huber_delta = horizon, huber_delta

    def predict_days_left(self, batch: Batch) -> Array:
        x = jnp.concatenate([_now_features(batch), context_features(batch.temp_k, self.group_emb, batch.group)], -1)
        return jax.nn.softplus(jax.vmap(self.net)(x))

    def loss(self, batch: Batch, key: Array) -> Array:
        """Huber loss on uncensored instances only (censored fruit have no days-left label)."""
        pred = self.predict_days_left(batch)
        err = optax.huber_loss(pred, batch.hi_rel, delta=self.huber_delta)
        w = ~batch.right_censored
        return jnp.sum(jnp.where(w, err, 0.0)) / jnp.maximum(jnp.sum(w), 1)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        return step_cdf(self.predict_days_left(batch), self.horizon)


class DiscreteSurvival(eqx.Module):
    head: DiscreteSurvivalHead
    group_emb: eqx.nn.Embedding | None
    horizon: int = eqx.field(static=True)

    def __init__(self, feat_dim: int, n_groups: int, horizon: int, width=64, depth=2, group_dim=4, *, key):
        k1, k2 = jax.random.split(key)
        self.group_emb = group_embedding(n_groups, group_dim, k2) if group_dim > 0 else None
        self.head = DiscreteSurvivalHead(feat_dim + 1 + group_dim, horizon, width, depth, key=k1)
        self.horizon = horizon

    def logits(self, batch: Batch) -> Array:
        x = jnp.concatenate([_now_features(batch), context_features(batch.temp_k, self.group_emb, batch.group)], -1)
        return jax.vmap(self.head)(x)

    def loss(self, batch: Batch, key: Array) -> Array:
        lp = interval_log_prob_from_logits(self.logits(batch), batch.lo_rel, batch.hi_rel, batch.right_censored)
        return -jnp.mean(lp)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        return pmf_to_cdf(self.logits(batch))
