"""Sequence model: history encoder (GRU / GRU-D / ODE-RNN) -> discrete survival head.

Trained with the interval-censored likelihood on one prediction instance per
item. With ``n_obs_mode: random`` in the trainer the visible prefix is
resampled every epoch, which is the "random photo dropout" that makes
single-photo and many-photo predictions come from one model.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..data.collate import Batch
from .base import interval_log_prob_from_logits, pmf_to_cdf
from .encoders import SequenceEncoder, VisitEncoder
from .heads import DiscreteSurvivalHead, context_features, group_embedding


class SequenceSurvival(eqx.Module):
    visit_enc: VisitEncoder
    seq_enc: SequenceEncoder
    head: DiscreteSurvivalHead
    group_emb: eqx.nn.Embedding | None
    horizon: int = eqx.field(static=True)

    def __init__(
        self,
        feat_dim: int,
        n_groups: int,
        horizon: int,
        encoder: str = "gru",
        hidden: int = 64,
        visit_dim: int = 32,
        group_dim: int = 4,
        use_images: bool = True,
        use_stages: bool = False,
        *,
        key,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.visit_enc = VisitEncoder(feat_dim, visit_dim, use_images, use_stages, key=k1)
        self.seq_enc = SequenceEncoder(encoder, visit_dim, hidden, key=k2)
        self.group_emb = group_embedding(n_groups, group_dim, k4) if group_dim > 0 else None
        self.head = DiscreteSurvivalHead(hidden + 1 + group_dim, horizon, key=k3)
        self.horizon = horizon

    def context(self, batch: Batch) -> Array:
        x = jax.vmap(jax.vmap(self.visit_enc))(batch.feat, batch.stage)  # [B, V, visit_dim]
        h = jax.vmap(self.seq_enc)(x, batch.t, batch.visible)
        return jnp.concatenate([h, context_features(batch.temp_k, self.group_emb, batch.group)], -1)

    def logits(self, batch: Batch) -> Array:
        return jax.vmap(self.head)(self.context(batch))

    def loss(self, batch: Batch, key: Array) -> Array:
        lp = interval_log_prob_from_logits(self.logits(batch), batch.lo_rel, batch.hi_rel, batch.right_censored)
        return -jnp.mean(lp)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        return pmf_to_cdf(self.logits(batch))
