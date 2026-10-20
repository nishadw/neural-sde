"""Encoders from a visible visit history to a context vector.

All three run over the padded visit axis with ``lax.scan`` and respect the
``visible`` mask, so hidden (future) visits never leak in.

* ``gru``: GRU on ``[x_k, dt_k]`` — the "GRU with time-gap features" baseline.
* ``gru_d``: the hidden state decays towards zero with the gap before each
  update, ``h <- h * exp(-softplus(w) * dt)`` (Che et al. 2016, hidden-state
  decay; input decay is unneeded because every visit has all its inputs).
* ``ode_rnn``: the hidden state follows ``dh/dt = f(h)`` between visits
  (fixed-step RK4) and a GRU update at each visit (Rubanova et al. 2019).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..data.schema import N_STAGES


class VisitEncoder(eqx.Module):
    """Per-visit input: ``[features (if used), one-hot stage (if used)] -> MLP -> x``."""

    net: eqx.nn.MLP
    use_images: bool = eqx.field(static=True)
    use_stages: bool = eqx.field(static=True)
    feat_dim: int = eqx.field(static=True)

    def __init__(self, feat_dim: int, out_dim: int, use_images: bool, use_stages: bool, width: int = 64, *, key):
        if not (use_images or use_stages):
            raise ValueError("a visit encoder needs images or stages")
        self.use_images, self.use_stages, self.feat_dim = use_images, use_stages, feat_dim
        in_dim = (feat_dim if use_images else 0) + (N_STAGES + 1 if use_stages else 0)
        self.net = eqx.nn.MLP(in_dim, out_dim, width, 1, activation=jax.nn.gelu, key=key)

    def __call__(self, feat: Array, stage: Array) -> Array:
        """``feat``: ``[D]``, ``stage``: scalar int (0 = missing). Returns ``[out_dim]``."""
        parts = []
        if self.use_images:
            parts.append(feat)
        if self.use_stages:
            parts.append(jax.nn.one_hot(stage, N_STAGES + 1))
        return self.net(jnp.concatenate(parts))


class SequenceEncoder(eqx.Module):
    kind: str = eqx.field(static=True)
    cell: eqx.nn.GRUCell
    decay: Array | None
    ode_net: eqx.nn.MLP | None
    hidden: int = eqx.field(static=True)
    ode_substeps: int = eqx.field(static=True)

    def __init__(self, kind: str, in_dim: int, hidden: int, *, key, ode_substeps: int = 4):
        if kind not in ("gru", "gru_d", "ode_rnn"):
            raise ValueError(kind)
        k1, k2 = jax.random.split(key)
        self.kind, self.hidden, self.ode_substeps = kind, hidden, ode_substeps
        self.cell = eqx.nn.GRUCell(in_dim + 1, hidden, key=k1)  # +1: the time gap as an input
        self.decay = jnp.zeros(hidden, jnp.float32) if kind == "gru_d" else None
        self.ode_net = (
            eqx.nn.MLP(hidden, hidden, hidden, 1, activation=jax.nn.tanh, key=k2) if kind == "ode_rnn" else None
        )

    def _evolve(self, h: Array, dt: Array) -> Array:
        if self.kind == "gru_d":
            return h * jnp.exp(-jax.nn.softplus(self.decay) * dt)
        if self.kind == "ode_rnn":
            step = dt / self.ode_substeps

            def rk4(h, _):
                k1 = self.ode_net(h)
                k2 = self.ode_net(h + 0.5 * step * k1)
                k3 = self.ode_net(h + 0.5 * step * k2)
                k4 = self.ode_net(h + step * k3)
                return h + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

            h, _ = jax.lax.scan(rk4, h, None, length=self.ode_substeps)
            return h
        return h

    def __call__(self, x: Array, t: Array, visible: Array) -> Array:
        """``x``: ``[V, in_dim]`` per-visit inputs, ``t``: ``[V]``, ``visible``: ``[V]`` bool -> final ``[hidden]``."""
        dt = jnp.diff(t, prepend=t[:1])

        def step(h, inp):
            xk, dtk, vis = inp
            h_evolved = self._evolve(h, dtk)
            h_new = self.cell(jnp.concatenate([xk, dtk[None]]), h_evolved)
            return jnp.where(vis, h_new, h), None

        h0 = jnp.zeros(self.hidden)
        h, _ = jax.lax.scan(step, h0, (x, dt, visible))
        return h
