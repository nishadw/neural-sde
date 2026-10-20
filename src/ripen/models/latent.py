"""The main model: a latent neural SDE (or ODE) with an ordinal stage head and a first-passage spoilage time.

Prior dynamics      dZ = kappa(T) mu_theta(Z) dt + sigma_theta(Z) dW
Posterior dynamics  dZ = [kappa(T) mu_theta(Z) + g_phi(Z, ctx, t)] dt + sigma_theta(Z) dW   for t <= t_now
Observation         stage_k ~ Ordinal(r(Z_{t_k})),   r(z) = w . z + c
Spoilage            tau = inf { t : r(Z_t) >= b },  b = top threshold of the ordinal head

Training maximises the latent-SDE ELBO of Li et al. (2020): expected
log-likelihood of the visible stage labels plus the log-probability of the
spoilage interval under the posterior path, minus the Girsanov KL
``1/2 int |u|^2 dt`` with ``u = (f_post - f_prior) / sigma``, minus
``KL(q(z0) || p(z0))``. The interval term is ``log[P(crossed by hi) - P(crossed by lo)]``
(right-censored: ``log[1 - P(crossed by t_last)]``) evaluated with the
smoothed crossing indicator of :mod:`ripen.models.fpt`.

Prediction rolls the prior forward from the posterior state at ``t_now`` with
``n_samples`` Brownian paths and reports the Monte Carlo first-passage CDF.

Gradients through the solve use :func:`nsde.sde_solve` so ``method`` can be
``"backprop"``, ``"checkpoint"`` or ``"adjoint"``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nsde import sde_solve

from ..data.collate import Batch
from .clock import make_clock
from .encoders import SequenceEncoder, VisitEncoder
from .fpt import crossing_prob, running_max_from
from .heads import OrdinalHead


class LatentDynamics(eqx.Module):
    visit_enc: VisitEncoder
    seq_enc: SequenceEncoder
    z0_net: eqx.nn.MLP
    prior_drift: eqx.nn.MLP | Array
    post_drift: eqx.nn.MLP
    log_sigma: eqx.nn.MLP | Array
    readout_w: Array
    readout_b: Array
    head: OrdinalHead
    clock: eqx.Module
    # static configuration
    kind: str = eqx.field(static=True)  # "sde" | "ode"
    latent_dim: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    t_max: float = eqx.field(static=True)
    n_sub: int = eqx.field(static=True)
    drift_kind: str = eqx.field(static=True)  # "mlp" | "linear"
    sigma_kind: str = eqx.field(static=True)  # "mlp" | "const"
    method: str = eqx.field(static=True)
    solver: str = eqx.field(static=True)
    brownian: str = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    n_train_samples: int = eqx.field(static=True)
    fpt_beta: float = eqx.field(static=True)
    min_sigma: float = eqx.field(static=True)
    kl_weight: float = eqx.field(static=True)
    spoil_weight: float = eqx.field(static=True)
    sigma_prior: float | None = eqx.field(static=True)
    sigma_prior_weight: float = eqx.field(static=True)

    def __init__(
        self,
        feat_dim: int,
        n_groups: int,
        horizon: int,
        t_max: float,
        *,
        key,
        kind: str = "sde",
        latent_dim: int = 4,
        hidden: int = 64,
        visit_dim: int = 32,
        width: int = 32,
        encoder: str = "gru",
        use_images: bool = True,
        use_stages: bool = True,
        drift_kind: str = "mlp",
        sigma_kind: str = "mlp",
        clock: str = "arrhenius",
        clock_kwargs: dict | None = None,
        fixed_thresholds: bool = False,
        n_sub: int = 2,
        method: str = "checkpoint",
        solver: str = "euler",
        brownian: str = "virtual_tree",
        n_samples: int = 64,
        n_train_samples: int = 4,
        fpt_beta: float = 0.1,
        min_sigma: float = 1e-3,
        kl_weight: float = 1.0,
        spoil_weight: float = 1.0,
        sigma_prior: float | None = None,
        sigma_prior_weight: float = 0.0,
        init_sigma: float = 0.3,
    ):
        if kind not in ("sde", "ode"):
            raise ValueError(kind)
        ks = jax.random.split(key, 8)
        self.kind, self.latent_dim, self.horizon, self.t_max, self.n_sub = (
            kind,
            latent_dim,
            horizon,
            float(t_max),
            n_sub,
        )
        self.drift_kind, self.sigma_kind = drift_kind, sigma_kind
        self.method, self.solver, self.brownian, self.n_samples = method, solver, brownian, n_samples
        self.n_train_samples = n_train_samples
        self.fpt_beta, self.min_sigma, self.kl_weight, self.spoil_weight = fpt_beta, min_sigma, kl_weight, spoil_weight
        self.sigma_prior, self.sigma_prior_weight = sigma_prior, sigma_prior_weight
        self.visit_enc = VisitEncoder(feat_dim, visit_dim, use_images, use_stages, key=ks[0])
        self.seq_enc = SequenceEncoder(encoder, visit_dim, hidden, key=ks[1])
        self.z0_net = eqx.nn.MLP(hidden, 2 * latent_dim, hidden, 1, activation=jax.nn.gelu, key=ks[2])
        if drift_kind == "linear":
            self.prior_drift = jnp.ones(latent_dim, jnp.float32)  # unit speed; the clock sets the rate
        else:
            self.prior_drift = eqx.nn.MLP(latent_dim, latent_dim, width, 2, activation=jax.nn.tanh, key=ks[3])
        self.post_drift = eqx.nn.MLP(latent_dim + hidden + 1, latent_dim, width, 2, activation=jax.nn.tanh, key=ks[4])
        if sigma_kind == "const":
            self.log_sigma = jnp.full((latent_dim,), jnp.log(init_sigma), jnp.float32)
        else:
            self.log_sigma = eqx.nn.MLP(latent_dim, latent_dim, width, 1, activation=jax.nn.tanh, key=ks[5])
        self.readout_w = jnp.concatenate([jnp.ones(1), jnp.zeros(latent_dim - 1)]).astype(jnp.float32)
        self.readout_b = jnp.zeros((), jnp.float32)
        self.head = OrdinalHead(fixed=fixed_thresholds)
        self.clock = make_clock(clock, n_groups, **(clock_kwargs or {}))

    # ----- pieces -----------------------------------------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(round((self.t_max + self.horizon) * self.n_sub))

    @property
    def t_end(self) -> float:
        return self.t_max + self.horizon

    def grid(self) -> Array:
        return jnp.linspace(0.0, self.t_end, self.n_steps + 1, dtype=jnp.float32)

    def ripeness(self, z: Array) -> Array:
        return z @ self.readout_w + self.readout_b

    def sigma(self, z: Array) -> Array:
        if self.kind == "ode":
            return jnp.zeros(self.latent_dim)
        raw = self.log_sigma if self.sigma_kind == "const" else self.log_sigma(z)
        return jnp.exp(raw) + self.min_sigma

    def f_prior(self, z: Array, kappa: Array) -> Array:
        mu = self.prior_drift if self.drift_kind == "linear" else self.prior_drift(z)
        return kappa * mu

    def f_post(self, t: Array, z: Array, ctx: Array, kappa: Array) -> Array:
        return self.f_prior(z, kappa) + self.post_drift(jnp.concatenate([z, ctx, jnp.atleast_1d(t) / 30.0]))

    def encode(self, batch: Batch) -> tuple[Array, Array, Array]:
        """Context from the visible visits; ``q(z0)`` mean and log-variance. All ``[B, ...]``."""
        x = jax.vmap(jax.vmap(self.visit_enc))(batch.feat, batch.stage)
        ctx = jax.vmap(self.seq_enc)(x, batch.t, batch.visible)
        mv = jax.vmap(self.z0_net)(ctx)
        return ctx, mv[:, : self.latent_dim], mv[:, self.latent_dim :]

    # ----- one path --------------------------------------------------------------------------------------------
    def _solve_path(self, params, static, z0: Array, ctx: Array, t_now: Array, kappa: Array, key: Array) -> Array:
        """Posterior dynamics up to ``t_now``, prior after; returns ``[n_steps + 1, latent_dim + 1]`` (last = KL)."""
        D = self.latent_dim

        def drift(t, y, args):
            p, ctx_, t_now_, kappa_ = args
            m = eqx.combine(p, static)
            z = y[:D]
            fp = m.f_prior(z, kappa_)
            fq = m.f_post(t, z, ctx_, kappa_)
            active = t <= t_now_
            u = (fq - fp) / jnp.maximum(m.sigma(z), 1e-3) if m.kind == "sde" else jnp.zeros_like(fq)
            dkl = jnp.where(active, 0.5 * jnp.sum(u**2), 0.0)
            return jnp.concatenate([jnp.where(active, fq, fp), jnp.atleast_1d(dkl)])

        def diffusion(t, y, args):
            p, *_ = args
            m = eqx.combine(p, static)
            return jnp.diag(jnp.concatenate([m.sigma(y[:D]), jnp.zeros(1)]))

        y0 = jnp.concatenate([z0, jnp.zeros(1)])
        return sde_solve(
            drift,
            diffusion,
            y0,
            (params, ctx, t_now, kappa),
            key,
            t1=self.t_end,
            n_steps=self.n_steps,
            solver=self.solver,
            method=self.method,
            ts=self.grid(),
            brownian=self.brownian,
        )

    def _visit_idx(self, t: Array) -> Array:
        return jnp.clip(jnp.round(t * self.n_sub).astype(jnp.int32), 0, self.n_steps)

    # ----- training ---------------------------------------------------------------------------------------------
    def loss(self, batch: Batch, key: Array) -> Array:
        """Negative ELBO, averaged over ``n_train_samples`` posterior paths per item.

        The posterior drift is active up to ``t_now``; beyond it the path follows the prior, so when the
        spoilage interval lies after ``t_now`` (visible-prefix dropout) the spoilage term trains the prior as a
        forecaster. The crossing probability is the Monte Carlo mean over the sampled paths.
        """
        params, static = eqx.partition(self, eqx.is_inexact_array)
        B, M, V = batch.batch_size, self.n_train_samples, batch.t.shape[1]
        ctx, mu0, logvar0 = self.encode(batch)
        k_z0, k_path = jax.random.split(key)
        z0 = mu0[None] + jnp.exp(0.5 * logvar0)[None] * jax.random.normal(k_z0, (M, B, self.latent_dim))
        kl_z0 = 0.5 * jnp.sum(jnp.exp(logvar0) + mu0**2 - 1.0 - logvar0, axis=-1)  # [B]
        kappa = self.clock(batch.temp_k, batch.group)
        keys = jax.random.split(k_path, M * B).reshape(M, B, 2)

        def solve(z, c, tn, ka, kk):
            return self._solve_path(params, static, z, c, tn, ka, kk)

        ys = jax.vmap(jax.vmap(solve), in_axes=(0, None, None, None, 0))(z0, ctx, batch.t_now, kappa, keys)
        z_path, kl_path = ys[..., : self.latent_dim], ys[:, :, -1, -1]  # [M, B, T, D], [M, B]
        r_path = z_path @ self.readout_w + self.readout_b  # [M, B, T]
        # stage likelihood at visible labelled visits
        idx = jnp.broadcast_to(self._visit_idx(batch.t)[None], (M, B, V))
        r_at = jnp.take_along_axis(r_path, idx, axis=2)
        lp_stage = self.head.log_prob(r_at, jnp.broadcast_to(batch.stage[None], (M, B, V)))
        w = (batch.visible & (batch.stage > 0))[None]
        obs_ll = jnp.mean(jnp.sum(jnp.where(w, lp_stage, 0.0), axis=2), axis=0)  # [B]
        # spoilage interval likelihood: smoothed crossing indicator, averaged over paths
        rm = jax.vmap(jax.vmap(lambda r: running_max_from(r, 0)))(r_path)
        crossed = jnp.mean(crossing_prob(rm, self.head.boundary, self.fpt_beta), axis=0)  # [B, T]
        p_lo = jnp.take_along_axis(crossed, self._visit_idx(batch.spoil_lo)[:, None], 1)[:, 0]
        p_hi = jnp.take_along_axis(crossed, self._visit_idx(batch.spoil_hi)[:, None], 1)[:, 0]
        spoil_ll = jnp.where(
            batch.right_censored, jnp.log(1.0 - p_lo + 1e-6), jnp.log(jnp.clip(p_hi - p_lo, 1e-6, 1.0))
        )
        kl = jnp.mean(kl_path, axis=0)  # accumulated only while the posterior is active
        elbo = obs_ll + self.spoil_weight * spoil_ll - self.kl_weight * (kl + kl_z0)
        loss = -jnp.mean(elbo)
        if self.sigma_prior is not None and self.sigma_prior_weight > 0 and self.kind == "sde":
            s = (
                self.sigma(jnp.zeros(self.latent_dim))
                if self.sigma_kind == "const"
                else jnp.mean(jax.vmap(jax.vmap(self.sigma))(z_path), axis=(0, 1))
            )
            loss = loss + self.sigma_prior_weight * jnp.sum((jnp.log(s) - jnp.log(self.sigma_prior)) ** 2)
        return loss

    # ----- prediction -------------------------------------------------------------------------------------------
    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        """Monte Carlo first-passage CDF on the days-left grid, from the posterior state at ``t_now``."""
        params, static = eqx.partition(self, eqx.is_inexact_array)
        B, M = batch.batch_size, self.n_samples
        ctx, mu0, logvar0 = self.encode(batch)
        kappa = self.clock(batch.temp_k, batch.group)
        k_z0, k_path = jax.random.split(key)
        eps = jax.random.normal(k_z0, (M, B, self.latent_dim))
        z0 = mu0[None] + jnp.exp(0.5 * logvar0)[None] * eps
        keys = jax.random.split(k_path, M * B).reshape(M, B, 2)

        def one(z, c, tn, ka, kk):
            ys = self._solve_path(params, static, z, c, tn, ka, kk)
            return jax.vmap(self.ripeness)(ys[:, : self.latent_dim])

        r = jax.vmap(jax.vmap(one, in_axes=(0, 0, 0, 0, 0)), in_axes=(0, None, None, None, 0))(
            z0, ctx, batch.t_now, kappa, keys
        )
        # r: [M, B, T]
        idx_now = self._visit_idx(batch.t_now)
        rm = jax.vmap(jax.vmap(running_max_from), in_axes=(0, None))(r, idx_now)  # [M, B, T]
        crossed = crossing_prob(rm, self.head.boundary, None)
        h = jnp.arange(self.horizon + 1, dtype=jnp.float32)
        q_idx = jnp.clip(
            idx_now[:, None] + jnp.round(h[None, :] * self.n_sub).astype(jnp.int32), 0, self.n_steps
        )  # [B, H+1]
        F = jnp.mean(jnp.take_along_axis(crossed, q_idx[None], axis=2), axis=0)  # [B, H+1]
        return F

    # ----- introspection ----------------------------------------------------------------------------------------
    def drift_on_grid(self, z_scalar: Array, kappa: float = 1.0) -> Array:
        """Prior drift's ripeness component along the readout direction, for plotting / tests."""
        zs = z_scalar[:, None] * self.readout_w[None, :]
        return jax.vmap(lambda z: self.f_prior(z, jnp.asarray(kappa)) @ self.readout_w)(zs)

    def sigma_on_grid(self, z_scalar: Array) -> Array:
        zs = z_scalar[:, None] * self.readout_w[None, :]
        return jax.vmap(lambda z: jnp.sqrt(jnp.sum((self.sigma(z) * self.readout_w) ** 2)))(zs)
