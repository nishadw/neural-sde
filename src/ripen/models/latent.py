"""The main model: a latent neural SDE (or ODE) whose event is the first passage of a progression coordinate.

Latent ``Z = (R, U)``: ``R`` is the scalar progression coordinate, ``U`` carries appearance.

    prior      dR = kappa(x) mu_R(R) dt + sqrt(kappa(x)) sigma_R(R) dW_R     (autonomous in R: "progression-structured")
               dU = kappa(x) mu_U(R, U) dt + sqrt(kappa(x)) sigma_U(R, U) dW_U
    i.e. the clock multiplies the generator: in physiological time s = kappa t the prior is dR = mu_R ds + sigma_R dW_s.
    posterior  drift += g_phi(Z, ctx, t)   for t <= t_now            (Li et al. 2020)
    stages     stage_k ~ Ordinal(R(t_k))                              (cumulative-link head, thresholds theta)
    features   y_k ~ N(dec(Z(t_k)), s^2 I)                            (optional; unlabeled sequences train the dynamics)
    event      tau = inf { t : R(t) >= b },  b = theta_{K-1}

Two likelihoods for the censoring observations (``event_likelihood``):

* ``"pde"`` (default, exact): the survival function of ``R`` in physiological time ``s = kappa t`` solves the
  backward Kolmogorov equation once per batch (:mod:`ripen.models.fpt_pde`); the interval-censored
  likelihood is the product of inter-visit bridge no-crossing factors and the last-interval survival difference
  (exact under the model), and forecasting from the last visit is a table look-up — no prior rollout.
* ``"mc"``: Monte Carlo paths rolled through the prior with a smoothed running-max crossing indicator (the
  baseline estimator; its gradient is biased by the smoothing).

Training maximises the latent-SDE ELBO: stage log-lik + feature log-lik + event log-lik - Girsanov KL
``1/2 int |(f_post - f_prior)/sigma|^2 dt`` - KL(q(z0) || p(z0)). Gradients through the posterior solve go
through :func:`nsde.sde_solve`, so ``method`` can be ``"backprop"``, ``"checkpoint"`` or ``"adjoint"``.
Setting ``structured=False`` recovers the unconstrained latent SDE (readout ``r = w . z + c``), for which
only the ``"mc"`` likelihood is available.
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
from .fpt_pde import lookup, markov_interval_log_lik, survival_table
from .heads import OrdinalHead


class LatentDynamics(eqx.Module):
    visit_enc: VisitEncoder
    seq_enc: SequenceEncoder
    z0_net: eqx.nn.MLP
    prior_r: eqx.nn.MLP | None  # structured: mu_R(r)
    log_sigma_r: eqx.nn.MLP | Array | None
    prior_u: eqx.nn.MLP | None  # structured: mu_U(z) for the appearance dims
    log_sigma_u: eqx.nn.MLP | Array | None
    prior_drift: eqx.nn.MLP | Array | None  # unstructured prior
    log_sigma: eqx.nn.MLP | Array | None
    post_drift: eqx.nn.MLP
    readout_w: Array
    readout_b: Array
    head: OrdinalHead
    clock: eqx.Module
    feature_dec: eqx.nn.MLP | None
    log_obs_scale: Array | None
    z0_prior_mean: Array
    z0_prior_logvar: Array
    z0_prior_learnable: bool = eqx.field(static=True)
    # static configuration
    kind: str = eqx.field(static=True)  # "sde" | "ode"
    latent_dim: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    t_max: float = eqx.field(static=True)
    n_sub: int = eqx.field(static=True)
    structured: bool = eqx.field(static=True)
    event_likelihood: str = eqx.field(static=True)  # "pde" | "mc"
    drift_kind: str = eqx.field(static=True)  # "mlp" | "linear"
    sigma_kind: str = eqx.field(static=True)  # "mlp" | "const"
    monotone_drift: bool = eqx.field(static=True)
    method: str = eqx.field(static=True)
    solver: str = eqx.field(static=True)
    brownian: str = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    n_train_samples: int = eqx.field(static=True)
    fpt_beta: float = eqx.field(static=True)
    min_sigma: float = eqx.field(static=True)
    sigma_max: float = eqx.field(static=True)
    mu_max: float = eqx.field(static=True)
    kappa_max: float = eqx.field(static=True)
    kl_weight: float = eqx.field(static=True)
    spoil_weight: float = eqx.field(static=True)
    sigma_prior: float | None = eqx.field(static=True)
    sigma_prior_weight: float = eqx.field(static=True)
    decode_features: bool = eqx.field(static=True)
    feat_dim: int = eqx.field(static=True)
    pde_grid: int = eqx.field(static=True)
    pde_ds: float = eqx.field(static=True)
    pde_steps: int = eqx.field(static=True)
    pde_r_min: float = eqx.field(static=True)

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
        structured: bool = True,
        event_likelihood: str = "pde",
        drift_kind: str = "mlp",
        sigma_kind: str = "mlp",
        monotone_drift: bool = True,
        clock: str = "arrhenius",
        clock_kwargs: dict | None = None,
        fixed_thresholds: bool = False,
        link_scale: float = 0.5,
        learn_link_scale: bool = True,
        n_sub: int = 2,
        method: str = "checkpoint",
        solver: str = "euler",
        brownian: str = "virtual_tree",
        n_samples: int = 64,
        n_train_samples: int = 4,
        fpt_beta: float = 0.1,
        min_sigma: float = 1e-3,
        sigma_max: float = 1.0,
        mu_max: float = 2.0,
        kappa_max: float = 2.0,
        kl_weight: float = 1.0,
        spoil_weight: float = 1.0,
        sigma_prior: float | None = None,
        sigma_prior_weight: float = 0.0,
        init_sigma: float = 0.3,
        decode_features: bool = False,
        z0_prior_mean: float = 0.0,
        z0_prior_std: float = 1.0,
        z0_prior_learnable: bool = True,
        pde_grid: int = 160,
        pde_ds: float = 0.02,
        pde_steps: int = 2500,
        pde_r_min: float = -1.0,
    ):
        if kind not in ("sde", "ode"):
            raise ValueError(kind)
        if event_likelihood not in ("pde", "mc"):
            raise ValueError(event_likelihood)
        if event_likelihood == "pde" and not structured:
            raise ValueError("the PDE likelihood needs the progression-structured prior (structured=True)")
        ks = jax.random.split(key, 10)
        self.kind, self.latent_dim, self.horizon, self.t_max, self.n_sub = (
            kind,
            latent_dim,
            horizon,
            float(t_max),
            n_sub,
        )
        self.structured, self.event_likelihood = structured, event_likelihood
        self.drift_kind, self.sigma_kind, self.monotone_drift = drift_kind, sigma_kind, monotone_drift
        self.method, self.solver, self.brownian = method, solver, brownian
        self.n_samples, self.n_train_samples, self.fpt_beta = n_samples, n_train_samples, fpt_beta
        self.min_sigma, self.sigma_max, self.mu_max, self.kappa_max = min_sigma, sigma_max, mu_max, kappa_max
        self.kl_weight, self.spoil_weight = kl_weight, spoil_weight
        self.sigma_prior, self.sigma_prior_weight = sigma_prior, sigma_prior_weight
        self.pde_grid, self.pde_ds, self.pde_steps, self.pde_r_min = pde_grid, pde_ds, pde_steps, pde_r_min
        self.visit_enc = VisitEncoder(feat_dim, visit_dim, use_images, use_stages, key=ks[0])
        self.seq_enc = SequenceEncoder(encoder, visit_dim, hidden, key=ks[1])
        self.z0_net = eqx.nn.MLP(hidden, 2 * latent_dim, hidden, 1, activation=jax.nn.gelu, key=ks[2])
        init_log_sigma = jnp.log(jnp.asarray(init_sigma, jnp.float32))
        tanh = jax.nn.tanh
        if structured:
            self.prior_r = (
                None if drift_kind == "linear" else eqx.nn.MLP(1, "scalar", width, 2, activation=tanh, key=ks[3])
            )
            if sigma_kind == "const":
                self.log_sigma_r = init_log_sigma
            else:
                self.log_sigma_r = eqx.nn.MLP(1, "scalar", width, 1, activation=tanh, key=ks[4])
            if latent_dim > 1:
                self.prior_u = eqx.nn.MLP(latent_dim, latent_dim - 1, width, 2, activation=tanh, key=ks[5])
                if sigma_kind == "const":
                    self.log_sigma_u = jnp.full((latent_dim - 1,), init_log_sigma)
                else:
                    self.log_sigma_u = eqx.nn.MLP(latent_dim, latent_dim - 1, width, 1, activation=tanh, key=ks[6])
            else:
                self.prior_u, self.log_sigma_u = None, None
            self.prior_drift, self.log_sigma = None, None
        else:
            self.prior_r = self.log_sigma_r = self.prior_u = self.log_sigma_u = None
            if drift_kind == "linear":
                self.prior_drift = jnp.ones(latent_dim, jnp.float32)
            else:
                self.prior_drift = eqx.nn.MLP(latent_dim, latent_dim, width, 2, activation=tanh, key=ks[3])
            if sigma_kind == "const":
                self.log_sigma = jnp.full((latent_dim,), init_log_sigma)
            else:
                self.log_sigma = eqx.nn.MLP(latent_dim, latent_dim, width, 1, activation=tanh, key=ks[4])
        self.readout_w = jnp.concatenate([jnp.ones(1), jnp.zeros(latent_dim - 1)]).astype(jnp.float32)
        self.readout_b = jnp.zeros((), jnp.float32)
        self.post_drift = eqx.nn.MLP(latent_dim + hidden + 1, latent_dim, width, 2, activation=tanh, key=ks[7])
        self.head = OrdinalHead(fixed=fixed_thresholds, link_scale=link_scale, learn_scale=learn_link_scale)
        self.clock = make_clock(clock, n_groups, **(clock_kwargs or {}))
        self.decode_features, self.feat_dim = decode_features, feat_dim
        # learnable Gaussian prior on z0: a fixed N(0, I) prior pulls the initial latent toward 0, which with
        # thresholds in stage units trades off against the clock rate (a lower start + a faster clock reproduce the
        # same transition times); letting the prior move removes that bias
        # the bottom stage is unbounded below, so an informative prior on the progression start (in stage units when
        # thresholds are fixed) is what anchors the level; a learnable prior adapts it to real data
        self.z0_prior_mean = jnp.full((latent_dim,), z0_prior_mean, jnp.float32).at[1:].set(0.0)
        self.z0_prior_logvar = jnp.full((latent_dim,), 2.0 * jnp.log(z0_prior_std), jnp.float32).at[1:].set(0.0)
        self.z0_prior_learnable = z0_prior_learnable
        if decode_features:
            self.feature_dec = eqx.nn.MLP(latent_dim, feat_dim, hidden, 1, activation=jax.nn.gelu, key=ks[8])
            self.log_obs_scale = jnp.zeros((), jnp.float32)
        else:
            self.feature_dec, self.log_obs_scale = None, None

    # ----- dynamics pieces ---------------------------------------------------------------------------------------
    @property
    def t_end(self) -> float:
        """Posterior solve horizon: the PDE handles the future, the MC estimator needs the prior rollout."""
        return self.t_max if self.event_likelihood == "pde" else self.t_max + self.horizon

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end * self.n_sub))

    def grid(self) -> Array:
        return jnp.linspace(0.0, self.t_end, self.n_steps + 1, dtype=jnp.float32)

    def mu_r(self, r: Array) -> Array:
        """Progression drift at unit clock rate (scalar in, scalar out); positive when ``monotone_drift``."""
        if self.drift_kind == "linear":
            return jnp.ones_like(r)
        raw = self.prior_r(jnp.atleast_1d(r))
        return self.mu_max * (jax.nn.sigmoid(raw) if self.monotone_drift else jnp.tanh(raw))

    def sigma_r(self, r: Array) -> Array:
        if self.kind == "ode":
            return jnp.zeros_like(r)
        if self.sigma_kind == "const":
            return jnp.clip(jnp.exp(self.log_sigma_r), self.min_sigma, self.sigma_max) * jnp.ones_like(r)
        raw = self.log_sigma_r(jnp.atleast_1d(r))
        return self.min_sigma + (self.sigma_max - self.min_sigma) * jax.nn.sigmoid(raw)

    def ripeness(self, z: Array) -> Array:
        return z @ self.readout_w + self.readout_b

    def sigma(self, z: Array) -> Array:
        if self.kind == "ode":
            return jnp.zeros(self.latent_dim)
        if self.structured:
            s_r = jnp.atleast_1d(self.sigma_r(z[0]))
            if self.latent_dim == 1:
                return s_r
            if self.sigma_kind == "const":
                s_u = jnp.clip(jnp.exp(self.log_sigma_u), self.min_sigma, self.sigma_max)
            else:
                s_u = self.min_sigma + (self.sigma_max - self.min_sigma) * jax.nn.sigmoid(self.log_sigma_u(z))
            return jnp.concatenate([s_r, s_u])
        raw = self.log_sigma if self.sigma_kind == "const" else self.log_sigma(z)
        return jnp.exp(raw) + self.min_sigma

    def f_prior(self, z: Array, kappa: Array) -> Array:
        if self.structured:
            parts = [jnp.atleast_1d(self.mu_r(z[0]))]
            if self.latent_dim > 1:
                parts.append(self.prior_u(z))
            return kappa * jnp.concatenate(parts)
        mu = self.prior_drift if self.drift_kind == "linear" else self.prior_drift(z)
        return kappa * mu

    def f_post(self, t: Array, z: Array, ctx: Array, kappa: Array) -> Array:
        return self.f_prior(z, kappa) + self.post_drift(jnp.concatenate([z, ctx, jnp.atleast_1d(t) / 30.0]))

    def rate(self, batch: Batch) -> Array:
        return jnp.clip(self.clock(batch.temp_k, batch.group), 1e-3, self.kappa_max)

    def pde_table(self) -> tuple[Array, Array]:
        """Survival table of the progression coordinate in physiological time (one solve per call)."""
        return survival_table(
            self.mu_r, self.sigma_r, self.head.boundary, self.pde_r_min, self.pde_grid, self.pde_ds, self.pde_steps
        )

    def encode(self, batch: Batch) -> tuple[Array, Array, Array]:
        """Context from the visible visits; ``q(z0)`` mean and log-variance. All ``[B, ...]``."""
        x = jax.vmap(jax.vmap(self.visit_enc))(batch.feat, batch.stage)
        ctx = jax.vmap(self.seq_enc)(x, batch.t, batch.visible)
        mv = jax.vmap(self.z0_net)(ctx)
        return ctx, mv[:, : self.latent_dim], mv[:, self.latent_dim :]

    # ----- one path ------------------------------------------------------------------------------------------------
    def _solve_path(self, params, static, z0: Array, ctx: Array, t_now: Array, kappa: Array, key: Array) -> Array:
        """Posterior dynamics up to ``t_now``, prior after; returns ``[n_steps + 1, latent_dim + 1]`` (last = KL)."""
        D = self.latent_dim

        def drift(t, y, args):
            p, ctx_, t_now_, kappa_ = args
            m = eqx.combine(p, static)
            z = y[:D]
            fp = m.f_prior(z, kappa_)
            # latent ODE: the path is the prior ODE from z0 ~ q(z0); a posterior drift would have no KL cost
            # (sigma = 0) and, without stage labels to anchor R (PBC2), drives R off the PDE grid
            fq = m.f_post(t, z, ctx_, kappa_) if m.kind == "sde" else fp
            active = t <= t_now_
            # clock on the generator: calendar-time diffusion is sqrt(kappa) * sigma (physiological units)
            s_cal = jnp.sqrt(kappa_) * m.sigma(z)
            u = (fq - fp) / jnp.maximum(s_cal, 1e-3) if m.kind == "sde" else jnp.zeros_like(fq)
            dkl = jnp.where(active, 0.5 * jnp.sum(u**2), 0.0)
            return jnp.concatenate([jnp.where(active, fq, fp), jnp.atleast_1d(dkl)])

        def diffusion(t, y, args):
            p, _ctx, _t_now, kappa_ = args
            m = eqx.combine(p, static)
            return jnp.diag(jnp.concatenate([jnp.sqrt(kappa_) * m.sigma(y[:D]), jnp.zeros(1)]))

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

    def _posterior_paths(self, batch: Batch, key: Array, n: int):
        """``n`` posterior paths per item: ``(z_path [n, B, T, D], kl [n, B], kl_z0 [B], kappa [B])``."""
        params, static = eqx.partition(self, eqx.is_inexact_array)
        B = batch.batch_size
        ctx, mu0, logvar0 = self.encode(batch)
        k_z0, k_path = jax.random.split(key)
        z0 = mu0[None] + jnp.exp(0.5 * logvar0)[None] * jax.random.normal(k_z0, (n, B, self.latent_dim))
        pm, plv = self.z0_prior_mean, self.z0_prior_logvar
        if not self.z0_prior_learnable:
            pm, plv = jax.lax.stop_gradient(pm), jax.lax.stop_gradient(plv)
        kl_z0 = 0.5 * jnp.sum(jnp.exp(logvar0 - plv) + (mu0 - pm) ** 2 / jnp.exp(plv) - 1.0 + plv - logvar0, axis=-1)
        kappa = self.rate(batch)
        keys = jax.random.split(k_path, n * B).reshape(n, B, 2)

        def solve(z, c, tn, ka, kk):
            return self._solve_path(params, static, z, c, tn, ka, kk)

        ys = jax.vmap(jax.vmap(solve), in_axes=(0, None, None, None, 0))(z0, ctx, batch.t_now, kappa, keys)
        return ys[..., : self.latent_dim], ys[:, :, -1, -1], kl_z0, kappa

    # ----- training ------------------------------------------------------------------------------------------------
    def loss(self, batch: Batch, key: Array) -> Array:
        """Negative ELBO averaged over ``n_train_samples`` posterior paths per item."""
        B, M, V = batch.batch_size, self.n_train_samples, batch.t.shape[1]
        z_path, kl_path, kl_z0, kappa = self._posterior_paths(batch, key, M)
        r_path = z_path @ self.readout_w + self.readout_b  # [M, B, T]
        idx = jnp.broadcast_to(self._visit_idx(batch.t)[None], (M, B, V))
        r_at = jnp.take_along_axis(r_path, idx, axis=2)  # [M, B, V]
        # stage likelihood at visible labelled visits
        lp_stage = self.head.log_prob(r_at, jnp.broadcast_to(batch.stage[None], (M, B, V)))
        w = (batch.visible & (batch.stage > 0))[None]
        obs_ll = jnp.mean(jnp.sum(jnp.where(w, lp_stage, 0.0), axis=2), axis=0)  # [B]
        if self.decode_features:
            z_at = jnp.take_along_axis(z_path, idx[..., None], axis=2)  # [M, B, V, D]
            mu_f = jax.vmap(jax.vmap(jax.vmap(self.feature_dec)))(z_at)
            s2 = jnp.exp(2.0 * self.log_obs_scale)
            lp_f = -0.5 * jnp.sum((batch.feat[None] - mu_f) ** 2, axis=-1) / s2 - 0.5 * self.feat_dim * (
                2.0 * self.log_obs_scale + jnp.log(2.0 * jnp.pi)
            )
            obs_ll = obs_ll + jnp.mean(jnp.sum(jnp.where(batch.visible[None], lp_f, 0.0), axis=2), axis=0)
        # event (censoring) likelihood
        if self.event_likelihood == "pde":
            S, r_grid = self.pde_table()

            sig_at = jax.vmap(jax.vmap(jax.vmap(self.sigma_r)))(r_at)  # [M, B, V] physiological diffusion

            def one(kap, t, r_v, vis, lo, hi, rc, sg):
                return markov_interval_log_lik(S, r_grid, self.pde_ds, kap, t, r_v, vis, lo, hi, rc, sg)

            over_items = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
            ll = jax.vmap(over_items, in_axes=(None, None, 0, None, None, None, None, 0))(
                kappa, batch.t, r_at, batch.visible, batch.spoil_lo, batch.spoil_hi, batch.right_censored, sig_at
            )  # [M, B]
            spoil_ll = jnp.mean(ll, axis=0)
        else:
            rm = jax.vmap(jax.vmap(lambda r: running_max_from(r, 0)))(r_path)
            crossed = jnp.mean(crossing_prob(rm, self.head.boundary, self.fpt_beta), axis=0)  # [B, T]
            p_lo = jnp.take_along_axis(crossed, self._visit_idx(batch.spoil_lo)[:, None], 1)[:, 0]
            p_hi = jnp.take_along_axis(crossed, self._visit_idx(batch.spoil_hi)[:, None], 1)[:, 0]
            spoil_ll = jnp.where(
                batch.right_censored, jnp.log(1.0 - p_lo + 1e-6), jnp.log(jnp.clip(p_hi - p_lo, 1e-6, 1.0))
            )
        kl = jnp.mean(kl_path, axis=0)
        elbo = obs_ll + self.spoil_weight * spoil_ll - self.kl_weight * (kl + kl_z0)
        loss = -jnp.mean(elbo)
        if self.sigma_prior is not None and self.sigma_prior_weight > 0 and self.kind == "sde":
            s = jnp.mean(self.sigma_on_grid(jnp.linspace(1.0, 5.0, 9)))
            loss = loss + self.sigma_prior_weight * (jnp.log(s) - jnp.log(self.sigma_prior)) ** 2
        return loss

    # ----- prediction ----------------------------------------------------------------------------------------------
    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        """``[B, horizon + 1]`` CDF of days-left from the last visible visit."""
        M = self.n_samples
        z_path, _, _, kappa = self._posterior_paths(batch, key, M)
        r_path = z_path @ self.readout_w + self.readout_b  # [M, B, T]
        idx_now = self._visit_idx(batch.t_now)
        h = jnp.arange(self.horizon + 1, dtype=jnp.float32)
        if self.event_likelihood == "pde":
            S, r_grid = self.pde_table()
            r_now = jnp.take_along_axis(r_path, idx_now[None, :, None], axis=2)[:, :, 0]  # [M, B]
            s_q = jnp.broadcast_to(kappa[None, :, None] * h[None, None, :], (M, batch.batch_size, h.shape[0]))
            surv = lookup(S, r_grid, self.pde_ds, s_q, jnp.broadcast_to(r_now[..., None], s_q.shape))
            return 1.0 - jnp.mean(surv, axis=0)
        rm = jax.vmap(jax.vmap(running_max_from), in_axes=(0, None))(r_path, idx_now)  # [M, B, T]
        crossed = crossing_prob(rm, self.head.boundary, None)
        q_idx = jnp.clip(idx_now[:, None] + jnp.round(h[None, :] * self.n_sub).astype(jnp.int32), 0, self.n_steps)
        return jnp.mean(jnp.take_along_axis(crossed, q_idx[None], axis=2), axis=0)

    # ----- introspection -------------------------------------------------------------------------------------------
    def drift_on_grid(self, r: Array, kappa: float = 1.0) -> Array:
        """Prior progression drift along the readout direction, for plotting / tests."""
        if self.structured:
            return kappa * jax.vmap(self.mu_r)(r)
        zs = r[:, None] * self.readout_w[None, :]
        return jax.vmap(lambda z: self.f_prior(z, jnp.asarray(kappa)) @ self.readout_w)(zs)

    def sigma_on_grid(self, r: Array) -> Array:
        if self.structured:
            return jax.vmap(self.sigma_r)(r)
        zs = r[:, None] * self.readout_w[None, :]
        return jax.vmap(lambda z: jnp.sqrt(jnp.sum((self.sigma(z) * self.readout_w) ** 2)))(zs)
