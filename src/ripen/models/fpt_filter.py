"""Exact inference for the progression coordinate: a grid filter (continuous-state HMM) on the 1-D diffusion.

The progression ``R`` follows ``dR = mu(R) ds + sigma(R) dW`` in physiological time ``s = kappa t`` and is
killed at the boundary ``b``. Instead of a sampled, amortised posterior (the latent-SDE ELBO) the density of
``R`` is carried on a grid and propagated **exactly** (up to discretisation) between visits with the forward
Kolmogorov (Fokker-Planck) operator, the transpose of the generator used for the survival table:

    p_{k+1} = exp(kappa dt_k L^T) p_k           (last grid node = cemetery, collects killed mass)
    p_k <- p_k * e_k(r),  Z_k = sum p_k,  log L += log Z_k   (emission of what was observed at visit k)

Emissions: an ordinal stage label gives ``e(r) = P(stage | r)`` on live nodes and ``1[stage == K]`` on the
cemetery; a photo gives ``e(r) = sum_k q_k(x) P(stage=k | r)`` with ``q(x)`` a learned soft stage prediction
(a hybrid DNN-HMM emission); the censoring observation "known unspoiled at t" zeroes the cemetery and "known
spoiled at t" keeps only the cemetery. The marginal log-likelihood of stages, photos and the censoring
observation is the product of the ``Z_k`` — no variational gap, no Monte Carlo — and forecasting from the last
visit integrates the survival table against the filtered density. Everything is differentiable, so ``mu``,
``sigma``, the clock, the thresholds and the emission network are trained by exact maximum likelihood.

Propagation over an arbitrary physiological duration uses binary decomposition of ``round(s / ds_T)`` over
precomputed powers ``P^(2^j)`` of the one-step propagator, so a batch of items with different clocks and visit
gaps costs ``log2(K)`` matrix-vector products per interval.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..data.collate import Batch
from ..data.schema import N_STAGES
from .clock import make_clock
from .encoders import VisitEncoder
from .fpt_pde import generator_matrix, survival_table
from .heads import OrdinalHead


def forward_propagator_powers(L: Array, ds: float, n_pow: int) -> Array:
    """``[n_pow, N, N]``: ``P^(2^j)`` for ``j = 0..n_pow-1`` with ``P = expm(ds L^T)`` (Crank-Nicolson approximant).

    ``L`` is the discrete backward generator with the absorbing boundary row zeroed; its transpose conserves
    total mass and accumulates killed mass in the last node.
    """
    n = L.shape[0]
    lt = L.T
    eye = jnp.eye(n)
    p1 = jnp.linalg.solve(eye - 0.5 * ds * lt, eye + 0.5 * ds * lt)

    def square(p, _):
        return p @ p, p

    _, pows = jax.lax.scan(square, p1, None, length=n_pow)
    return pows


def propagate(pows: Array, p: Array, n_steps: Array) -> Array:
    """Apply ``P^n_steps`` to the density ``p`` by binary decomposition (``n_steps`` a traced int)."""
    n_pow = pows.shape[0]
    n_steps = jnp.clip(n_steps, 0, 2**n_pow - 1)

    def body(carry, j):
        q, n = carry
        bit = (n >> j) & 1
        q = jnp.where(bit == 1, pows[j] @ q, q)
        return (q, n), None

    (q, _), _ = jax.lax.scan(body, (p, n_steps), jnp.arange(n_pow))
    return jnp.clip(q, 0.0, None)


def propagate_time(pows: Array, p: Array, s_over_ds: Array) -> Array:
    """Propagate by a real physiological duration ``s`` (given as ``s / ds``): integer steps by binary
    decomposition, then linear interpolation to the next step, so the result is differentiable in ``s`` (and hence
    in the clock rate)."""
    x = jnp.clip(s_over_ds, 0.0, 2 ** pows.shape[0] - 1.0)
    n = jnp.floor(x).astype(jnp.int32)
    w = x - n
    q = propagate(pows, p, n)
    q1 = jnp.clip(pows[0] @ q, 0.0, None)
    return (1.0 - w) * q + w * q1


class FirstPassageFilter(eqx.Module):
    prior_r: eqx.nn.MLP | None
    log_sigma_r: eqx.nn.MLP | Array
    head: OrdinalHead
    clock: eqx.Module
    emission: eqx.nn.MLP | None  # photo features -> logits over N_STAGES (soft stage), None if no images
    visit_enc: VisitEncoder | None
    z0_mean: Array
    z0_logstd: Array
    horizon: int = eqx.field(static=True)
    drift_kind: str = eqx.field(static=True)
    sigma_kind: str = eqx.field(static=True)
    monotone_drift: bool = eqx.field(static=True)
    min_sigma: float = eqx.field(static=True)
    sigma_max: float = eqx.field(static=True)
    mu_max: float = eqx.field(static=True)
    kappa_max: float = eqx.field(static=True)
    use_images: bool = eqx.field(static=True)
    use_stages: bool = eqx.field(static=True)
    n_grid: int = eqx.field(static=True)
    r_min: float = eqx.field(static=True)
    ds: float = eqx.field(static=True)
    n_pow: int = eqx.field(static=True)
    pde_ds: float = eqx.field(static=True)
    pde_steps: int = eqx.field(static=True)

    def __init__(
        self,
        feat_dim: int,
        n_groups: int,
        horizon: int,
        *,
        key,
        use_images: bool = True,
        use_stages: bool = True,
        drift_kind: str = "mlp",
        sigma_kind: str = "mlp",
        monotone_drift: bool = True,
        width: int = 32,
        clock: str = "arrhenius",
        clock_kwargs: dict | None = None,
        fixed_thresholds: bool = False,
        link_scale: float = 0.5,
        learn_link_scale: bool = True,
        min_sigma: float = 1e-3,
        sigma_max: float = 1.0,
        mu_max: float = 2.0,
        kappa_max: float = 2.0,
        init_sigma: float = 0.3,
        z0_mean: float = 1.5,
        z0_std: float = 0.5,
        n_grid: int = 160,
        r_min: float = -1.0,
        ds: float = 0.02,
        n_pow: int = 11,
        pde_ds: float = 0.02,
        pde_steps: int = 2500,
        emission_width: int = 64,
        visit_dim: int = 32,
    ):
        if not (use_images or use_stages):
            raise ValueError("need images or stages")
        ks = jax.random.split(key, 4)
        tanh = jax.nn.tanh
        self.horizon, self.drift_kind, self.sigma_kind, self.monotone_drift = (
            horizon,
            drift_kind,
            sigma_kind,
            monotone_drift,
        )
        self.min_sigma, self.sigma_max, self.mu_max, self.kappa_max = min_sigma, sigma_max, mu_max, kappa_max
        self.use_images, self.use_stages = use_images, use_stages
        self.n_grid, self.r_min, self.ds, self.n_pow = n_grid, r_min, ds, n_pow
        self.pde_ds, self.pde_steps = pde_ds, pde_steps
        self.prior_r = None if drift_kind == "linear" else eqx.nn.MLP(1, "scalar", width, 2, activation=tanh, key=ks[0])
        if sigma_kind == "const":
            self.log_sigma_r = jnp.log(jnp.asarray(init_sigma, jnp.float32))
        else:
            self.log_sigma_r = eqx.nn.MLP(1, "scalar", width, 1, activation=tanh, key=ks[1])
        self.head = OrdinalHead(fixed=fixed_thresholds, link_scale=link_scale, learn_scale=learn_link_scale)
        self.clock = make_clock(clock, n_groups, **(clock_kwargs or {}))
        if use_images:
            self.visit_enc = VisitEncoder(feat_dim, visit_dim, True, False, key=ks[2])
            self.emission = eqx.nn.MLP(visit_dim, N_STAGES, emission_width, 1, activation=jax.nn.gelu, key=ks[3])
        else:
            self.visit_enc, self.emission = None, None
        self.z0_mean = jnp.asarray(z0_mean, jnp.float32)
        self.z0_logstd = jnp.log(jnp.asarray(z0_std, jnp.float32))

    # ----- dynamics --------------------------------------------------------------------------------------------
    def mu_r(self, r: Array) -> Array:
        if self.drift_kind == "linear":
            return jnp.ones_like(r)
        raw = self.prior_r(jnp.atleast_1d(r))
        return self.mu_max * (jax.nn.sigmoid(raw) if self.monotone_drift else jnp.tanh(raw))

    def sigma_r(self, r: Array) -> Array:
        if self.sigma_kind == "const":
            return jnp.clip(jnp.exp(self.log_sigma_r), self.min_sigma, self.sigma_max) * jnp.ones_like(r)
        return self.min_sigma + (self.sigma_max - self.min_sigma) * jax.nn.sigmoid(self.log_sigma_r(jnp.atleast_1d(r)))

    def rate(self, batch: Batch) -> Array:
        return jnp.clip(self.clock(batch.temp_k, batch.group), 1e-3, self.kappa_max)

    def operators(self) -> tuple[Array, Array, Array, Array]:
        """``(r_grid, forward propagator powers, survival table, L)`` for the current parameters."""
        L, r_grid, _ = generator_matrix(self.mu_r, self.sigma_r, self.head.boundary, self.r_min, self.n_grid)
        pows = forward_propagator_powers(L, self.ds, self.n_pow)
        S, _ = survival_table(
            self.mu_r, self.sigma_r, self.head.boundary, self.r_min, self.n_grid, self.pde_ds, self.pde_steps
        )
        return r_grid, pows, S, L

    def initial_density(self, r_grid: Array) -> Array:
        std = jnp.exp(self.z0_logstd)
        p = jnp.exp(-0.5 * ((r_grid - self.z0_mean) / std) ** 2).at[-1].set(0.0)
        return p / jnp.sum(p)

    # ----- emissions -------------------------------------------------------------------------------------------
    def stage_emission(self, r_grid: Array) -> Array:
        """``[N_STAGES, N]``: P(stage = k | r) on live nodes; the cemetery node emits only stage K."""
        lp = jax.vmap(self.head.log_probs)(r_grid)  # [N, K]
        e = jnp.exp(lp).T  # [K, N]
        e = e.at[:, -1].set(0.0).at[N_STAGES - 1, -1].set(1.0)
        return e

    def visit_emission(self, e_stage: Array, stage: Array, feat: Array, alive: Array, dead: Array) -> Array:
        """Emission vector over the grid for one visit (stage 0 = unlabelled)."""
        n = e_stage.shape[1]
        e = jnp.ones(n)
        if self.use_stages:
            e_lab = e_stage[jnp.clip(stage - 1, 0, N_STAGES - 1)]
            e = jnp.where(stage > 0, e_lab, e)
        if self.use_images:
            q = jax.nn.softmax(self.emission(self.visit_enc(feat, jnp.asarray(0))))  # [K]
            e = e * (q @ e_stage)
        cem = jnp.zeros(n).at[-1].set(1.0)
        live = 1.0 - cem
        e = jnp.where(alive, e * live, e)
        e = jnp.where(dead, e * cem, e)
        return e

    # ----- one item --------------------------------------------------------------------------------------------
    def _filter(self, r_grid, pows, e_stage, kappa, t, stage, feat, visible, spoil_lo, spoil_hi, right_censored):
        """Returns ``(log-likelihood, filtered live density at the last visible visit)``."""
        p0 = self.initial_density(r_grid)

        def step(carry, inp):
            p, logz, t_prev = carry
            tk, sk, fk, vis = inp
            p_prop = propagate_time(pows, p, kappa * jnp.clip(tk - t_prev, 0.0, None) / self.ds)
            alive = tk <= spoil_lo + 1e-6
            dead = (~right_censored) & (tk >= spoil_hi - 1e-6)
            e = self.visit_emission(e_stage, sk, fk, alive, dead)
            p_new = p_prop * e
            z = jnp.sum(p_new)
            p_new = p_new / jnp.maximum(z, 1e-30)
            p_out = jnp.where(vis, p_new, p)
            logz_out = jnp.where(vis, logz + jnp.log(jnp.maximum(z, 1e-30)), logz)
            t_out = jnp.where(vis, tk, t_prev)
            return (p_out, logz_out, t_out), None

        (p, logz, t_last), _ = jax.lax.scan(step, (p0, jnp.asarray(0.0), t[0]), (t, stage, feat, visible))
        # censoring observation beyond the last visible visit: survive to lo, then (cross by hi | right-censored).
        # If the scan already consumed the spoiled visit (its "dead" emission), the event is accounted for.
        p_lo = propagate_time(pows, p, kappa * jnp.clip(spoil_lo - t_last, 0.0, None) / self.ds)
        s_lo = jnp.sum(p_lo[:-1])
        x_hi = kappa * jnp.clip(spoil_hi - spoil_lo, 0.0, None) / self.ds
        x_hi = jnp.where(spoil_hi - spoil_lo < 1e-6, 1.0, x_hi)  # zero-width interval: one propagation step
        p_hi = propagate_time(pows, p_lo.at[-1].set(0.0), x_hi)
        killed = jnp.sum(p_hi[-1])
        consumed_dead = (~right_censored) & (t_last >= spoil_hi - 1e-6)
        term = jnp.where(right_censored, jnp.log(jnp.maximum(s_lo, 1e-30)), jnp.log(jnp.maximum(killed, 1e-30)))
        logz = logz + jnp.where(consumed_dead, 0.0, term)
        live = p.at[-1].set(0.0)
        return logz, live / jnp.maximum(jnp.sum(live), 1e-30)

    # ----- interface -------------------------------------------------------------------------------------------
    def loss(self, batch: Batch, key: Array) -> Array:
        r_grid, pows, _, _ = self.operators()
        e_stage = self.stage_emission(r_grid)
        kappa = self.rate(batch)
        ll, _ = jax.vmap(
            lambda ka, t, s, f, v, lo, hi, rc: self._filter(r_grid, pows, e_stage, ka, t, s, f, v, lo, hi, rc)
        )(kappa, batch.t, batch.stage, batch.feat, batch.visible, batch.spoil_lo, batch.spoil_hi, batch.right_censored)
        return -jnp.mean(ll)

    def predict_cdf(self, batch: Batch, key: Array) -> Array:
        r_grid, pows, S, _ = self.operators()
        e_stage = self.stage_emission(r_grid)
        kappa = self.rate(batch)
        # for forecasting, the history is the visible prefix only: no censoring information is used
        far = jnp.asarray(1e6, jnp.float32)
        _, live = jax.vmap(
            lambda ka, t, s, f, v: self._filter(r_grid, pows, e_stage, ka, t, s, f, v, far, far, jnp.asarray(True))
        )(kappa, batch.t, batch.stage, batch.feat, batch.visible)  # [B, N]
        h = jnp.arange(self.horizon + 1, dtype=jnp.float32)
        x = jnp.clip(kappa[:, None] * h[None, :] / self.pde_ds, 0.0, S.shape[0] - 1.0)  # [B, H+1], fractional rows
        k0 = jnp.floor(x).astype(jnp.int32)
        k1 = jnp.minimum(k0 + 1, S.shape[0] - 1)
        w = x - k0
        surv_rows = (1.0 - w)[..., None] * S[k0] + w[..., None] * S[k1]  # [B, H+1, N]
        surv = jnp.einsum("bhn,bn->bh", surv_rows, live)
        return 1.0 - surv

    def drift_on_grid(self, r: Array, kappa: float = 1.0) -> Array:
        return kappa * jax.vmap(self.mu_r)(r)

    def sigma_on_grid(self, r: Array) -> Array:
        return jax.vmap(self.sigma_r)(r)
