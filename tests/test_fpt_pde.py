"""The PDE first-passage survival: matches the inverse-Gaussian law, is differentiable, agrees with Monte Carlo."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripen.models.fpt import inverse_gaussian_cdf
from ripen.models.fpt_pde import lookup, markov_interval_log_lik, survival_table

V, SIG, B, RMIN, DS = 0.473, 0.25, 5.0, -1.0, 0.02


@pytest.fixture(scope="module")
def table():
    S, r = survival_table(lambda r: jnp.asarray(V), lambda r: jnp.asarray(SIG), jnp.asarray(B), RMIN, 160, DS, 1500)
    return S, r


def test_table_shape_and_boundaries(table):
    S, r = table
    assert S.shape == (1501, 160) and r.shape == (160,)
    assert float(S[0, 0]) == 1.0 and float(S[0, -1]) == 0.0
    assert bool(jnp.all(S[:, -1] == 0.0))
    assert bool(jnp.all(jnp.diff(S, axis=0) <= 1e-6))  # survival is non-increasing in time
    assert bool(jnp.all((S >= 0) & (S <= 1)))


def test_matches_inverse_gaussian(table):
    S, r = table
    z0 = 1.2
    ts = np.array([3.0, 5.0, 6.5, 8.0, 10.0, 12.0, 15.0])
    pde = 1.0 - np.asarray(lookup(S, r, DS, jnp.asarray(ts), jnp.full(ts.shape, z0)))
    ig = inverse_gaussian_cdf(ts, B - z0, V, SIG)
    assert np.max(np.abs(pde - ig)) < 0.02, np.abs(pde - ig)


def test_grid_refinement():
    """Halving h and ds shrinks the inverse-Gaussian error (second-order scheme where central differences apply)."""
    z0, ts = 1.2, np.array([5.0, 6.5, 8.0, 10.0])
    ig = inverse_gaussian_cdf(ts, B - z0, V, SIG)
    errs = []
    for n, ds in [(80, 0.04), (160, 0.02), (320, 0.01)]:
        S, r = survival_table(
            lambda r: jnp.asarray(V), lambda r: jnp.asarray(SIG), jnp.asarray(B), RMIN, n, ds, int(12 / ds)
        )
        pde = 1.0 - np.asarray(lookup(S, r, ds, jnp.asarray(ts), jnp.full(ts.shape, z0)))
        errs.append(np.max(np.abs(pde - ig)))
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[1] < 0.5 * errs[0] + 1e-4 and errs[2] < 0.005, errs


def test_gradient_matches_finite_difference():
    def f(v):
        S, r = survival_table(lambda r: v, lambda r: jnp.asarray(SIG), jnp.asarray(B), RMIN, 120, DS, 800)
        return lookup(S, r, DS, jnp.asarray(8.0), jnp.asarray(1.2))

    g = float(jax.grad(f)(jnp.asarray(V)))
    fd = float((f(jnp.asarray(V + 1e-3)) - f(jnp.asarray(V - 1e-3))) / 2e-3)
    assert abs(g - fd) < 0.05 * max(abs(fd), 1e-3)


def test_markov_interval_likelihood_cases(table):
    S, r = table
    t = jnp.array([0.0, 2.0, 4.0, 6.0, 0.0])
    vis = jnp.array([1, 1, 1, 1, 0], bool)
    r_at = jnp.array([3.0, 3.8, 4.3, 4.6, 0.0])  # close enough to the boundary for the bridge factors to matter
    kappa = jnp.asarray(1.0)
    sig = jnp.full(t.shape, SIG)
    # event in (6, 8]: likely for r=4.6 with v~0.47 -> log-lik well above the floor
    ll_ev = markov_interval_log_lik(
        S, r, DS, kappa, t, r_at, vis, jnp.asarray(6.0), jnp.asarray(8.0), jnp.asarray(False), sig
    )
    # event in (6, 6.1]: unlikely -> much lower
    ll_fast = markov_interval_log_lik(
        S, r, DS, kappa, t, r_at, vis, jnp.asarray(6.0), jnp.asarray(6.1), jnp.asarray(False), sig
    )
    # right-censored at 6
    ll_rc = markov_interval_log_lik(
        S, r, DS, kappa, t, r_at, vis, jnp.asarray(6.0), jnp.asarray(1e6), jnp.asarray(True), sig
    )
    assert ll_ev > ll_fast
    assert ll_rc <= 0.0 and ll_rc > -3.0
    # a faster clock makes the right-censored observation less likely (more chance to cross between visits)
    ll_rc_fast = markov_interval_log_lik(
        S, r, DS, jnp.asarray(2.0), t, r_at, vis, jnp.asarray(6.0), jnp.asarray(1e6), jnp.asarray(True), sig
    )
    assert ll_rc_fast < ll_rc


def test_bridge_factor_matches_brownian_bridge_formula(table):
    """With both endpoints known, the no-crossing factor is the Brownian-bridge one, and it decreases as the endpoints
    approach the boundary."""
    S, r = table
    t = jnp.array([0.0, 1.0, 2.0])
    vis = jnp.array([1, 1, 1], bool)
    sig = jnp.full(3, SIG)
    far = jnp.array([1.0, 1.5, 2.0])
    near = jnp.array([4.3, 4.6, 4.8])
    ll_far = markov_interval_log_lik(
        S, r, DS, jnp.asarray(1.0), t, far, vis, jnp.asarray(2.0), jnp.asarray(1e6), jnp.asarray(True), sig
    )
    ll_near = markov_interval_log_lik(
        S, r, DS, jnp.asarray(1.0), t, near, vis, jnp.asarray(2.0), jnp.asarray(1e6), jnp.asarray(True), sig
    )
    assert ll_far > ll_near
    # explicit check of one bridge factor: r 4.3 -> 4.6 over 1 unit, sigma 0.25, kappa 1
    expected = np.log(1 - np.exp(-2 * (5 - 4.3) * (5 - 4.6) / (SIG**2 * 1.0)))
    ll_one = markov_interval_log_lik(
        S,
        r,
        DS,
        jnp.asarray(1.0),
        jnp.array([0.0, 1.0]),
        jnp.array([4.3, 4.6]),
        jnp.array([1, 1], bool),
        jnp.asarray(1.0),
        jnp.asarray(1e6),
        jnp.asarray(True),
        jnp.full(2, SIG),
    )
    s_last = float(lookup(S, r, DS, jnp.asarray(0.0), jnp.asarray(4.6)))
    assert abs(float(ll_one) - (expected + np.log(s_last))) < 1e-3


def test_zero_width_interval_has_gradient(table):
    S, r = table
    t = jnp.array([0.0, 0.0])
    vis = jnp.array([1, 0], bool)

    def f(r0):
        return markov_interval_log_lik(
            S,
            r,
            DS,
            jnp.asarray(1.0),
            t,
            jnp.array([r0, 0.0]),
            vis,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(False),
            jnp.full(2, SIG),
        )

    g = float(jax.grad(f)(jnp.asarray(4.7)))
    assert np.isfinite(g) and g != 0.0 and float(f(jnp.asarray(4.9))) > float(f(jnp.asarray(3.0)))


def test_pde_and_mc_predictions_agree(synthetic_items, synthetic_info):
    """Same latent model, two event estimators: the forecast CDFs should be close."""
    from ripen.data import collate
    from ripen.models import build_model

    batch = collate(synthetic_items[:6], synthetic_info, n_obs=2)
    common = {
        "name": "latent_sde",
        "horizon": 15,
        "latent_dim": 1,
        "drift_kind": "linear",
        "sigma_kind": "const",
        "fixed_thresholds": True,
        "clock": "group",
        "n_sub": 2,
        "n_samples": 128,
        "init_sigma": 0.25,
    }
    pde = build_model({**common, "event_likelihood": "pde"}, synthetic_info, jax.random.PRNGKey(0))
    mc = build_model({**common, "event_likelihood": "mc"}, synthetic_info, jax.random.PRNGKey(0))
    F_pde = np.asarray(pde.predict_cdf(batch, jax.random.PRNGKey(1)))
    F_mc = np.asarray(mc.predict_cdf(batch, jax.random.PRNGKey(1)))
    assert np.max(np.abs(F_pde - F_mc)) < 0.12, np.max(np.abs(F_pde - F_mc))
