"""The exact grid filter: mass conservation, transport, zero-width intervals, leakage, and agreement with the law."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripen.data import SyntheticConfig, collate, dataset_info, simulate
from ripen.data.schema import Item, Visit
from ripen.models import build_model
from ripen.models.fpt import inverse_gaussian_cdf
from ripen.models.fpt_filter import forward_propagator_powers, propagate, propagate_time
from ripen.models.fpt_pde import generator_matrix

V, SIG, B, RMIN = 0.473, 0.25, 5.0, -1.0


@pytest.fixture(scope="module")
def ops():
    L, r, h = generator_matrix(lambda r: jnp.asarray(1.0), lambda r: jnp.asarray(SIG), jnp.asarray(B), RMIN, 160)
    return L, r, forward_propagator_powers(L, 0.02, 13)


def test_mass_conservation_and_cemetery(ops):
    L, r, pows = ops
    p0 = jnp.exp(-0.5 * ((r - 1.2) / 0.15) ** 2).at[-1].set(0.0)
    p0 = p0 / p0.sum()
    for n in (1, 10, 100, 500):
        q = propagate(pows, p0, jnp.asarray(n))
        assert abs(float(q.sum()) - 1.0) < 1e-4
        killed = float(q[-1])
        ig = float(inverse_gaussian_cdf(np.array([n * 0.02]), B - 1.2, 1.0, SIG)[0])
        assert abs(killed - ig) < 0.02, (n, killed, ig)


def test_transport_mean_and_variance(ops):
    L, r, pows = ops
    p0 = jnp.exp(-0.5 * ((r - 0.5) / 0.1) ** 2).at[-1].set(0.0)
    p0 = p0 / p0.sum()
    q = propagate_time(pows, p0, jnp.asarray(1.0 / 0.02))  # physiological time 1
    live = q[:-1] / q[:-1].sum()
    mean = float((r[:-1] * live).sum())
    var = float(((r[:-1] - mean) ** 2 * live).sum())
    assert abs(mean - 1.5) < 0.02 and abs(var - (0.01 + SIG**2)) < 0.01


def test_propagate_time_is_continuous_and_differentiable(ops):
    L, r, pows = ops
    p0 = jnp.exp(-0.5 * ((r - 1.0) / 0.2) ** 2).at[-1].set(0.0)
    p0 = p0 / p0.sum()
    f = lambda x: jnp.sum(propagate_time(pows, p0, x)[:-1])  # noqa: E731
    assert abs(float(f(jnp.asarray(11.0 - 1e-4))) - float(f(jnp.asarray(11.0 + 1e-4)))) < 1e-4
    g = float(jax.grad(f)(jnp.asarray(150.3)))
    assert np.isfinite(g) and g < 0  # more physiological time -> less live mass


def _filter_model(info, **kw):
    cfg = {
        "name": "fpt_filter",
        "horizon": 15,
        "use_images": False,
        "drift_kind": "linear",
        "sigma_kind": "const",
        "fixed_thresholds": True,
        "clock": "group",
        "link_scale": 0.1,
        "learn_link_scale": False,
        "init_sigma": 0.25,
        **kw,
    }
    return build_model(cfg, info, jax.random.PRNGKey(0))


def test_zero_width_interval_and_first_visit_spoiled():
    """An item seen spoiled at its very first visit has a finite likelihood with a finite, non-zero gradient."""
    it = Item("x", "synthetic", "T20", 20.0, [Visit(0.0, 5)], 0.0, 0.0)
    base = simulate(SyntheticConfig(n_items=6, t_max=10.0))
    info = dataset_info(base + [it])
    m = _filter_model(info)
    b = collate([it], info, n_obs="all")
    val = float(m.loss(b, jax.random.PRNGKey(0)))
    assert np.isfinite(val) and val < 40.0, val
    import equinox as eqx

    _, grads = eqx.filter_value_and_grad(lambda mm: mm.loss(b, jax.random.PRNGKey(0)))(m)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)
    assert any(bool(jnp.any(x != 0)) for x in leaves)


def test_predict_ignores_censoring_and_matches_law():
    items = simulate(SyntheticConfig(n_items=30, t_max=20.0, seed=4))
    info = dataset_info(items)
    import equinox as eqx

    m = _filter_model(info)
    m = eqx.tree_at(lambda mm: mm.clock.log_rates, m, jnp.log(jnp.array([0.228, 0.473, 0.518])))
    m = eqx.tree_at(lambda mm: mm.z0_mean, m, jnp.asarray(1.2))
    m = eqx.tree_at(lambda mm: mm.z0_logstd, m, jnp.log(jnp.asarray(0.15)))
    fresh = [it.prefix(1) for it in items if it.storage == "T20"]
    b = collate(fresh, info, n_obs=1)
    F = np.asarray(m.predict_cdf(b, jax.random.PRNGKey(0)))
    assert (np.diff(F, axis=1) >= -1e-6).all()
    b2 = eqx.tree_at(lambda bb: bb.spoil_hi, b, b.spoil_hi + 3.0)
    assert np.allclose(F, np.asarray(m.predict_cdf(b2, jax.random.PRNGKey(0))))  # no leakage
    h = np.arange(16.0)
    truth = np.mean([inverse_gaussian_cdf(h, 5 - it.meta["z0"], 0.473, 0.25 * np.sqrt(0.473)) for it in fresh], axis=0)
    assert np.max(np.abs(F.mean(0) - truth)) < 0.05
