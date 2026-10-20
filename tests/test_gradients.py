import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nsde import NeuralSDE, sde_solve


@pytest.fixture(scope="module", autouse=True)
def _x64():
    """These tolerances need float64; switch it on for this module only."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _loss(model, key, method, solver="reversible_heun", n_steps=40):
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def drift(t, y, p):
        return eqx.combine(p, static).drift(t, y)

    def diffusion(t, y, p):
        return eqx.combine(p, static).diffusion(t, y)

    def f(p):
        y = sde_solve(drift, diffusion, jnp.ones(3), p, key, n_steps=n_steps, solver=solver, method=method)
        return jnp.sum(y**2)

    return jax.value_and_grad(f)(params)


@pytest.fixture(scope="module")
def model():
    return NeuralSDE(3, width=16, depth=2, key=jax.random.PRNGKey(0))


def _max_rel_err(g1, g2):
    leaves1, leaves2 = jax.tree.leaves(g1), jax.tree.leaves(g2)
    num = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves1, leaves2, strict=True))
    den = max(float(jnp.max(jnp.abs(b))) for b in leaves2)
    return num / den


def test_forward_values_match_across_methods(model):
    key = jax.random.PRNGKey(1)
    v_adj, _ = _loss(model, key, "adjoint")
    v_bp, _ = _loss(model, key, "backprop")
    assert jnp.allclose(v_adj, v_bp, rtol=1e-10)


def test_adjoint_close_to_backprop(model):
    """The continuous adjoint is an approximation of the exact (backprop) gradient."""
    key = jax.random.PRNGKey(2)
    _, g_adj = _loss(model, key, "adjoint")
    _, g_bp = _loss(model, key, "backprop")
    assert _max_rel_err(g_adj, g_bp) < 1e-3


def test_checkpointing_matches_full_backprop(model):
    key = jax.random.PRNGKey(3)
    _, g_ck = _loss(model, key, "checkpoint")
    _, g_bp = _loss(model, key, "backprop")
    assert _max_rel_err(g_ck, g_bp) < 1e-10


def test_diffusion_is_positive_diagonal(model):
    s = model.diffusion(0.3, jnp.zeros(3))
    assert s.shape == (3, 3)
    assert jnp.all(jnp.diag(s) > 0)
    assert jnp.allclose(s, jnp.diag(jnp.diag(s)))
