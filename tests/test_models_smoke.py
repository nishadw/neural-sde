"""Every model in the registry: builds, has a finite loss and gradient, and predicts a valid CDF."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripen.data import collate
from ripen.models import MODEL_REGISTRY, build_model

HORIZON = 12
EXTRA = {
    "latent_sde": {"n_samples": 8, "n_train_samples": 2, "n_sub": 1},
    "latent_ode": {"n_samples": 8, "n_train_samples": 2, "n_sub": 1},
}


@pytest.fixture(scope="module")
def batch(synthetic_items, synthetic_info):
    return collate(synthetic_items[:6], synthetic_info, n_obs="random", rng=np.random.default_rng(0))


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_model_one_batch(name, synthetic_info, batch):
    cfg = {"name": name, "horizon": HORIZON, **EXTRA.get(name, {})}
    model = build_model(cfg, synthetic_info, jax.random.PRNGKey(0))
    loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(model)
    assert np.isfinite(float(loss))
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0)) for g in leaves), "no parameter received gradient"
    F = np.asarray(model.predict_cdf(batch, jax.random.PRNGKey(2)))
    assert F.shape == (batch.batch_size, HORIZON + 1)
    assert (F >= -1e-6).all() and (F <= 1 + 1e-6).all()
    assert (np.diff(F, axis=1) >= -1e-6).all()


def test_sequence_encoders_ignore_hidden_visits(synthetic_items, synthetic_info):
    """Changing a visit beyond n_obs must not change the prediction."""
    items = synthetic_items[:4]
    b1 = collate(items, synthetic_info, n_obs=2)
    feat2 = b1.feat.at[:, 3:].set(99.0)
    b2 = eqx.tree_at(lambda b: b.feat, b1, feat2)
    for name in ("gru_gap", "gru_d", "ode_rnn"):
        model = build_model({"name": name, "horizon": HORIZON}, synthetic_info, jax.random.PRNGKey(0))
        F1, F2 = model.predict_cdf(b1, jax.random.PRNGKey(0)), model.predict_cdf(b2, jax.random.PRNGKey(0))
        assert np.allclose(np.asarray(F1), np.asarray(F2)), name


@pytest.mark.parametrize("method", ["backprop", "checkpoint", "adjoint"])
def test_latent_sde_gradient_methods_agree(method, synthetic_info, batch):
    """The three ways of differentiating the SDE solve give the same loss and close gradients."""
    cfg = {
        "name": "latent_sde",
        "horizon": HORIZON,
        "n_samples": 4,
        "n_train_samples": 1,
        "n_sub": 1,
        "solver": "reversible_heun",
        "brownian": "virtual_tree",
        "method": method,
    }
    model = build_model(cfg, synthetic_info, jax.random.PRNGKey(0))
    loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(model)
    ref_cfg = {**cfg, "method": "backprop"}
    ref = build_model(ref_cfg, synthetic_info, jax.random.PRNGKey(0))
    loss_ref, grads_ref = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(ref)
    assert float(loss) == pytest.approx(float(loss_ref), rel=1e-4)
    g1 = jnp.concatenate([g.ravel() for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))])
    g2 = jnp.concatenate([g.ravel() for g in jax.tree.leaves(eqx.filter(grads_ref, eqx.is_inexact_array))])
    rel = float(jnp.max(jnp.abs(g1 - g2)) / (jnp.max(jnp.abs(g2)) + 1e-12))
    # The continuous adjoint is an approximation whose error falls like 1/steps (see experiments/adjoint_tradeoffs.py);
    # with one step per day and a running-max spoilage term it lands near 1e-1 here. Checkpointing is exact.
    assert rel < (0.2 if method == "adjoint" else 1e-5)
