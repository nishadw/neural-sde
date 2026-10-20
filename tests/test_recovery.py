"""The fake-fruit milestone: a label-only latent SDE with a linear drift, a constant diffusion and a per-group
clock must recover the simulator's rates, its noise level and its first-passage distribution.

The latent is pinned to stage units by fixing the ordinal thresholds at 2, 3, 4, 5; otherwise latent scale,
rate and sigma are only identified up to a common factor.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ripen.data import SyntheticConfig, collate, dataset_info, make_splits, select, simulate
from ripen.data.synthetic import analytic_spoilage_cdf
from ripen.models import LatentDynamics
from ripen.train import TrainConfig, train


@pytest.mark.slow
def test_latent_sde_recovers_rates_sigma_and_fpt():
    scfg = SyntheticConfig(n_items=180, t_max=24.0, sigma=0.25, seed=11)
    items = simulate(scfg)
    sp = make_splits(items, seed=0)
    tr, va, te = select(items, sp["train"]), select(items, sp["val"]), select(items, sp["test"])
    info = dataset_info(items)
    model = LatentDynamics(
        info.feat_dim,
        len(info.groups),
        horizon=24,
        t_max=info.t_max,
        key=jax.random.PRNGKey(0),
        latent_dim=1,
        hidden=32,
        visit_dim=16,
        use_images=False,
        use_stages=True,
        drift_kind="linear",
        sigma_kind="const",
        clock="group",
        clock_kwargs={"init_rate": 0.4},
        fixed_thresholds=True,
        n_sub=2,
        n_train_samples=4,
        n_samples=128,
        init_sigma=0.4,
    )
    tcfg = TrainConfig(epochs=14, batch_size=32, lr=2e-2, patience=14, n_obs_mode="random", seed=0)
    model, hist = train(model, tr, va, info, tcfg, jax.random.PRNGKey(1), log=lambda *_: None)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]

    rates = {g: float(r) for g, r in zip(info.groups, np.asarray(model.clock.rates()), strict=True)}
    for g, true in scfg.rates.items():
        assert abs(rates[g] - true) / true < 0.25, (g, rates[g], true)
    sigma = float(np.asarray(model.sigma(jnp.zeros(1)))[0])
    assert abs(sigma - scfg.sigma) / scfg.sigma < 0.5, sigma

    fresh = [it.prefix(1) for it in te if it.storage == "T20"]
    F = np.asarray(model.predict_cdf(collate(fresh, info, n_obs=1), jax.random.PRNGKey(3))).mean(0)
    t = np.arange(F.shape[0], dtype=float)
    truth = np.mean([analytic_spoilage_cdf(t, it.meta["z0"], 0.473, scfg.sigma) for it in fresh], axis=0)
    assert np.max(np.abs(F - truth)) < 0.15, np.max(np.abs(F - truth))
