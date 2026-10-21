"""The fake-fruit milestone: a label-only latent SDE with a linear drift, a constant diffusion and a per-group
clock must recover what the model identifies (docs/theory.md P3): the *ratios* of the group rates, the noise
level, the first-passage distribution, and — with two temperatures — the Arrhenius activation energy.

The latent is pinned to stage units by fixing the ordinal thresholds at 2, 3, 4, 5. The absolute rate scale is
confounded with the latent origin (the bottom stage is open below), so it is not asserted; the link scale is
fixed and sharp because the simulator's labels are hard floors of the latent.
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
        link_scale=0.1,
        learn_link_scale=False,
        n_sub=2,
        n_train_samples=4,
        n_samples=128,
        init_sigma=0.4,
    )
    tcfg = TrainConfig(epochs=14, batch_size=32, lr=2e-2, patience=14, n_obs_mode="random", seed=0)
    model, hist = train(model, tr, va, info, tcfg, jax.random.PRNGKey(1), log=lambda *_: None)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]

    rates = {g: float(r) for g, r in zip(info.groups, np.asarray(model.clock.rates()), strict=True)}
    for g in ("T10", "Tamb"):
        ratio, true_ratio = rates[g] / rates["T20"], scfg.rates[g] / scfg.rates["T20"]
        assert abs(ratio - true_ratio) / true_ratio < 0.15, (g, ratio, true_ratio)
    sigma = float(np.asarray(model.sigma(jnp.zeros(1)))[0])
    assert abs(sigma - scfg.sigma) / scfg.sigma < 0.5, sigma

    fresh = [it.prefix(1) for it in te if it.storage == "T20"]
    F = np.asarray(model.predict_cdf(collate(fresh, info, n_obs=1), jax.random.PRNGKey(3))).mean(0)
    t = np.arange(F.shape[0], dtype=float)
    truth = np.mean(
        [analytic_spoilage_cdf(t, it.meta["z0"], 0.473, scfg.calendar_sigma(0.473)) for it in fresh], axis=0
    )
    # the amortised-posterior ELBO leaves a gap on the common rate factor (docs/verifier_report.md §3), which shifts
    # the fresh-fruit forecast; the exact grid filter (ripen.models.fpt_filter) is held to 0.15 instead
    assert np.max(np.abs(F - truth)) < 0.3, np.max(np.abs(F - truth))


@pytest.mark.slow
def test_arrhenius_clock_recovers_activation_energy():
    """Two temperatures, rates following an Arrhenius law with E_a = 50 kJ/mol: the clock recovers E_a within 15%."""
    from ripen.models.clock import R_GAS

    ea, k_ref, t_ref = 50_000.0, 0.473, 20.0
    temps = {"T10": 10.0, "T20": 20.0}
    rates = {
        g: k_ref * float(np.exp(-ea / R_GAS * (1 / (c + 273.15) - 1 / (t_ref + 273.15)))) for g, c in temps.items()
    }
    scfg = SyntheticConfig(n_items=160, t_max=28.0, sigma=0.25, rates=rates, temps=temps, seed=5)
    items = simulate(scfg)
    sp = make_splits(items, seed=0)
    tr, va = select(items, sp["train"]), select(items, sp["val"])
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
        clock="arrhenius",
        clock_kwargs={"ea_j_mol": 30_000.0, "rate_ref": 0.3, "t_ref_c": t_ref},
        fixed_thresholds=True,
        link_scale=0.1,
        learn_link_scale=False,
        n_sub=2,
        n_train_samples=4,
        n_samples=32,
        init_sigma=0.4,
    )
    tcfg = TrainConfig(epochs=14, batch_size=32, lr=2e-2, patience=14, n_obs_mode="random", seed=0)
    model, _ = train(model, tr, va, info, tcfg, jax.random.PRNGKey(1), log=lambda *_: None)
    ea_hat = float(model.clock.ea)
    assert abs(ea_hat - ea) / ea < 0.15, ea_hat  # E_a depends on the rate ratio only, which is identified
