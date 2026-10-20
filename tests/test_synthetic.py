import math

import numpy as np

from ripen.data import SPOILED_STAGE, SyntheticConfig, simulate, subsample_visits
from ripen.data.synthetic import analytic_spoilage_cdf


def test_intervals_are_consistent_with_labels():
    items = simulate(SyntheticConfig(n_items=60, t_max=30.0, seed=3))
    for it in items:
        stages = [v.stage for v in it.visits]
        assert all(1 <= s <= SPOILED_STAGE for s in stages)
        if it.right_censored:
            assert stages[-1] < SPOILED_STAGE and it.spoil_lo == it.t_last
        else:
            assert stages[-1] == SPOILED_STAGE and it.spoil_hi == it.t_last
            assert all(s < SPOILED_STAGE for s in stages[:-1])
            assert it.spoil_lo < it.spoil_hi <= it.meta["tau"] + 1.0 and it.spoil_lo <= it.meta["tau"]


def test_first_passage_matches_inverse_gaussian():
    """The simulator's true spoilage times follow the analytic first-passage law."""
    cfg = SyntheticConfig(
        n_items=900, t_max=80.0, sigma=0.3, z0_sd=1e-6, rates={"T20": 0.473}, temps={"T20": 20.0}, seed=5
    )
    items = simulate(cfg)
    taus = np.array([it.meta["tau"] for it in items])
    assert np.isfinite(taus).all()
    grid = np.linspace(1, 40, 80)
    emp = np.array([(taus <= t).mean() for t in grid])
    ana = analytic_spoilage_cdf(grid, cfg.z0_mean, 0.473, 0.3)
    assert np.max(np.abs(emp - ana)) < 0.06


def test_subsampling_widens_interval_and_keeps_consistency():
    items = simulate(SyntheticConfig(n_items=30, t_max=30.0, seed=2))
    for it in items:
        sub = subsample_visits(it, 2)
        assert sub.visits[0].t == 0.0
        assert sub.n_visits <= it.n_visits
        if not sub.right_censored:
            assert sub.spoil_hi - sub.spoil_lo >= it.spoil_hi - it.spoil_lo
            assert sub.spoil_lo <= it.spoil_lo and sub.spoil_hi >= it.spoil_hi
        else:
            assert math.isinf(sub.spoil_hi)


def test_missing_visits_and_label_noise():
    items = simulate(SyntheticConfig(n_items=20, t_max=20.0, p_missing=0.4, label_noise=0.2, seed=9))
    assert all(it.visits[0].t == 0.0 for it in items)
    assert any(it.n_visits < 21 for it in items)
