import jax.numpy as jnp
import numpy as np
import pytest

from ripen.eval.metrics import compute_metrics, harrell_cindex, quantile_from_cdf, reliability_curve
from ripen.models.base import interval_log_prob_from_logits, pmf_to_cdf, step_cdf


def test_crps_equals_mae_for_point_predictions():
    d_hat = jnp.array([3.0, 7.0, 0.0, 12.0])
    F = np.asarray(step_cdf(d_hat, 20))
    hi = np.array([4.0, 5.0, 2.0, 12.0])
    lo = hi - 1
    m = compute_metrics(F, lo, hi)
    assert m.mae == pytest.approx(np.mean(np.abs(np.asarray(d_hat) - hi)))
    assert m.crps == pytest.approx(m.mae)
    assert m.width80 == pytest.approx(0.0)


def test_quantiles_and_coverage():
    H = 20
    h = np.arange(H + 1)
    # uniform over days 0..10 -> F(h) = (h+1)/11
    F = np.clip((h + 1) / 11.0, 0, 1)[None].repeat(3, 0)
    q50 = quantile_from_cdf(F, 0.5)
    assert 4.0 <= q50[0] <= 5.5
    hi = np.array([5.0, 9.0, 15.0])
    m = compute_metrics(F, hi - 1, hi)
    assert m.coverage80 == pytest.approx(2 / 3)
    assert m.n_uncensored == 3


def test_censored_handling():
    F = np.asarray(step_cdf(jnp.array([3.0, 3.0]), 10))
    lo = np.array([2.0, 8.0])
    hi = np.array([3.0, np.inf])
    m = compute_metrics(F, lo, hi)
    assert m.n_uncensored == 1 and m.n == 2
    # censored instance predicted spoiled at day 3 but known unspoiled through day 8 -> penalised
    assert m.crps_censored > m.crps


def test_cindex_against_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(0)
    n = 80
    time = rng.exponential(5, n)
    event = rng.random(n) > 0.3
    score = -time + rng.normal(0, 2, n)  # higher score ~ earlier event
    ours = harrell_cindex(score, time, event)
    ref = lifelines.utils.concordance_index(time, -score, event)
    assert ours == pytest.approx(ref, abs=1e-9)


def test_reliability_curve_perfect_calibration():
    rng = np.random.default_rng(1)
    N, H = 4000, 15
    hi = rng.integers(1, 12, N).astype(float)
    # oracle CDF: a step at the true day -> predicted P in {0,1} and perfectly calibrated
    F = (np.arange(H + 1)[None, :] >= hi[:, None]).astype(float)
    _, _, _, ece = reliability_curve(F, hi - 1, hi)
    assert ece < 1e-9


def test_interval_log_prob_sums_bins():
    logits = jnp.log(jnp.array([[0.1, 0.2, 0.3, 0.25, 0.15]]))  # days 0,1,2,3 and "after 3"
    lo, hi = jnp.array([0.0]), jnp.array([2.0])
    lp = interval_log_prob_from_logits(logits, lo, hi, jnp.array([False]))
    assert float(lp[0]) == pytest.approx(np.log(0.2 + 0.3), abs=1e-5)
    lp_c = interval_log_prob_from_logits(logits, jnp.array([1.0]), jnp.array([1e6]), jnp.array([True]))
    assert float(lp_c[0]) == pytest.approx(np.log(0.3 + 0.25 + 0.15), abs=1e-5)
    lp_exact = interval_log_prob_from_logits(logits, jnp.array([0.0]), jnp.array([0.0]), jnp.array([False]))
    assert float(lp_exact[0]) == pytest.approx(np.log(0.1), abs=1e-5)
    F = np.asarray(pmf_to_cdf(logits))
    assert np.allclose(F[0], np.cumsum([0.1, 0.2, 0.3, 0.25]))
