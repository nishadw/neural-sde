import jax.numpy as jnp
import numpy as np

from ripen.models.fpt import crossing_prob, first_passage_cdf, inverse_gaussian_cdf, running_max_from


def test_running_max_ignores_segment_before_start():
    r = jnp.array([5.0, 0.0, 1.0, 0.5, 2.0])
    rm = running_max_from(r, 1)
    assert rm[0] == -jnp.inf
    assert np.allclose(np.asarray(rm[1:]), [0.0, 1.0, 1.0, 2.0])


def test_smoothed_indicator_tends_to_hard():
    rm = jnp.array([0.0, 0.9, 1.1, 3.0])
    hard = crossing_prob(rm, 1.0, None)
    soft = crossing_prob(rm, 1.0, 1e-3)
    assert np.allclose(np.asarray(hard), [0, 0, 1, 1])
    assert np.allclose(np.asarray(soft), np.asarray(hard), atol=1e-3)


def test_monte_carlo_first_passage_matches_analytic():
    rng = np.random.default_rng(0)
    M, T, dt, v, s, d = 4000, 2000, 0.01, 0.5, 0.3, 3.0
    inc = v * dt + s * np.sqrt(dt) * rng.normal(size=(M, T))
    paths = np.concatenate([np.zeros((M, 1)), np.cumsum(inc, 1)], 1)
    q = np.arange(0, T + 1, 100)
    F = np.asarray(
        first_passage_cdf(jnp.asarray(paths, jnp.float32), jnp.asarray(d), jnp.asarray(0), jnp.asarray(q), None)
    )
    ana = inverse_gaussian_cdf(q * dt, d, v, s)
    assert np.max(np.abs(F - ana)) < 0.03


def test_inverse_gaussian_cdf_limits():
    t = np.array([0.0, 1e-6, 1e6])
    F = inverse_gaussian_cdf(t, 2.0, 1.0, 0.5)
    assert F[0] == 0.0 and F[1] < 1e-6 and abs(F[2] - 1.0) < 1e-9
