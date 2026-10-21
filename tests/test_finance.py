import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nsde import bs_call, heston_call, implied_vol


@pytest.fixture(scope="module", autouse=True)
def _x64():
    """The pricer round-trips need float64; switch it on for this module only (a session-wide toggle leaks)."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def test_put_call_parity_and_bounds():
    S0, K, T, r, s = 100.0, 95.0, 0.7, 0.03, 0.25
    c = bs_call(S0, K, T, r, s)
    assert max(S0 - K * np.exp(-r * T), 0) < c < S0


def test_implied_vol_round_trip():
    K = jnp.linspace(0.7, 1.3, 13)
    T = 0.5
    vols = 0.15 + 0.2 * (K - 1.0) ** 2
    prices = bs_call(1.0, K, T, 0.01, vols)
    recovered = implied_vol(prices, 1.0, K, T, 0.01)
    assert jnp.max(jnp.abs(recovered - vols)) < 1e-8


def test_heston_reduces_to_black_scholes_without_vol_of_vol():
    # xi -> 0 with v0 = theta: variance is constant, so Heston == Black-Scholes.
    v = 0.04
    for K in (0.8, 1.0, 1.25):
        h = heston_call(1.0, K, 1.0, 0.02, v, 2.0, v, 1e-4, -0.5)
        b = float(bs_call(1.0, K, 1.0, 0.02, np.sqrt(v)))
        assert abs(h - b) < 1e-5


def test_heston_negative_correlation_gives_downside_skew():
    p = dict(S0=1.0, T=0.5, r=0.0, v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
    iv = {}
    for K in (0.85, 1.15):
        price = heston_call(K=K, **p)
        iv[K] = float(implied_vol(price, 1.0, K, 0.5, 0.0))
    assert iv[0.85] > iv[1.15]
