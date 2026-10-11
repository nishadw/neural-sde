"""Black-Scholes, implied volatility, and semi-analytic Heston prices.

Used to build a "market" implied-volatility smile (from Heston) that a neural
local-volatility SDE is then calibrated to.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy import integrate


# ----------------------------------------------------------------- Black-Scholes
def bs_call(S0, K, T, r, sigma):
    """Black-Scholes European call price (vectorised, differentiable)."""
    sqrtT = jnp.sqrt(T)
    d1 = (jnp.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


def bs_vega(S0, K, T, r, sigma):
    d1 = (jnp.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    return S0 * norm.pdf(d1) * jnp.sqrt(T)


@jax.jit
def implied_vol(price, S0, K, T, r):
    """Implied volatility by bisection on [1e-4, 3]. Vectorised over inputs."""
    lo = jnp.full(jnp.broadcast_shapes(jnp.shape(price), jnp.shape(K), jnp.shape(T)), 1e-4)
    hi = jnp.full_like(lo, 3.0)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_high = bs_call(S0, K, T, r, mid) > price
        return jnp.where(too_high, lo, mid), jnp.where(too_high, mid, hi)

    lo, hi = jax.lax.fori_loop(0, 100, body, (lo, hi))
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------ Heston
def heston_call(S0, K, T, r, v0, kappa, theta, xi, rho):
    """Heston (1993) call price by Gil-Pelaez inversion.

    Uses the "little Heston trap" form of the characteristic function
    (Albrecher et al., 2007), which is numerically stable for long maturities.
    Scalar inputs; numpy/scipy (used only to build targets, not differentiated).
    """

    def cf(u):
        iu = 1j * u
        d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi**2 * (iu + u**2))
        g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
        e = np.exp(-d * T)
        C = r * iu * T + kappa * theta / xi**2 * ((kappa - rho * xi * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (kappa - rho * xi * iu - d) / xi**2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0 + iu * np.log(S0))

    lnK = np.log(K)

    def p1_int(u):
        return (np.exp(-1j * u * lnK) * cf(u - 1j) / (1j * u * cf(-1j))).real

    def p2_int(u):
        return (np.exp(-1j * u * lnK) * cf(u) / (1j * u)).real

    P1 = 0.5 + integrate.quad(p1_int, 1e-8, 200, limit=500)[0] / np.pi
    P2 = 0.5 + integrate.quad(p2_int, 1e-8, 200, limit=500)[0] / np.pi
    return S0 * P1 - K * np.exp(-r * T) * P2
