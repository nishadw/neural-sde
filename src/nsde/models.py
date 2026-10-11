"""Neural SDE models: a drift network and a diagonal diffusion network."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class NeuralSDE(eqx.Module):
    """dZ = mu_theta(t, Z) dt + sigma_theta(t, Z) dW, with diagonal noise.

    Both networks take (t, z) concatenated. The diffusion is passed through a
    softplus so it stays positive, and is returned as a diagonal matrix so it
    can be used directly with ``diffrax.ControlTerm``.
    """

    drift_net: eqx.nn.MLP
    diffusion_net: eqx.nn.MLP
    state_dim: int = eqx.field(static=True)
    min_sigma: float = eqx.field(static=True)

    def __init__(self, state_dim: int, width: int = 32, depth: int = 2, *, key, min_sigma: float = 1e-3):
        k1, k2 = jax.random.split(key)
        self.state_dim = state_dim
        self.min_sigma = min_sigma
        self.drift_net = eqx.nn.MLP(state_dim + 1, state_dim, width, depth, activation=jax.nn.tanh, key=k1)
        self.diffusion_net = eqx.nn.MLP(state_dim + 1, state_dim, width, depth, activation=jax.nn.tanh, key=k2)

    def drift(self, t, z):
        return self.drift_net(jnp.concatenate([jnp.atleast_1d(t), z]))

    def diffusion(self, t, z):
        s = jax.nn.softplus(self.diffusion_net(jnp.concatenate([jnp.atleast_1d(t), z]))) + self.min_sigma
        return jnp.diag(s)


class LocalVolatility(eqx.Module):
    """Neural local-volatility surface sigma_theta(t, log-moneyness) > 0."""

    net: eqx.nn.MLP
    floor: float = eqx.field(static=True)
    x_scale: float = eqx.field(static=True)

    def __init__(self, width: int = 32, depth: int = 3, *, key, floor: float = 0.02, x_scale: float = 0.1):
        self.net = eqx.nn.MLP(2, "scalar", width, depth, activation=jax.nn.tanh, key=key)
        self.floor = floor
        self.x_scale = x_scale  # log-moneyness is O(0.1); rescale so the MLP sees O(1) inputs

    def __call__(self, t, x):
        return jax.nn.softplus(self.net(jnp.stack([t, x / self.x_scale])) - 1.0) * 0.5 + self.floor
