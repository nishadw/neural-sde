"""Temperature clocks: how fast ripening runs as a function of storage.

``ArrheniusClock`` is the food-science prior (Van Boekel 2008; Taoukis &
Labuza 1989): rate ``kappa(T) = kappa_ref * exp(-Ea/R (1/T - 1/T_ref))``.
``GroupClock`` learns one rate per storage group and is the escape hatch for
the avocado data, where Tamb (mean 18.7 C) ripened faster than T20 (20 C) so
a mean-temperature Arrhenius law cannot fit all three groups.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

R_GAS = 8.314  # J / (mol K)


class ArrheniusClock(eqx.Module):
    log_ea: Array  # log of activation energy in J/mol
    log_rate_ref: Array
    t_ref_k: float = eqx.field(static=True)
    learn_rate_ref: bool = eqx.field(static=True)

    def __init__(self, ea_j_mol: float = 50_000.0, rate_ref: float = 0.473, t_ref_c: float = 20.0, learn_rate_ref=True):
        self.log_ea = jnp.log(jnp.asarray(ea_j_mol, jnp.float32))
        self.log_rate_ref = jnp.log(jnp.asarray(rate_ref, jnp.float32))
        self.t_ref_k = t_ref_c + 273.15
        self.learn_rate_ref = learn_rate_ref

    @property
    def ea(self) -> Array:
        return jnp.exp(self.log_ea)

    def q10(self, t_c: float = 15.0) -> Array:
        """Rate ratio for a 10 C rise around ``t_c``."""
        return self.__call__(jnp.asarray(t_c + 10 + 273.15)) / self.__call__(jnp.asarray(t_c + 273.15))

    def __call__(self, temp_k: Array, group: Array | None = None) -> Array:
        return jnp.exp(self.log_rate_ref) * jnp.exp(-self.ea / R_GAS * (1.0 / temp_k - 1.0 / self.t_ref_k))


class GroupClock(eqx.Module):
    log_rates: Array

    def __init__(self, n_groups: int, init_rate: float = 0.4):
        self.log_rates = jnp.full((n_groups,), jnp.log(init_rate), jnp.float32)

    def rates(self) -> Array:
        return jnp.exp(self.log_rates)

    def __call__(self, temp_k: Array, group: Array) -> Array:
        return jnp.exp(self.log_rates)[group]


class FixedClock(eqx.Module):
    """``kappa == 1``: the drift network carries all the speed (no temperature information)."""

    def __call__(self, temp_k: Array, group: Array | None = None) -> Array:
        return jnp.ones_like(temp_k)


def make_clock(kind: str, n_groups: int, **kw):
    if kind == "arrhenius":
        return ArrheniusClock(**kw)
    if kind == "group":  # clock_kwargs are written for the Arrhenius clock; only init_rate applies here
        return GroupClock(n_groups, **{k: v for k, v in kw.items() if k == "init_rate"})
    if kind == "fixed":
        return FixedClock()
    raise ValueError(f"unknown clock {kind!r}")
