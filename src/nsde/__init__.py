"""nsde: neural SDEs in JAX, with explicit control over how gradients are computed."""

from .finance import bs_call, bs_vega, heston_call, implied_vol
from .models import LocalVolatility, NeuralSDE
from .solve import make_adjoint, sde_solve

__all__ = [
    "NeuralSDE",
    "LocalVolatility",
    "sde_solve",
    "make_adjoint",
    "bs_call",
    "bs_vega",
    "implied_vol",
    "heston_call",
]
__version__ = "0.1.0"
