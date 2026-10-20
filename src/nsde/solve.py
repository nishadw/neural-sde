"""Thin wrapper around diffrax for solving and differentiating SDEs.

The point of this module is to make the gradient method an explicit choice:

* ``"backprop"``  - discretize-then-differentiate. Reverse-mode AD through every
  solver step (``RecursiveCheckpointAdjoint`` with one checkpoint per step).
  Exact gradients of the numerical solution; memory grows with the step count.
* ``"checkpoint"`` - the same, but with diffrax's default recursive
  checkpointing. Memory grows roughly logarithmically, at extra compute.
* ``"adjoint"``   - differentiate-then-discretize. The stochastic adjoint
  (``BacksolveAdjoint``): solve a backward SDE for the gradient. Memory is
  constant in the step count. Exact only with an algebraically reversible
  solver such as reversible Heun.

Parameters that need gradients must be passed through ``args`` (not closed
over), otherwise ``BacksolveAdjoint`` cannot see them.
"""

from __future__ import annotations

from collections.abc import Callable

import diffrax as dfx
import jax.numpy as jnp

SOLVERS = {
    "euler": dfx.Euler,
    "heun": dfx.Heun,
    "reversible_heun": dfx.ReversibleHeun,
}


def make_adjoint(method: str, n_steps: int):
    if method == "backprop":
        return dfx.RecursiveCheckpointAdjoint(checkpoints=n_steps)
    if method == "checkpoint":
        return dfx.RecursiveCheckpointAdjoint()
    if method == "adjoint":
        return dfx.BacksolveAdjoint()
    raise ValueError(f"unknown gradient method {method!r}")


def sde_solve(
    drift: Callable,
    diffusion: Callable,
    y0,
    args,
    key,
    *,
    t1: float = 1.0,
    n_steps: int = 100,
    solver: str = "reversible_heun",
    method: str = "adjoint",
    ts=None,
    bm_tol: float | None = None,
    brownian: str = "virtual_tree",
):
    """Solve dY = drift(t, Y, args) dt + diffusion(t, Y, args) dW on [0, t1].

    ``diffusion`` must return a (state_dim, state_dim) matrix.
    Returns the solution at ``ts`` (default: only at t1).
    """
    dt = t1 / n_steps
    shape = jnp.shape(y0)
    if brownian == "virtual_tree":
        bm = dfx.VirtualBrownianTree(0.0, t1, tol=bm_tol if bm_tol is not None else dt / 8, shape=shape, key=key)
    elif brownian == "unsafe":  # fast, fixed-step; diffrax >= 0.7 rejects it under reverse-mode adjoints
        bm = dfx.UnsafeBrownianPath(shape=shape, key=key)
    else:
        raise ValueError(brownian)
    terms = dfx.MultiTerm(dfx.ODETerm(drift), dfx.ControlTerm(diffusion, bm))
    saveat = dfx.SaveAt(ts=ts) if ts is not None else dfx.SaveAt(t1=True)
    sol = dfx.diffeqsolve(
        terms,
        SOLVERS[solver](),
        0.0,
        t1,
        dt0=dt,
        y0=y0,
        args=args,
        saveat=saveat,
        adjoint=make_adjoint(method, n_steps),
        max_steps=n_steps + 1,
    )
    return sol.ys
