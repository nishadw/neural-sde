"""Memory, time, and gradient accuracy of three ways to differentiate an SDE solve.

For a neural SDE (16-dim state, 64 sample paths) and an increasing number of
solver steps, compare:

  backprop    reverse-mode AD through every step (one checkpoint per step)
  checkpoint  diffrax's recursive checkpointing (default RecursiveCheckpointAdjoint)
  adjoint     the stochastic adjoint (BacksolveAdjoint), optimise-then-discretise

Memory is XLA's compiled temporary-buffer size for the gradient computation.
Gradient error is measured against full backprop (exact for the discretised solve).

Usage:  python experiments/adjoint_tradeoffs.py [--quick]
"""

import argparse
import json
import pathlib
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nsde import NeuralSDE, sde_solve
from style import COLORS, save, setup

jax.config.update("jax_enable_x64", True)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def build(n_steps, method, solver, model, n_paths, key, tol):
    """All sample paths are solved as one SDE with a (n_paths * d)-dim state, so
    the adjoint carries a single parameter-gradient accumulator."""
    import lineax as lx

    params, static = eqx.partition(model, eqx.is_inexact_array)
    d = model.state_dim

    def drift(t, y, p):
        m = eqx.combine(p, static)
        return jax.vmap(lambda z: m.drift(t, z))(y.reshape(n_paths, d)).reshape(-1)

    def diffusion(t, y, p):
        m = eqx.combine(p, static)
        diag = jax.vmap(lambda z: jnp.diag(m.diffusion(t, z)))(y.reshape(n_paths, d)).reshape(-1)
        return lx.DiagonalLinearOperator(diag)

    y0 = jnp.full(n_paths * d, 0.5)

    def loss(p):
        y = sde_solve(drift, diffusion, y0, p, key, n_steps=n_steps, solver=solver, method=method, bm_tol=tol)
        return jnp.sum(y**2) / n_paths

    return jax.jit(jax.grad(loss)), params


def rel_err(g, ref):
    num = jnp.sqrt(sum(jnp.sum((a - b) ** 2) for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(ref))))
    den = jnp.sqrt(sum(jnp.sum(b**2) for b in jax.tree.leaves(ref)))
    return float(num / den)


def plot(out, steps):
    from matplotlib.ticker import NullLocator

    setup()
    methods = ["backprop", "checkpoint", "adjoint"]
    labels = {"backprop": "Backprop through every step", "checkpoint": "Recursive checkpointing", "adjoint": "Stochastic adjoint"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in methods:
        axes[0].plot(out[m]["steps"], out[m]["temp_mb"], "-o", color=COLORS[m], label=labels[m])
        axes[1].plot(out[m]["steps"], np.array(out[m]["seconds"]), "-o", color=COLORS[m], label=labels[m])
    axes[2].plot(out["adjoint"]["steps"], out["adjoint"]["rel_err"], "-o", color=COLORS["adjoint"], label="Adjoint, reversible Heun")
    axes[2].plot(out["adjoint_heun"]["steps"], out["adjoint_heun"]["rel_err"], "--o", color=COLORS["muted"], label="Adjoint, Heun")
    axes[0].set(title="Memory for the gradient", ylabel="XLA temp buffers (MB)", yscale="log")
    axes[1].set(title="Time per gradient", ylabel="seconds", yscale="log")
    axes[2].set(title="Adjoint gradient error vs. backprop", ylabel="relative L2 error", yscale="log")
    for ax in axes:
        ax.set(xlabel="solver steps", xscale="log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.07))
    axes[2].legend(fontsize=9)
    save(fig, ROOT / "assets" / "adjoint_tradeoffs.png")


def main(quick=False):
    setup()
    steps = [16, 32, 64, 128] if quick else [16, 32, 64, 128, 256, 512]
    n_paths = 16 if quick else 32
    model = NeuralSDE(16, width=64, depth=2, key=jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    tol = 1.0 / (8 * max(steps))  # same Brownian paths at every resolution
    methods = ["backprop", "checkpoint", "adjoint"]
    out = {m: {"steps": [], "temp_mb": [], "seconds": [], "rel_err": []} for m in methods}
    out["adjoint_heun"] = {"steps": [], "rel_err": []}

    for n in steps:
        grads = {}
        for m in methods:
            f, params = build(n, m, "reversible_heun", model, n_paths, key, tol)
            compiled = f.lower(params).compile()
            mem = compiled.memory_analysis().temp_size_in_bytes / 2**20
            g = compiled(params)
            jax.block_until_ready(g)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                jax.block_until_ready(compiled(params))
                ts.append(time.perf_counter() - t0)
            grads[m] = g
            out[m]["steps"].append(n)
            out[m]["temp_mb"].append(mem)
            out[m]["seconds"].append(float(np.median(ts)))
        for m in methods:
            out[m]["rel_err"].append(rel_err(grads[m], grads["backprop"]))
        # adjoint error with a non-reversible (Stratonovich) Heun solver, for comparison
        fb, params = build(n, "backprop", "heun", model, n_paths, key, tol)
        fa, _ = build(n, "adjoint", "heun", model, n_paths, key, tol)
        out["adjoint_heun"]["steps"].append(n)
        out["adjoint_heun"]["rel_err"].append(rel_err(fa(params), fb(params)))
        print(f"steps={n:4d}  " + "  ".join(f"{m}: {out[m]['temp_mb'][-1]:7.2f} MB {out[m]['seconds'][-1]*1e3:7.1f} ms" for m in methods)
              + f"  adjoint err={out['adjoint']['rel_err'][-1]:.2e}", flush=True)

    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "adjoint_tradeoffs.json", "w"), indent=2)

    plot(out, steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from results/adjoint_tradeoffs.json")
    a = ap.parse_args()
    if a.plot_only:
        out = json.load(open(ROOT / "results" / "adjoint_tradeoffs.json"))
        plot(out, out["backprop"]["steps"])
    else:
        main(a.quick)
