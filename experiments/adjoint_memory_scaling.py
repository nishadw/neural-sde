"""What does the stochastic adjoint's (constant-in-steps) memory scale with?

Compiles the adjoint gradient for a few model sizes and numbers of noise
channels (state dims x sample paths) and reports XLA's temporary-buffer size.
Compile-only, so it runs in a minute or two.

Usage:  python experiments/adjoint_memory_scaling.py
"""

import json
import pathlib

import equinox as eqx
import jax
from adjoint_tradeoffs import build

from nsde import NeuralSDE

jax.config.update("jax_enable_x64", True)
ROOT = pathlib.Path(__file__).resolve().parents[1]

rows = []
for d, width, n_paths in [(16, 64, 16), (16, 64, 32), (16, 64, 64), (16, 32, 32), (16, 128, 32)]:
    model = NeuralSDE(d, width=width, depth=2, key=jax.random.PRNGKey(0))
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))
    row = {"noise_channels": d * n_paths, "params": int(n_params)}
    for method in ("adjoint", "backprop"):
        f, p = build(64, method, "reversible_heun", model, n_paths, jax.random.PRNGKey(1), 1 / 4096)
        row[f"{method}_temp_mb"] = f.lower(p).compile().memory_analysis().temp_size_in_bytes / 2**20
    rows.append(row)
    print(row, flush=True)

(ROOT / "results").mkdir(exist_ok=True)
json.dump(rows, open(ROOT / "results" / "adjoint_memory_scaling.json", "w"), indent=2)
