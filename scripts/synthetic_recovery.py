"""The fake-fruit test at full scale: fit a label-only latent SDE to simulated ripening and compare the
recovered drift, diffusion and first-passage distribution with the truth. Writes
results/synthetic/recovery.json and results/synthetic/recovery.png.

    python scripts/synthetic_recovery.py                 # ~5-10 min on CPU
    python scripts/synthetic_recovery.py --quick         # ~1 min, looser
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ripen.data import SyntheticConfig, collate, dataset_info, make_splits, select, simulate
from ripen.data.synthetic import analytic_spoilage_cdf
from ripen.eval.plots import recovery_panel
from ripen.models import LatentDynamics
from ripen.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scfg = SyntheticConfig(n_items=150 if args.quick else 600, t_max=30.0, sigma=0.25, seed=args.seed)
    items = simulate(scfg)
    sp = make_splits(items, seed=args.seed)
    tr, va, te = select(items, sp["train"]), select(items, sp["val"]), select(items, sp["test"])
    info = dataset_info(items)
    key = jax.random.PRNGKey(args.seed)
    model = LatentDynamics(
        info.feat_dim,
        len(info.groups),
        horizon=30,
        t_max=info.t_max,
        key=key,
        latent_dim=1,
        use_images=False,
        use_stages=True,
        drift_kind="linear",
        sigma_kind="const",
        clock="group",
        clock_kwargs={"init_rate": 0.4},
        fixed_thresholds=True,
        n_sub=2,
        n_train_samples=4,
        n_samples=256,
    )
    tcfg = TrainConfig(
        epochs=8 if args.quick else 40, batch_size=32, lr=1e-2, patience=8, n_obs_mode="random", seed=args.seed
    )
    model, _ = train(model, tr, va, info, tcfg, jax.random.PRNGKey(args.seed + 1), out_dir=out / "run")

    rates_fit = {g: float(r) for g, r in zip(info.groups, np.asarray(model.clock.rates()), strict=True)}
    sigma_fit = float(np.asarray(model.sigma(jnp.zeros(1)))[0])
    # first-passage CDF from a fresh item in T20 (stage-1 start) vs the inverse-Gaussian truth
    fresh = [it.prefix(1) for it in te if it.storage == "T20"][:32]
    batch = collate(fresh, info, n_obs=1)
    F = np.asarray(model.predict_cdf(batch, jax.random.PRNGKey(7))).mean(0)
    t = np.arange(F.shape[0], dtype=float)
    z0s = np.array([it.meta["z0"] for it in fresh])
    cdf_true = np.mean([analytic_spoilage_cdf(t, z0, scfg.rates["T20"], scfg.sigma) for z0 in z0s], axis=0)
    rep = {
        "rates_true": scfg.rates,
        "rates_fit": rates_fit,
        "sigma_true": scfg.sigma,
        "sigma_fit": sigma_fit,
        "fpt_max_abs_cdf_err_T20": float(np.max(np.abs(F - cdf_true))),
        "n_items": scfg.n_items,
        "quick": args.quick,
    }
    (out / "recovery.json").write_text(json.dumps(rep, indent=2))
    z = np.linspace(1, 5, 50)
    recovery_panel(
        z,
        np.full_like(z, scfg.rates["T20"]),
        np.asarray(model.drift_on_grid(jnp.asarray(z, jnp.float32), rates_fit["T20"])),
        np.full_like(z, scfg.sigma),
        np.asarray(model.sigma_on_grid(jnp.asarray(z, jnp.float32))),
        t,
        cdf_true,
        F,
        out / "recovery.png",
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
