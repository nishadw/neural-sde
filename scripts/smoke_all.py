"""Every model, one batch, forward + gradient + predict, on synthetic data (and on avocado if the data is present).

python scripts/smoke_all.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from ripen.data import SyntheticConfig, collate, dataset_info, simulate
from ripen.data.avocado import load_avocado
from ripen.data.features import load_feature_cache
from ripen.models import MODEL_REGISTRY, build_model

ROOT = Path(__file__).resolve().parents[1]


def smoke(items, label: str, horizon: int = 20) -> bool:
    info = dataset_info(items)
    rng = np.random.default_rng(0)
    batch = collate(items[:8], info, n_obs="random", rng=rng)
    ok = True
    for name in MODEL_REGISTRY:
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "model" / f"{name}.yaml"), resolve=False)
        cfg["horizon"] = horizon
        if name.startswith("latent"):
            cfg["n_samples"], cfg["n_train_samples"] = 8, 2
        t0 = time.time()
        try:
            model = build_model(cfg, info, jax.random.PRNGKey(0))
            loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(batch, jax.random.PRNGKey(1)))(model)
            gnorm = float(
                jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))))
            )
            F = model.predict_cdf(batch, jax.random.PRNGKey(2))
            assert F.shape == (batch.batch_size, horizon + 1), F.shape
            assert bool(jnp.all((F >= -1e-6) & (F <= 1 + 1e-6))) and bool(jnp.all(jnp.diff(F, axis=1) >= -1e-6))
            assert np.isfinite(float(loss)) and np.isfinite(gnorm)
            print(f"  {label:9s} {name:14s} loss {float(loss):9.4f}  |grad| {gnorm:9.3f}  {time.time() - t0:5.1f}s  ok")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {label:9s} {name:14s} FAILED: {type(e).__name__}: {e}")
    return ok


def main() -> None:
    print("synthetic")
    ok = smoke(simulate(SyntheticConfig(n_items=24, t_max=20.0)), "synthetic")
    avo = ROOT / "data" / "avocado"
    if avo.exists() and any(avo.rglob("*.xlsx")):
        feats_path = ROOT / "data" / "features" / "avocado_resnet18.npz"
        feats = load_feature_cache(feats_path) if feats_path.exists() else None
        print(f"avocado (features: {'cached' if feats else 'none'})")
        ok &= smoke(load_avocado(avo, features=feats), "avocado", horizon=45)
    else:
        print("avocado: data/avocado not present, skipped")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
