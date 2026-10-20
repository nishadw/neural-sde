"""Train one model on one dataset and evaluate it. Everything is driven by configs/.

python scripts/train.py model=latent_sde data=synthetic
python scripts/train.py model=gru_d data=avocado train=default
python scripts/train.py +experiment=logo splits.held_out=T20 model=latent_sde data=avocado
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import jax
from omegaconf import DictConfig, OmegaConf

from ripen.models import build_model
from ripen.pipeline import (
    dataset_info,
    evaluate_and_write,
    load_items,
    model_cfg_dict,
    n_obs_ablation,
    select,
    splits_for,
)
from ripen.train import TrainConfig, train


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    items = load_items(cfg.data)
    sp = splits_for(cfg, items)
    tr, va, te = select(items, sp["train"]), select(items, sp["val"]), select(items, sp["test"])
    info = dataset_info(items)
    (out / "info.json").write_text(json.dumps(info.to_dict(), indent=2))
    print(f"{cfg.data.name}: {len(tr)}/{len(va)}/{len(te)} items, groups {info.groups}, feat_dim {info.feat_dim}")
    key = jax.random.PRNGKey(int(cfg.seed))
    k_model, k_train, k_eval = jax.random.split(key, 3)
    model = build_model(model_cfg_dict(cfg), info, k_model)
    tcfg = TrainConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    model, _ = train(model, tr, va, info, tcfg, k_train, out_dir=out)
    k1, k2, k3 = jax.random.split(k_eval, 3)
    m_test = evaluate_and_write(model, te, info, cfg, out, k1, tag="test")
    evaluate_and_write(model, va, info, cfg, out, k2, tag="val")
    n_obs_ablation(model, te, info, cfg, out, k3)
    o = m_test["overall"]
    print(
        f"test: MAE {o['mae']:.3f}  CRPS {o['crps']:.3f}  cov80 {o['coverage80']:.2f}  "
        f"width80 {o['width80']:.2f}  C {o['cindex']:.3f}  ECE {o['ece']:.3f}  "
        f"(n={o['n']}, uncensored {o['n_uncensored']})"
    )


if __name__ == "__main__":
    main()
