"""Glue used by the scripts: load items for a data config, make or load splits, evaluate and write a run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .data import (
    DatasetInfo,
    Item,
    SyntheticConfig,
    dataset_info,
    leave_one_group_out,
    load_splits,
    make_splits,
    save_splits,
    select,
    simulate,
    subsample_visits,
)
from .data.avocado import load_avocado
from .data.cmapss import load_cmapss
from .data.features import load_feature_cache
from .data.pbc2 import load_pbc2
from .data.stevens import load_manifest
from .eval import predict_all, summarize_by_group
from .eval.plots import predicted_vs_true, reliability_diagram
from .models import build_model

REGISTRY_EVENT_DATASETS = ("pbc2", "cmapss")  # event time known from a registry, not from a labelled visit


def subsample_visits_known_event(item: Item, every: int, offset: int = 0) -> Item:
    """Like :func:`subsample_visits`, for items whose event time does not come from a visit's stage label.

    Keeps the first visit and every ``every``-th one. The event time ``spoil_hi`` is unchanged (it is
    observed exactly from the registry, not at a visit); ``spoil_lo`` moves back to the last kept visit
    before it. Right-censored items keep their censoring time.
    """
    if every < 1:
        raise ValueError("every must be >= 1")
    kept = [v for i, v in enumerate(item.visits) if i == 0 or (i - offset) % every == 0]
    lo = item.spoil_lo
    if not item.right_censored:
        before = [v.t for v in kept if v.t < item.spoil_hi]
        lo = max(before) if before else item.spoil_hi
    return replace(item, visits=kept, spoil_lo=lo)


def load_items(data_cfg: DictConfig) -> list[Item]:
    """Items for ``data.name in {synthetic, avocado, stevens, pbc2, cmapss}``, after optional visit subsampling."""
    name = data_cfg.name
    if name == "synthetic":
        keys = {f.name for f in SyntheticConfig.__dataclass_fields__.values()}
        kw = {k: v for k, v in OmegaConf.to_container(data_cfg, resolve=True).items() if k in keys}
        items = simulate(SyntheticConfig(**kw))
    elif name == "avocado":
        feats = load_feature_cache(data_cfg.features_file) if data_cfg.get("features_file") else None
        items = load_avocado(data_cfg.root, data_cfg.get("labels_file"), data_cfg.get("images_dir"), feats)
    elif name == "stevens":
        feats = load_feature_cache(data_cfg.features_file) if data_cfg.get("features_file") else None
        items = load_manifest(data_cfg.manifest, feats)
    elif name == "a2sl":
        from .data.a2sl import load_a2sl_avocado, load_a2sl_pilot, load_scan_cache

        feats = load_feature_cache(data_cfg.features_file) if data_cfg.get("features_file") else None
        scans = load_scan_cache(data_cfg.scans_file) if data_cfg.get("scans_file") else None
        items = load_a2sl_avocado(
            data_cfg.get("manifests", "data/manifests"),
            features=feats,
            scans=scans,
            photo_dim=int(data_cfg.get("photo_dim", 512)),
            labels=data_cfg.get("labels", "jmw2026"),
        )
        if data_cfg.get("include_pilot", False):
            items += load_a2sl_pilot(data_cfg.get("manifests", "data/manifests"), features=feats)
    elif name == "pbc2":
        items = load_pbc2(
            data_cfg.path,
            transplant_as_event=bool(data_cfg.get("transplant_as_event", False)),
            log_skewed=bool(data_cfg.get("log_skewed", True)),
        )
    elif name == "cmapss":
        subsets = data_cfg.get("subsets", ["FD001"])
        items = load_cmapss(
            data_cfg.root,
            subsets=(subsets,) if isinstance(subsets, str) else tuple(str(s) for s in subsets),
            inspect_every=int(data_cfg.get("inspect_every", 10)),
            jitter=int(data_cfg.get("jitter", 2)),
            seed=int(data_cfg.get("seed", 0)),
            drop_tail=float(data_cfg.get("drop_tail", 0.0)),
            drop_constant=bool(data_cfg.get("drop_constant", True)),
        )
    else:
        raise KeyError(f"unknown dataset {name!r}")
    every = int(data_cfg.get("subsample_every", 1) or 1)
    if every > 1:
        offset = int(data_cfg.get("subsample_offset", 0) or 0)
        sub = subsample_visits_known_event if name in REGISTRY_EVENT_DATASETS else subsample_visits
        items = [sub(it, every, offset) for it in items]
    return items


def splits_for(cfg: DictConfig, items: list[Item]) -> dict[str, list[str]]:
    """Load the split file if it exists, else make one and save it (so every model sees the same items)."""
    mode = cfg.splits.mode
    suffix = cfg.splits.held_out if mode == "logo" else f"seed{cfg.splits.seed}"
    path = Path(cfg.splits.file or f"results/splits/{cfg.data.name}_{mode}_{suffix}.json")
    if path.exists():
        return load_splits(path)
    if mode == "random":
        sp = make_splits(items, seed=int(cfg.splits.seed))
    elif mode == "logo":
        sp = leave_one_group_out(items, cfg.splits.held_out, seed=int(cfg.splits.seed))
    else:
        raise ValueError(mode)
    save_splits(sp, path, extra={"mode": mode, "seed": int(cfg.splits.seed), "data": cfg.data.name})
    return sp


def model_cfg_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg.model, resolve=True)


def evaluate_and_write(model, items, info: DatasetInfo, cfg: DictConfig, out: Path, key, tag: str = "test") -> dict:
    """Predict on ``items``, compute per-group / overall / fruit-weighted metrics, write JSON and figures."""
    preds = predict_all(
        model, items, info, key, n_obs_list=_n_obs_list(cfg.eval.n_obs_list), batch_size=cfg.eval.batch_size
    )
    out.mkdir(parents=True, exist_ok=True)
    if preds.F.shape[0] == 0:
        (out / f"metrics_{tag}.json").write_text(json.dumps({"overall": {"n": 0}}, indent=2))
        return {
            "overall": {
                "n": 0,
                "mae": float("nan"),
                "crps": float("nan"),
                "coverage80": float("nan"),
                "width80": float("nan"),
                "cindex": float("nan"),
                "ece": float("nan"),
                "n_uncensored": 0,
            }
        }
    metrics = summarize_by_group(preds.F, preds.lo, preds.hi, preds.group, info.groups, preds.item_ids)
    (out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    reliability_diagram(preds.F, preds.lo, preds.hi, out / f"reliability_{tag}.png", title=f"{cfg.model.name} / {tag}")
    predicted_vs_true(preds.F, preds.hi, out / f"pred_vs_true_{tag}.png", title=f"{cfg.model.name} / {tag}")
    np.savez_compressed(
        out / f"predictions_{tag}.npz",
        F=preds.F,
        lo=preds.lo,
        hi=preds.hi,
        group=preds.group,
        item_ids=preds.item_ids,
        n_obs=preds.n_obs,
        t_now=preds.t_now,
        stage_now=preds.stage_now,
    )
    return metrics


def n_obs_ablation(model, items, info, cfg: DictConfig, out: Path, key) -> dict:
    """Metrics as a function of how many photos are visible (direction 4)."""
    res = {}
    for k in cfg.eval.n_obs_ablation:
        key, sub = jax.random.split(key)
        preds = predict_all(model, items, info, sub, n_obs_list=[int(k)], batch_size=cfg.eval.batch_size)
        if preds.F.shape[0]:
            res[str(k)] = summarize_by_group(preds.F, preds.lo, preds.hi, preds.group, info.groups, preds.item_ids)[
                "overall"
            ]
    (out / "metrics_by_n_obs.json").write_text(json.dumps(res, indent=2))
    return res


def _n_obs_list(x):
    return "all_unspoiled" if isinstance(x, str) else [int(v) for v in x]


def build_from_run(run_dir: str | Path):
    """Rebuild a trained model from a run directory (config.yaml, info.json, model.eqx)."""
    from .train import load_model

    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    info = DatasetInfo.from_dict(json.loads((run_dir / "info.json").read_text()))
    template = build_model(model_cfg_dict(cfg), info, jax.random.PRNGKey(int(cfg.seed)))
    return load_model(template, run_dir / "model.eqx"), cfg, info


__all__ = [
    "build_from_run",
    "dataset_info",
    "evaluate_and_write",
    "load_items",
    "model_cfg_dict",
    "n_obs_ablation",
    "select",
    "splits_for",
    "subsample_visits_known_event",
]
