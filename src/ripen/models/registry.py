"""Build any model from a config dict: ``build_model(cfg, info, key)``."""

from __future__ import annotations

import jax

from ..data.collate import DatasetInfo
from .baselines import DirectRegressor, DiscreteSurvival, StageLinearRule
from .fpt_filter import FirstPassageFilter
from .latent import LatentDynamics
from .sequence import SequenceSurvival


def _seq(encoder):
    def make(cfg, info, key):
        return SequenceSurvival(
            info.feat_dim,
            len(info.groups),
            cfg["horizon"],
            encoder=encoder,
            hidden=cfg.get("hidden", 64),
            visit_dim=cfg.get("visit_dim", 32),
            group_dim=cfg.get("group_dim", 4),
            use_images=cfg.get("use_images", True),
            use_stages=cfg.get("use_stages", False),
            key=key,
        )

    return make


def _latent(kind):
    def make(cfg, info, key):
        kw = {k: v for k, v in cfg.items() if k not in ("name", "horizon")}
        return LatentDynamics(info.feat_dim, len(info.groups), cfg["horizon"], info.t_max, key=key, kind=kind, **kw)

    return make


MODEL_REGISTRY = {
    "stage_linear": lambda cfg, info, key: StageLinearRule(
        info.feat_dim, len(info.groups), cfg["horizon"], cfg.get("width", 64), cfg.get("depth", 2), key=key
    ),
    "direct_reg": lambda cfg, info, key: DirectRegressor(
        info.feat_dim,
        len(info.groups),
        cfg["horizon"],
        cfg.get("width", 64),
        cfg.get("depth", 2),
        cfg.get("group_dim", 4),
        cfg.get("huber_delta", 2.0),
        key=key,
    ),
    "discrete_surv": lambda cfg, info, key: DiscreteSurvival(
        info.feat_dim,
        len(info.groups),
        cfg["horizon"],
        cfg.get("width", 64),
        cfg.get("depth", 2),
        cfg.get("group_dim", 4),
        key=key,
    ),
    "gru_gap": _seq("gru"),
    "gru_d": _seq("gru_d"),
    "ode_rnn": _seq("ode_rnn"),
    "fpt_filter": lambda cfg, info, key: FirstPassageFilter(
        info.feat_dim,
        len(info.groups),
        cfg["horizon"],
        key=key,
        **{k: v for k, v in cfg.items() if k not in ("name", "horizon", "seed")},
    ),
    "latent_ode": _latent("ode"),
    "latent_sde": _latent("sde"),
}


def build_model(cfg: dict, info: DatasetInfo, key=None):
    """``cfg`` must have ``name`` and ``horizon``; other keys are model-specific (see the configs/ directory)."""
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    key = key if key is not None else jax.random.PRNGKey(cfg.get("seed", 0))
    return MODEL_REGISTRY[name](dict(cfg), info, key)
