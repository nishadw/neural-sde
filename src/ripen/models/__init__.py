"""Model zoo. Every model is an :class:`eqx.Module` with ``loss(batch, key)`` and ``predict_cdf(batch, key)``."""

from .base import ShelfLifeModel
from .baselines import DirectRegressor, DiscreteSurvival, StageLinearRule
from .clock import ArrheniusClock, FixedClock, GroupClock, make_clock
from .latent import LatentDynamics
from .registry import MODEL_REGISTRY, build_model
from .sequence import SequenceSurvival

__all__ = [
    "ArrheniusClock",
    "DirectRegressor",
    "DiscreteSurvival",
    "FixedClock",
    "GroupClock",
    "LatentDynamics",
    "MODEL_REGISTRY",
    "SequenceSurvival",
    "ShelfLifeModel",
    "StageLinearRule",
    "build_model",
    "make_clock",
]
