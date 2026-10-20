"""Fake fruit: ripening simulated from a known SDE, quantised to stages, observed on a visit schedule.

The latent ripeness ``Z`` lives in stage units. Starting near stage 1 it
follows ``dZ = v_g dt + sigma dW`` for storage group ``g``; the item is
spoiled at the first time ``Z`` reaches ``SPOILED_STAGE``. Defaults are the
rates implied by Table 4 of Xavier et al. (2024): 0.228 / 0.473 / 0.518
stages per day for T10 / T20 / Tamb. For constant drift the first-passage
time is inverse-Gaussian, which the tests use as ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .schema import N_STAGES, SPOILED_STAGE, Item, Visit, spoil_interval_from_visits

DEFAULT_RATES = {"T10": 0.228, "T20": 0.473, "Tamb": 0.518}
DEFAULT_TEMPS = {"T10": 10.0, "T20": 20.0, "Tamb": 18.7}


@dataclass
class SyntheticConfig:
    n_items: int = 300
    rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATES))
    temps: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TEMPS))
    sigma: float = 0.25  # stage units per sqrt(day)
    z0_mean: float = 1.2
    z0_sd: float = 0.15
    visit_interval: int = 1  # days between scheduled visits
    p_missing: float = 0.0  # probability a scheduled visit is skipped (never the first)
    t_max: float = 40.0  # study ends; unspoiled items are right-censored here
    dt: float = 0.01  # simulation step for the latent path
    feat_dim: int = 8  # synthetic "image features": a noisy linear image of the stage
    feat_noise: float = 0.3
    label_noise: float = 0.0  # probability a stage label is off by one
    produce: str = "synthetic"
    seed: int = 0


def _stage_of(z: float) -> int:
    return int(min(max(math.floor(z), 1), N_STAGES))


def simulate(cfg: SyntheticConfig, rng: np.random.Generator | None = None) -> list[Item]:
    """Simulate ``cfg.n_items`` items spread evenly over the storage groups."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    groups = list(cfg.rates)
    W = rng.normal(size=(N_STAGES, cfg.feat_dim)).astype(np.float32)  # fixed stage -> feature map
    n_steps = int(round(cfg.t_max / cfg.dt))
    items: list[Item] = []
    for i in range(cfg.n_items):
        g = groups[i % len(groups)]
        v, sd = cfg.rates[g], cfg.sigma
        z0 = float(np.clip(rng.normal(cfg.z0_mean, cfg.z0_sd), 1.0, SPOILED_STAGE - 0.5))
        noise = rng.normal(size=n_steps) * sd * math.sqrt(cfg.dt)
        path = z0 + np.cumsum(v * cfg.dt + noise)
        path = np.concatenate([[z0], path])
        hit = np.nonzero(path >= SPOILED_STAGE)[0]
        tau = float(hit[0] * cfg.dt) if hit.size else math.inf
        # visit schedule
        sched = np.arange(0.0, cfg.t_max + 1e-9, cfg.visit_interval)
        keep = rng.random(sched.size) >= cfg.p_missing
        keep[0] = True
        visits: list[Visit] = []
        for t in sched[keep]:
            k = min(int(round(t / cfg.dt)), n_steps)
            stage = SPOILED_STAGE if t >= tau else _stage_of(path[k])
            if cfg.label_noise > 0 and stage < SPOILED_STAGE and rng.random() < cfg.label_noise:
                stage = int(np.clip(stage + rng.choice([-1, 1]), 1, SPOILED_STAGE - 1))
            feat = (W[stage - 1] + rng.normal(size=cfg.feat_dim) * cfg.feat_noise).astype(np.float32)[None]
            visits.append(Visit(t=float(t), stage=stage, features=feat))
            if stage >= SPOILED_STAGE:
                break
        lo, hi = spoil_interval_from_visits(visits)
        items.append(
            Item(
                item_id=f"syn{i:04d}",
                produce=cfg.produce,
                storage=g,
                temp_c=cfg.temps[g],
                visits=visits,
                spoil_lo=lo,
                spoil_hi=hi,
                meta={"tau": tau, "rate": v, "sigma": sd, "z0": z0},
            )
        )
    return items


def analytic_spoilage_cdf(t: np.ndarray, z0: float, rate: float, sigma: float) -> np.ndarray:
    """P(tau <= t) for the simulator's constant-drift latent, from a fresh start at ``z0``."""
    from ..models.fpt import inverse_gaussian_cdf  # local import: models depend on data, not the reverse

    return inverse_gaussian_cdf(np.asarray(t, float), SPOILED_STAGE - z0, rate, sigma)
