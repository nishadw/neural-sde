"""Metrics on a predicted CDF of days-left, identical for every model.

Inputs are NumPy: ``F[n, h]`` = P(days_left <= h) on the integer grid
``h = 0..H``; ``lo[n]``, ``hi[n]`` the spoilage interval in days from now
(``hi = inf`` if right-censored). Point predictions arrive as step CDFs.

* MAE: |median − hi| on uncensored instances (the paper's "Days Left" is the day the spoiled stage was first seen).
* CRPS: ``sum_h (F(h) − 1[h >= hi])^2`` (unit-day Riemann sum) on uncensored instances; it equals MAE for a
  step CDF, so the baselines are comparable. ``crps_censored`` is the right-censored variant of
  Avati et al. (2018): ``sum_{h < lo} F(h)^2`` — a censored instance is only penalised for mass before ``lo``.
* 80 % interval: quantiles 0.1 and 0.9 by inverting ``F``; coverage on uncensored instances, width on all.
* C-index: Harrell's, with ``-median`` as the risk score; events at ``hi``, censoring at ``lo``.
* Reliability: for horizons ``h``, predicted ``F(h)`` vs observed ``1[hi <= h]`` on instances where the outcome
  by ``h`` is known (uncensored, or censored with ``lo >= h``). Summarised by the expected calibration error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Metrics:
    n: int
    n_uncensored: int
    mae: float
    crps: float
    crps_censored: float
    coverage80: float
    width80: float
    cindex: float
    ece: float

    def to_dict(self) -> dict:
        return asdict(self)


def quantile_from_cdf(F: np.ndarray, q: float) -> np.ndarray:
    """Smallest grid day ``h`` with ``F[h] >= q`` (the distribution lives on integer days; no interpolation, so a
    point prediction's median is the point itself). ``F`` is ``[N, H+1]``; if ``q`` is never reached, returns ``H``."""
    H1 = F.shape[1]
    reached = F >= q - 1e-9
    return np.where(reached.any(1), reached.argmax(1), H1 - 1).astype(float)


def harrell_cindex(score: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Fraction of comparable pairs ordered correctly; higher ``score`` means earlier event. Ties count 1/2."""
    score, time, event = np.asarray(score, float), np.asarray(time, float), np.asarray(event, bool)
    conc, comp = 0.0, 0
    for i in np.nonzero(event)[0]:
        comparable = (time > time[i]) | ((time == time[i]) & ~event)
        comparable[i] = False
        comp += int(comparable.sum())
        conc += float(np.sum(score[i] > score[comparable]) + 0.5 * np.sum(score[i] == score[comparable]))
    return conc / comp if comp else float("nan")


def reliability_curve(F: np.ndarray, lo: np.ndarray, hi: np.ndarray, horizons=(1, 2, 3, 5, 7, 10), n_bins: int = 10):
    """Pooled over horizons: returns ``(bin_pred, bin_obs, bin_count, ece)``."""
    preds, obs = [], []
    H = F.shape[1] - 1
    for h in horizons:
        if h > H:
            continue
        known = ~np.isinf(hi) | (lo >= h)
        y = (hi <= h).astype(float)
        preds.append(F[known, h])
        obs.append(y[known])
    if not preds:
        return np.array([]), np.array([]), np.array([]), float("nan")
    p, y = np.concatenate(preds), np.concatenate(obs)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bp = np.array([p[b == k].mean() if np.any(b == k) else np.nan for k in range(n_bins)])
    bo = np.array([y[b == k].mean() if np.any(b == k) else np.nan for k in range(n_bins)])
    bc = np.array([np.sum(b == k) for k in range(n_bins)])
    ece = float(np.nansum(np.abs(bp - bo) * bc) / max(bc.sum(), 1))
    return bp, bo, bc, ece


def compute_metrics(F: np.ndarray, lo: np.ndarray, hi: np.ndarray, weights: np.ndarray | None = None) -> Metrics:
    F = np.asarray(F, float)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    N, H1 = F.shape
    h = np.arange(H1, dtype=float)
    unc = ~np.isinf(hi)
    w = np.ones(N) if weights is None else np.asarray(weights, float)
    med = quantile_from_cdf(F, 0.5)
    q10, q90 = quantile_from_cdf(F, 0.1), quantile_from_cdf(F, 0.9)

    def wmean(x, m):
        return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if np.any(m) else float("nan")

    mae = wmean(np.abs(med - hi), unc)
    ind = (h[None, :] >= hi[:, None]).astype(float)
    crps_u = np.sum((F - ind) ** 2, axis=1)
    crps_c = np.sum(np.where(h[None, :] < lo[:, None], F**2, 0.0), axis=1)
    crps = wmean(crps_u, unc)
    crps_cens = wmean(np.where(unc, crps_u, crps_c), np.ones(N, bool))
    cover = wmean(((q10 <= hi) & (hi <= q90)).astype(float), unc)
    width = wmean(q90 - q10, np.ones(N, bool))
    time = np.where(unc, hi, lo)
    cidx = harrell_cindex(-med, time, unc)
    _, _, _, ece = reliability_curve(F, lo, hi)
    return Metrics(int(N), int(unc.sum()), mae, crps, crps_cens, cover, width, cidx, ece)


def summarize_by_group(F, lo, hi, groups: np.ndarray, group_names, item_ids: np.ndarray | None = None) -> dict:
    """Overall and per-group metrics; with ``item_ids``, also ``fruit_weighted`` (each item weighs 1 in total)."""
    out = {"overall": compute_metrics(F, lo, hi).to_dict()}
    for g, name in enumerate(group_names):
        m = groups == g
        if m.any():
            out[name] = compute_metrics(F[m], lo[m], hi[m]).to_dict()
    if item_ids is not None:
        _, inv, counts = np.unique(item_ids, return_inverse=True, return_counts=True)
        out["fruit_weighted"] = compute_metrics(F, lo, hi, weights=1.0 / counts[inv]).to_dict()
    return out
