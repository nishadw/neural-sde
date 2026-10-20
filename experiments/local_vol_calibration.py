"""Calibrate a neural local-volatility SDE to a Heston implied-volatility smile.

Market:  European call prices from a Heston model (semi-analytic), three
         maturities, strikes 80%-120% of spot. These play the role of quotes.
Model:   d log S = (r - sigma^2 / 2) dt + sigma_theta(t, log S) dW, with
         sigma_theta a small MLP. Prices are Monte Carlo averages over paths.
Fit:     minimise vega-weighted squared price error by differentiating the
         Monte Carlo estimator through the SDE solver (pathwise gradients),
         in the spirit of Gierjatowicz et al., "Robust pricing and hedging via
         neural SDEs" (2020), at toy scale.

Half of the strikes are held out from training and used to check the fit
between quoted strikes.

Usage:  python experiments/local_vol_calibration.py [--quick]
"""

import argparse
import json
import pathlib
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib.pyplot as plt
import numpy as np
import optax
from style import COLORS, save, setup

from nsde import LocalVolatility, bs_vega, heston_call, implied_vol, sde_solve

ROOT = pathlib.Path(__file__).resolve().parents[1]

HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
S0, R = 1.0, 0.0
MATURITIES = np.array([0.25, 0.5, 1.0])
ALL_STRIKES = np.round(np.arange(0.80, 1.2001, 0.025), 4)
TRAIN_MASK = np.arange(len(ALL_STRIKES)) % 2 == 0  # 0.80, 0.85, ..., 1.20 train; in-between held out
STEPS_PER_YEAR = 50


def market():
    prices = np.array([[heston_call(S0, K, T, R, **HESTON) for K in ALL_STRIKES] for T in MATURITIES])
    ivs = np.array(
        implied_vol(jnp.array(prices), S0, jnp.array(ALL_STRIKES)[None, :], jnp.array(MATURITIES)[:, None], R)
    )
    return prices, ivs


def simulate(model, key, n_paths):
    """Terminal log-prices at each maturity, shape (n_maturities, n_paths)."""
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def sigma(t, x, p):
        m = eqx.combine(p, static)
        return jax.vmap(lambda xi: m(t, xi))(x)

    drift = lambda t, x, p: R - 0.5 * sigma(t, x, p) ** 2
    diffusion = lambda t, x, p: lx.DiagonalLinearOperator(sigma(t, x, p))
    n_steps = int(MATURITIES[-1] * STEPS_PER_YEAR)
    return sde_solve(
        drift,
        diffusion,
        jnp.zeros(n_paths),
        params,
        key,
        t1=float(MATURITIES[-1]),
        n_steps=n_steps,
        solver="euler",
        method="checkpoint",
        ts=jnp.array(MATURITIES),
        bm_tol=0.5 / STEPS_PER_YEAR,
    )


def mc_prices(model, key, n_paths):
    x = simulate(model, key, n_paths)  # (T, paths)
    s = jnp.exp(x)
    s = s / jnp.mean(s, axis=1, keepdims=True) * S0  # moment matching: E[S_T] = S0 (r = 0)
    payoff = jnp.maximum(s[:, :, None] - jnp.array(ALL_STRIKES)[None, None, :], 0.0)
    return jnp.exp(-R * jnp.array(MATURITIES))[:, None] * jnp.mean(payoff, axis=1)  # (T, K)


def main(quick=False):
    setup()
    t0 = time.time()
    target_p, target_iv = market()
    print("market smile (vol pts):\n", np.round(100 * target_iv, 2))
    vega = np.array(bs_vega(S0, ALL_STRIKES[None, :], MATURITIES[:, None], R, target_iv))
    w = jnp.array(TRAIN_MASK[None, :] / np.maximum(vega, 1e-3) ** 2)

    iters = 150 if quick else 500
    n_paths = 4096 if quick else 8192
    model = LocalVolatility(width=32, depth=3, key=jax.random.PRNGKey(0))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    sched = optax.cosine_decay_schedule(1e-2, iters, alpha=0.02)
    opt = optax.adam(sched)
    state = opt.init(params)

    def loss_fn(p, key):
        prices = mc_prices(eqx.combine(p, static), key, n_paths)
        return jnp.sum(w * (prices - jnp.array(target_p)) ** 2) / jnp.sum(TRAIN_MASK) / len(MATURITIES)

    @jax.jit
    def step(p, s, key):
        loss, g = jax.value_and_grad(loss_fn)(p, key)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, loss

    history = []
    key = jax.random.PRNGKey(42)
    for i in range(iters):
        key, sub = jax.random.split(key)
        params, state, loss = step(params, state, sub)
        history.append(float(jnp.sqrt(loss)) * 100)  # approx. RMS implied-vol error, vol points
        if i % 50 == 0 or i == iters - 1:
            print(f"iter {i:4d}  approx IV RMSE {history[-1]:.3f} vol pts  ({time.time() - t0:.0f}s)", flush=True)

    # ---- evaluate on fresh paths (many), every strike
    model = eqx.combine(params, static)
    eval_paths, chunks = (65536, 4) if quick else (65536, 8)
    keys = jax.random.split(jax.random.PRNGKey(7), chunks)
    price_fn = jax.jit(lambda k: mc_prices(model, k, eval_paths))
    fit_p = np.mean([np.array(price_fn(k)) for k in keys], axis=0)
    fit_iv = np.array(
        implied_vol(jnp.array(fit_p), S0, jnp.array(ALL_STRIKES)[None, :], jnp.array(MATURITIES)[:, None], R)
    )
    flat = float(np.mean(target_iv[:, TRAIN_MASK]))
    rmse = lambda a, m: float(np.sqrt(np.mean((a[:, m] - target_iv[:, m]) ** 2)) * 100)
    res = {
        "heston_params": HESTON,
        "maturities": MATURITIES.tolist(),
        "strikes": ALL_STRIKES.tolist(),
        "train_strikes": ALL_STRIKES[TRAIN_MASK].tolist(),
        "target_iv": target_iv.tolist(),
        "neural_lv_iv": fit_iv.tolist(),
        "iv_rmse_vol_pts": {
            "neural_lv_train": rmse(fit_iv, TRAIN_MASK),
            "neural_lv_held_out": rmse(fit_iv, ~TRAIN_MASK),
            "flat_black_scholes_train": rmse(np.full_like(target_iv, flat), TRAIN_MASK),
            "flat_black_scholes_held_out": rmse(np.full_like(target_iv, flat), ~TRAIN_MASK),
        },
        "training": {
            "iterations": iters,
            "paths_per_iter": n_paths,
            "steps_per_year": STEPS_PER_YEAR,
            "eval_paths": eval_paths * chunks,
            "seconds": time.time() - t0,
        },
    }
    print(json.dumps(res["iv_rmse_vol_pts"], indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / "local_vol_calibration.json", "w"), indent=2)

    # ---- figures
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.7), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for j, T in enumerate(MATURITIES):
        ax = axes[j]
        ax.axhline(100 * flat, color=COLORS["baseline"], ls="--", lw=1.2, label="Flat Black-Scholes")
        ax.plot(ALL_STRIKES, 100 * fit_iv[j], color=COLORS["model"], lw=2.2, label="Neural local-vol SDE")
        ax.plot(
            ALL_STRIKES[TRAIN_MASK],
            100 * target_iv[j, TRAIN_MASK],
            "o",
            color=COLORS["target"],
            ms=5,
            label="Heston quotes (train)",
        )
        ax.plot(
            ALL_STRIKES[~TRAIN_MASK],
            100 * target_iv[j, ~TRAIN_MASK],
            "o",
            mfc="white",
            mec=COLORS["target"],
            ms=5,
            label="Heston quotes (held out)",
        )
        ax.set(title=f"Maturity T = {T:g} yr", xlabel="strike / spot", ylabel="implied vol (%)" if j == 0 else None)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    tt = np.linspace(0.02, 1.0, 60)
    xx = np.linspace(np.log(0.7), np.log(1.3), 80)
    surf = np.array(jax.vmap(lambda t: jax.vmap(lambda x: model(t, x))(jnp.array(xx)))(jnp.array(tt)))
    im = axes[3].pcolormesh(np.exp(xx), tt, 100 * surf, cmap="Reds", shading="auto")
    fig.colorbar(im, ax=axes[3], label="local vol (%)")
    axes[3].set(title="Learned local volatility σθ(t, S)", xlabel="S / S0", ylabel="t (years)")
    save(fig, ROOT / "assets" / "local_vol_calibration.png")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history, color=COLORS["model"], lw=1)
    ax.set(
        yscale="log",
        xlabel="iteration",
        ylabel="≈ IV RMSE (vol pts)",
        title="Calibration loss (Monte Carlo, train strikes)",
    )
    save(fig, ROOT / "assets" / "local_vol_training.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
