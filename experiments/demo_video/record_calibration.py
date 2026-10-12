"""Re-run the full local-vol calibration, recording the fitted smile every few iterations for the demo video."""
import sys, json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

import equinox as eqx, jax, jax.numpy as jnp, numpy as np, optax
import local_vol_calibration as lv
from nsde import LocalVolatility, bs_vega, implied_vol

OUT = pathlib.Path(__file__).parent / "_cache" / "calib_frames.json"
iters, n_paths, every = 500, 8192, 5

target_p, target_iv = lv.market()
vega = np.array(bs_vega(lv.S0, lv.ALL_STRIKES[None, :], lv.MATURITIES[:, None], lv.R, target_iv))
w = jnp.array(lv.TRAIN_MASK[None, :] / np.maximum(vega, 1e-3) ** 2)
model = LocalVolatility(width=32, depth=3, key=jax.random.PRNGKey(0))
params, static = eqx.partition(model, eqx.is_inexact_array)
opt = optax.adam(optax.cosine_decay_schedule(1e-2, iters, alpha=0.02))
state = opt.init(params)

def loss_fn(p, key):
    prices = lv.mc_prices(eqx.combine(p, static), key, n_paths)
    return jnp.sum(w * (prices - jnp.array(target_p)) ** 2) / jnp.sum(lv.TRAIN_MASK) / len(lv.MATURITIES)

@jax.jit
def step(p, s, key):
    loss, g = jax.value_and_grad(loss_fn)(p, key)
    upd, s = opt.update(g, s, p)
    return optax.apply_updates(p, upd), s, loss

@jax.jit
def snapshot_iv(p):
    prices = lv.mc_prices(eqx.combine(p, static), jax.random.PRNGKey(7), 32768)
    return implied_vol(prices, lv.S0, jnp.array(lv.ALL_STRIKES)[None, :], jnp.array(lv.MATURITIES)[:, None], lv.R)

tt = jnp.linspace(0.02, 1.0, 40); xx = jnp.linspace(np.log(0.7), np.log(1.3), 60)
@jax.jit
def surface(p):
    m = eqx.combine(p, static)
    return jax.vmap(lambda t: jax.vmap(lambda x: m(t, x))(xx))(tt)

frames, history = [], []
key = jax.random.PRNGKey(42)
for i in range(iters + 1):
    if i % every == 0:
        frames.append({"iter": i, "iv": np.array(snapshot_iv(params)).tolist(),
                       "surface": np.array(surface(params)).tolist()})
        print(f"iter {i}  loss {history[-1] if history else float('nan'):.3f}", flush=True)
    if i == iters:
        break
    key, sub = jax.random.split(key)
    params, state, loss = step(params, state, sub)
    history.append(float(jnp.sqrt(loss)) * 100)

json.dump({"strikes": lv.ALL_STRIKES.tolist(), "train_mask": lv.TRAIN_MASK.tolist(),
           "maturities": lv.MATURITIES.tolist(), "target_iv": np.array(target_iv).tolist(),
           "surface_t": np.array(tt).tolist(), "surface_s": np.exp(np.array(xx)).tolist(),
           "history": history, "frames": frames}, open(OUT, "w"))
print("wrote", OUT)
