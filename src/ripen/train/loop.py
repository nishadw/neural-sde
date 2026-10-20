"""One trainer for every model: Adam, item-level minibatches, early stopping on validation loss."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.collate import DatasetInfo, collate
from ..data.schema import Item


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    patience: int = 10
    n_obs_mode: str = "random"  # "random" (visible-prefix dropout) | "all"
    seed: int = 0
    log_every: int = 1
    max_steps: int | None = None  # for smoke tests

    def to_dict(self) -> dict:
        return asdict(self)


def _batches(items: list[Item], info: DatasetInfo, bs: int, rng: np.random.Generator, n_obs_mode: str, shuffle: bool):
    order = np.arange(len(items))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(items), bs):
        chunk = [items[j] for j in order[i : i + bs]]
        yield collate(chunk, info, n_obs=n_obs_mode, rng=rng, item_index=[int(j) for j in order[i : i + bs]])


def evaluate_loss(model, items: list[Item], info: DatasetInfo, bs: int, key, n_obs_mode: str = "random", seed=123):
    rng = np.random.default_rng(seed)
    step = eqx.filter_jit(lambda m, b, k: m.loss(b, k))
    tot, n = 0.0, 0
    for b in _batches(items, info, bs, rng, n_obs_mode, shuffle=False):
        key, sub = jax.random.split(key)
        tot += float(step(model, b, sub)) * b.batch_size
        n += b.batch_size
    return tot / max(n, 1)


def train(
    model,
    train_items: list[Item],
    val_items: list[Item],
    info: DatasetInfo,
    cfg: TrainConfig,
    key,
    out_dir: str | Path | None = None,
    log=print,
):
    """Returns ``(best_model, history)``. ``history`` is a list of dicts; also written to ``out_dir/history.csv``."""
    rng = np.random.default_rng(cfg.seed)
    opt = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(cfg.lr, weight_decay=cfg.weight_decay))
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(model, opt_state, batch, key):
        loss, grads = eqx.filter_value_and_grad(lambda m: m.loss(batch, key))(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    if hasattr(model, "fit_closed_form"):
        model = model.fit_closed_form(collate(train_items, info, n_obs="all"))

    history: list[dict] = []
    best, best_val, bad, n_steps = model, float("inf"), 0, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        tot, n = 0.0, 0
        for batch in _batches(train_items, info, cfg.batch_size, rng, cfg.n_obs_mode, shuffle=True):
            key, sub = jax.random.split(key)
            model, opt_state, loss = step(model, opt_state, batch, sub)
            if not bool(jnp.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            tot += float(loss) * batch.batch_size
            n += batch.batch_size
            n_steps += 1
            if cfg.max_steps is not None and n_steps >= cfg.max_steps:
                break
        if hasattr(model, "fit_closed_form"):
            model = model.fit_closed_form(collate(train_items, info, n_obs="all"))
        key, sub = jax.random.split(key)
        val = evaluate_loss(model, val_items, info, cfg.batch_size, sub, cfg.n_obs_mode) if val_items else float("nan")
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": val, "seconds": time.time() - t0}
        history.append(rec)
        if epoch % cfg.log_every == 0:
            log(f"epoch {epoch:3d}  train {rec['train_loss']:.4f}  val {val:.4f}  {rec['seconds']:.0f}s")
        if val < best_val or not val_items:
            best, best_val, bad = model, val, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop at epoch {epoch} (best val {best_val:.4f})")
                break
        if cfg.max_steps is not None and n_steps >= cfg.max_steps:
            break
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        save_model(best, out / "model.eqx")
    return best, history


def save_model(model, path: str | Path) -> None:
    eqx.tree_serialise_leaves(str(path), model)


def load_model(template, path: str | Path):
    """``template`` is a freshly built model with the same config; its leaves are overwritten from ``path``."""
    return eqx.tree_deserialise_leaves(str(path), template)
