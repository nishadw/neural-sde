"""Item-level splits. Never split by photo: two photos of one fruit must not straddle train and test."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .schema import Item


def make_splits(
    items: list[Item], fracs: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 0, stratify: str = "storage"
) -> dict[str, list[str]]:
    """Random 70/15/15 split of item ids, stratified by ``storage`` (or ``produce``)."""
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must sum to 1")
    rng = np.random.default_rng(seed)
    by_key: dict[str, list[str]] = defaultdict(list)
    for it in items:
        by_key[getattr(it, stratify)].append(it.item_id)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for key in sorted(by_key):
        ids = sorted(by_key[key])
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        out["train"] += ids[:n_tr]
        out["val"] += ids[n_tr : n_tr + n_va]
        out["test"] += ids[n_tr + n_va :]
    return out


def leave_one_group_out(items: list[Item], held_out: str, val_frac: float = 0.15, seed: int = 0, key: str = "storage"):
    """Train on every group but ``held_out``; test on ``held_out``. Validation is carved from the training groups."""
    rng = np.random.default_rng(seed)
    test = sorted(it.item_id for it in items if getattr(it, key) == held_out)
    rest = sorted(it.item_id for it in items if getattr(it, key) != held_out)
    if not test or not rest:
        raise ValueError(f"held_out={held_out!r} leaves an empty split")
    rng.shuffle(rest)
    n_va = int(round(val_frac * len(rest)))
    return {"train": rest[n_va:], "val": rest[:n_va], "test": test}


def save_splits(splits: dict[str, list[str]], path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": splits, **(extra or {})}
    Path(path).write_text(json.dumps(payload, indent=2))


def load_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())["splits"]


def select(items: list[Item], ids: list[str]) -> list[Item]:
    want = set(ids)
    return [it for it in items if it.item_id in want]
