"""Write the item-level split files every run reads, so all models see identical train/val/test items.

python scripts/make_splits.py data=avocado                      # random 70/15/15, seed from splits.seed
python scripts/make_splits.py data=avocado splits.mode=logo splits.held_out=T10
"""

from __future__ import annotations

from collections import Counter

import hydra
from omegaconf import DictConfig

from ripen.pipeline import load_items, select, splits_for


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    items = load_items(cfg.data)
    sp = splits_for(cfg, items)
    for name, ids in sp.items():
        sub = select(items, ids)
        groups = Counter(it.storage for it in sub)
        cens = sum(it.right_censored for it in sub)
        print(f"{name:5s} {len(sub):4d} items  groups {dict(groups)}  right-censored {cens}")


if __name__ == "__main__":
    main()
