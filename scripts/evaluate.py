"""Re-evaluate a finished run, optionally on different data settings (sparser visits, other n_obs).

python scripts/evaluate.py results/runs/latent_sde_avocado                 # test set, as trained
python scripts/evaluate.py results/runs/latent_sde_avocado --every 2       # photos every other day
python scripts/evaluate.py results/runs/latent_sde_avocado --n-obs 1 3 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from omegaconf import OmegaConf

from ripen.pipeline import build_from_run, evaluate_and_write, load_items, select, splits_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=int, default=None, help="subsample visits to every k-th before predicting")
    ap.add_argument("--n-obs", type=int, nargs="*", default=None, help="prefix lengths to evaluate (default: all)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    model, cfg, info = build_from_run(args.run_dir)
    cfg = OmegaConf.merge(cfg, {})
    if args.every:
        cfg.data.subsample_every = args.every
    if args.n_obs:
        cfg.eval.n_obs_list = list(args.n_obs)
    items = load_items(cfg.data)
    sp = splits_for(cfg, items)
    tag = args.tag or f"{args.split}" + (f"_every{args.every}" if args.every else "") + (
        f"_nobs{'-'.join(map(str, args.n_obs))}" if args.n_obs else ""
    )
    m = evaluate_and_write(
        model, select(items, sp[args.split]), info, cfg, Path(args.run_dir), jax.random.PRNGKey(1), tag=tag
    )
    o = m["overall"]
    print(
        f"{tag}: MAE {o['mae']:.3f} CRPS {o['crps']:.3f} cov80 {o['coverage80']:.2f} "
        f"width80 {o['width80']:.2f} C {o['cindex']:.3f}"
    )


if __name__ == "__main__":
    main()
