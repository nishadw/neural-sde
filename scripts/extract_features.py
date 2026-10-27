"""Cache frozen-backbone features for every photo of a dataset (needs the [features] extra).

python scripts/extract_features.py --data avocado --root data/avocado --out data/features/avocado_resnet18.npz
python scripts/extract_features.py --data stevens --manifest data/stevens/manifest.csv \
    --out data/features/stevens_resnet18.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ripen.data.avocado import load_avocado
from ripen.data.features import extract_features, save_feature_cache
from ripen.data.stevens import load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["avocado", "stevens", "a2sl"], required=True)
    ap.add_argument("--root", default="data/avocado")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--manifest", default="data/stevens/manifest.csv")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "convnext_tiny"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.data == "avocado":
        items = load_avocado(args.root, args.labels, args.images_dir)
    elif args.data == "a2sl":
        from ripen.data.a2sl import load_a2sl_avocado, load_a2sl_pilot

        items = load_a2sl_avocado(labels=None) + load_a2sl_pilot()
    else:
        items = load_manifest(args.manifest)
    paths = sorted({p for it in items for v in it.visits for p in v.image_paths})
    missing = [p for p in paths if not Path(p).exists()]
    if len(missing) > 0.01 * len(paths):  # a handful is expected (the avocado release lacks 12 photos)
        raise FileNotFoundError(f"{len(missing)} image paths do not exist, e.g. {missing[:3]}")
    if missing:
        print(f"skipping {len(missing)} missing photos, e.g. {missing[:3]}")
        paths = sorted(set(paths) - set(missing))
    print(f"{len(items)} items, {len(paths)} photos -> {args.backbone} on {args.device}")
    feats = extract_features(paths, args.backbone, args.batch_size, args.device)
    save_feature_cache(args.out, feats)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
