"""Write the avocado set in the Stevens manifest schema, to exercise the stub loader end to end.

python scripts/export_manifest.py --root data/avocado --out data/stevens/manifest.csv
"""

from __future__ import annotations

import argparse

from ripen.data.avocado import load_avocado
from ripen.data.stevens import export_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/avocado")
    ap.add_argument("--out", default="data/stevens/manifest.csv")
    args = ap.parse_args()
    items = load_avocado(args.root)
    export_manifest(items, args.out)
    print(f"wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
