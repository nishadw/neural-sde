"""Download the 'Hass' Avocado Ripening Photographic Dataset (Mendeley 10.17632/3xd9n945v8.1, CC BY 4.0).

The public API lists the root folder (the label spreadsheet) and one sub-folder with the 14,710 JPEGs
(~470 MB). This script lists both, downloads everything into ``data/avocado/`` (spreadsheet at the top,
images under ``images/``), skips files that already exist with the right size, and verifies SHA-256 when
asked. Safe to re-run; resumes where it stopped.

    python scripts/download_avocado.py                 # 16 parallel downloads
    python scripts/download_avocado.py --verify        # also check SHA-256 of every file
    nohup python scripts/download_avocado.py > data/avocado/download.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "3xd9n945v8"
VERSION = 1
API = f"https://data.mendeley.com/public-api/datasets/{DATASET}"
HEADERS = {"User-Agent": "Mozilla/5.0 (ripen downloader)"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def list_files() -> list[dict]:
    """Every file in the dataset with its download URL, size and sha256 (root folder + sub-folders)."""
    folders = [("root", None)] + [(f["name"], f["id"]) for f in get_json(f"{API}/folders/{VERSION}")]
    out = []
    for name, fid in folders:
        start = 0
        while True:
            u = f"{API}/files?folder_id={fid or 'root'}&version={VERSION}&$start={start}&$limit=1000"
            chunk = get_json(u)
            if not chunk:
                break
            for f in chunk:
                cd = f.get("content_details") or {}
                out.append(
                    {
                        "folder": name,
                        "filename": f["filename"],
                        "size": int(f.get("size") or cd.get("size") or 0),
                        "sha256": cd.get("sha256_hash"),
                        "url": cd.get("download_url"),
                    }
                )
            start += len(chunk)
            if len(chunk) < 1000:
                break
    return out


def target_path(root: Path, f: dict) -> Path:
    return root / f["filename"] if f["folder"] == "root" else root / "images" / f["filename"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(f: dict, root: Path, verify: bool) -> tuple[str, str]:
    dst = target_path(root, f)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == f["size"]:
        if not verify or (f["sha256"] and sha256_of(dst) == f["sha256"]):
            return f["filename"], "skip"
    err = "unknown"
    for attempt in range(4):
        try:
            req = urllib.request.Request(f["url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as fh:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            if dst.stat().st_size != f["size"]:
                raise OSError(f"size mismatch {dst.stat().st_size} != {f['size']}")
            if verify and f["sha256"] and sha256_of(dst) != f["sha256"]:
                raise OSError("sha256 mismatch")
            return f["filename"], "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return f["filename"], f"FAILED: {err}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/avocado")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    listing = root / "mendeley_files.json"
    if listing.exists():
        files = json.loads(listing.read_text())
    else:
        files = list_files()
        listing.write_text(json.dumps(files))
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB -> {root}", flush=True)
    counts = {"ok": 0, "skip": 0, "failed": 0}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(download, f, root, args.verify) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            if status.startswith("FAILED"):
                counts["failed"] += 1
                print(f"{name}: {status}", flush=True)
            else:
                counts[status] += 1
            if i % 500 == 0 or i == len(files):
                msg = f"{i}/{len(files)}  ok {counts['ok']}  skip {counts['skip']}  failed {counts['failed']}"
                print(f"{msg}  {time.time() - t0:.0f}s", flush=True)
    print("done", counts, flush=True)
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()
