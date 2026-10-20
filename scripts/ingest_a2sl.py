"""Turn the raw A2SL (University of Alabama ECE) microwave-imaging drops into manifests and scan caches.

Three sets live under ``data/a2sl/`` (raw, never committed):

* ``avocado_jun2023/`` — 15 Hass avocados bought 2023-06-22, scanned every other day (study days 1, 3, ..., 13).
  Avocados 1, 2, 3 and 11 were scanned at every visit; one further avocado was cut open (destructive check)
  at each visit: 4 on day 1, 5 on day 3, 6 on day 5, 7 on day 7, 8 on day 9, 9 on day 11, 10 on day 13.
  Per visit and avocado: one side photo (HEIC), six 75x75 reconstructions at rotation 0..300 deg (xlsx) and
  their PNG renderings; a cut photo for the avocado cut that day. Mass, length and diameter per fruit.
* ``pilot_may2023/`` — the May 2023 pilot: 6 avocados photographed daily for 13 days from 4 views
  (plus top/bottom from day 8), 12-angle reconstruction PNGs (no numeric grids), cut on staggered days.
* ``watermelon_2022/`` — 7 watermelons, each scanned once at up to 5 heights x 5 rotations x 5 frequencies
  (3-5 GHz, 96x96 "ImagePoints" grids); ripeness is a text label from the purchase log.

Outputs (``data/manifests/*.csv`` are committed; ``*.npz`` and converted JPEGs are not):

    data/manifests/a2sl_avocado_items.csv      one row per avocado: cut day, tracked-to-end flag, mass, size
    data/manifests/a2sl_avocado_visits.csv     one row per (avocado, visit): day, photo paths, scan keys
    data/manifests/a2sl_pilot_visits.csv       one row per (avocado, day, photo)
    data/manifests/a2sl_watermelon.csv         one row per scan grid, with the sample's ripeness label
    data/a2sl/scans_avocado_jun2023.npz        key "a{id}_d{day}_r{angle}" -> 75x75 float32 (NaN outside the disk)
    data/a2sl/scans_watermelon_2022.npz        key "s{id}_{height}_r{rot}_f{GHz}" -> 96x96 float32
    data/a2sl/<set>/jpg/...                    HEIC converted to 1024-px JPEG (sips on macOS, pillow-heif elsewhere)

Run once: ``python scripts/ingest_a2sl.py`` (HEIC conversion takes a few minutes).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A2SL = ROOT / "data" / "a2sl"
MAN = ROOT / "data" / "manifests"

JUN_DAYS = {1: "Jun_22", 3: "Jun_24", 5: "Jun_26", 7: "Jun_28", 9: "Jun_30", 11: "Jul_2", 13: "Jul_4"}
JUN_CUT_DAY = {4: 1, 5: 3, 6: 5, 7: 7, 8: 9, 9: 11, 10: 13}  # avocado -> study day it was cut
JUN_TRACKED = {1, 2, 3, 11}
PILOT_CUT_DAY = {3: 1, 4: 4, 5: 7, 6: 10, 1: 13, 2: 13}
WM_HEIGHTS = {
    "low": "low",
    "middlelow": "middle_low",
    "midlow": "middle_low",
    "middle": "middle",
    "mid": "middle",
    "middlehigh": "middle_high",
    "midhigh": "middle_high",
    "high": "high",
}


def convert_heic(src: Path, dst: Path, max_px: int = 1024) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-Z", str(max_px), str(src), "--out", str(dst)],
            check=True,
            capture_output=True,
        )
        return
    try:
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        im = Image.open(src)
        im.thumbnail((max_px, max_px))
        im.convert("RGB").save(dst, quality=90)
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("need macOS `sips` or `pip install pillow-heif` to convert HEIC") from e


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


# ----------------------------------------------------------------------------------------------- avocado jun 2023
def ingest_avocado(convert: bool) -> None:
    base = A2SL / "avocado_jun2023"
    props = pd.read_excel(base / "Avocado_information.xlsx", sheet_name="Avacado Physical Properties", header=None)
    props = props.set_index(0).T.rename(columns=lambda c: str(c).strip())
    props["Avacado"] = props["Avacado"].astype(int)
    items = []
    for _, r in props.iterrows():
        a = int(r["Avacado"])
        items.append(
            {
                "item_id": f"avo{a:02d}",
                "avocado": a,
                "origin": r.get("Origin"),
                "purchase_date": "2023-06-22",
                "mass_g": float(r["Mass(g)"]),
                "length_mm": float(r["Length(mm)"]),
                "diameter_mm": float(r["Diameter(mm)"]),
                "cut_day": JUN_CUT_DAY.get(a, np.nan),
                "tracked_to_end": a in JUN_TRACKED,
                "n_scan_visits": 7 if a in JUN_TRACKED else (1 if a in JUN_CUT_DAY else 0),
            }
        )
    pd.DataFrame(items).to_csv(MAN / "a2sl_avocado_items.csv", index=False)

    scans: dict[str, np.ndarray] = {}
    visits = []
    for k, (day, date) in enumerate(sorted(JUN_DAYS.items()), start=1):
        photo_dir = next(base.glob(f"01_photo/{k:02d}_day_{day}_{date}"))
        raw_dir = base / "02_raw_csv_files" / f"Day_{day}_{date}"
        png_dir = photo_dir / f"Day_{day}_{date}"
        ids = {int(re.match(r"avo_(\d+)_", p.name).group(1)) for p in raw_dir.glob("avo_*_*.xlsx")}
        ids |= {int(p.name[:2]) for p in photo_dir.glob("[0-9][0-9]_side.HEIC")}  # fruit 12-15 were photographed only
        ids = sorted(ids)
        for a in ids:
            keys = []
            for ang in (0, 60, 120, 180, 240, 300):
                f = raw_dir / f"avo_{a}_{ang}.xlsx"
                if f.exists():
                    key = f"a{a}_d{day}_r{ang}"
                    scans[key] = pd.read_excel(f, header=None).to_numpy(np.float32)
                    keys.append(key)
            side = photo_dir / f"{a:02d}_side.HEIC"
            cut = photo_dir / f"{a:02d}_cut.HEIC"
            side_jpg = base / "jpg" / f"day{day:02d}" / f"{a:02d}_side.jpg"
            cut_jpg = base / "jpg" / f"day{day:02d}" / f"{a:02d}_cut.jpg"
            if convert and side.exists():
                convert_heic(side, side_jpg)
            if convert and cut.exists():
                convert_heic(cut, cut_jpg)
            pngs = sorted(png_dir.glob(f"avo_{a}_*.png"))
            visits.append(
                {
                    "item_id": f"avo{a:02d}",
                    "avocado": a,
                    "day": day,
                    "date": f"2023-{date.replace('Jun_', '06-').replace('Jul_', '07-').zfill(5)}",
                    "visit_index": k,
                    "side_photo": rel(side_jpg) if side.exists() else "",
                    "side_photo_heic": rel(side) if side.exists() else "",
                    "cut_photo": rel(cut_jpg) if cut.exists() else "",
                    "cut_today": cut.exists(),
                    "scan_keys": ";".join(keys),
                    "scan_pngs": ";".join(rel(p) for p in pngs),
                }
            )
    # alternative reconstructions of angle 0 for the tracked fruit
    for f in (base / "02_raw_csv_files" / "DMAS_RP9.3").glob("*.xlsx"):
        m = re.match(r"(\d+)_0_RP9\.3_day_(\d+)\.xlsx", f.name)
        if m:
            scans[f"a{int(m.group(1))}_d{int(m.group(2))}_r0_dmas"] = pd.read_excel(f, header=None).to_numpy(np.float32)
    for f in (base / "02_raw_csv_files" / "raw_data_CDAS").glob("*/*.xlsx"):
        m = re.match(r"(\d+)_0_RP11\.xlsx", f.name)
        date = f.parent.name  # 6_22 ...
        day = {v.replace("Jun_", "6_").replace("Jul_", "7_"): d for d, v in JUN_DAYS.items()}.get(date)
        if m and day is not None:
            scans[f"a{int(m.group(1))}_d{day}_r0_cdas"] = pd.read_excel(f, header=None).to_numpy(np.float32)
    pd.DataFrame(visits).sort_values(["avocado", "day"]).to_csv(MAN / "a2sl_avocado_visits.csv", index=False)
    np.savez_compressed(A2SL / "scans_avocado_jun2023.npz", **scans)
    print(f"avocado_jun2023: {len(items)} fruit, {len(visits)} visits, {len(scans)} scan grids")


# ----------------------------------------------------------------------------------------------- pilot may 2023
def ingest_pilot(convert: bool) -> None:
    base = A2SL / "pilot_may2023"
    rows = []
    for day_dir in sorted(base.glob("[0-9][0-9]_[dD]ay_[0-9][0-9]")):
        day = int(day_dir.name[:2])
        for p in sorted(list(day_dir.glob("*.HEIC")) + list(day_dir.glob("*.JPEG"))):
            m = re.match(r"(\d\d)_(.+)\.(HEIC|JPEG)", p.name)
            if not m:
                continue
            a, view = int(m.group(1)), m.group(2)
            kind = "cut" if "cut" in view else "photo"
            if p.suffix == ".HEIC":
                jpg = base / "jpg" / f"day{day:02d}" / f"{a:02d}_{view}.jpg"
                if convert:
                    convert_heic(p, jpg)
                path = rel(jpg)
            else:
                path = rel(p)
            rows.append(
                {
                    "item_id": f"pilot{a:02d}",
                    "avocado": a,
                    "day": day,
                    "view": view,
                    "kind": kind,
                    "photo": path,
                    "cut_day": PILOT_CUT_DAY.get(a, np.nan),
                }
            )
        for s in sorted((day_dir / "scans").glob("*.png")):
            m = re.match(r"(\d\d)_avo_(\d+)\.png", s.name)
            if m:
                rows.append(
                    {
                        "item_id": f"pilot{int(m.group(1)):02d}",
                        "avocado": int(m.group(1)),
                        "day": day,
                        "view": f"scan_r{m.group(2)}",
                        "kind": "scan_png",
                        "photo": rel(s),
                        "cut_day": PILOT_CUT_DAY.get(int(m.group(1)), np.nan),
                    }
                )
    df = pd.DataFrame(rows).sort_values(["avocado", "day", "kind", "view"])
    df.to_csv(MAN / "a2sl_pilot_visits.csv", index=False)
    print(
        f"pilot_may2023: {df.item_id.nunique()} fruit, {df[df.kind == 'photo'].shape[0]} photos, "
        f"{df[df.kind == 'cut'].shape[0]} cut photos, {df[df.kind == 'scan_png'].shape[0]} scan PNGs"
    )


# ----------------------------------------------------------------------------------------------- watermelon 2022
def ingest_watermelon() -> None:
    base = A2SL / "watermelon_2022"
    log = pd.read_excel(base / "Watermelon Data Log.xlsx", sheet_name="Watermelon", header=None)
    hdr = log.iloc[1].tolist()
    log = log.iloc[2:].copy()
    log.columns = [str(h).replace("\n", " ").strip() for h in hdr]
    log = log[log.iloc[:, 0].astype(str).str.startswith("Sample")]
    meta = {}
    for _, r in log.iterrows():
        sid = int(str(r.iloc[0]).split()[-1])
        meta[sid] = {
            "place": r.get("Place of Purchase"),
            "ripeness": r.get("Ripeness"),
            "purchase_date": r.get("Date of Purchase"),
            "length_cm": r.get("Length (Longest Dimension)"),
            "width_cm": r.get("Width (Shortest Dimension)"),
            "origin": r.get("Origin of Production"),
            "type": r.get("Type"),
            "size": r.get("Size"),
            "weight_kg": r.get("Weight (kg)"),
        }
    scans, rows = {}, []
    for sdir in sorted(base.glob("Sample *")):
        sid = int(sdir.name.split()[-1])
        for csv in sorted(sdir.rglob("ImagePoints*.csv")):
            ghz = float(re.search(r"ImagePoints([\d.]+)\.csv", csv.name).group(1))
            pos = csv.parent.name if csv.parent != next(sdir.glob("*Image*")) else "single"
            m = re.search(
                r"(Low|MiddleLow|MidLow|Middle|Mid|MiddleHigh|MidHigh|High)(\d*)$",
                pos.replace("Watermelon", "").replace("WMS9", "").replace("Sample", ""),
            )
            m2 = re.search(r"^(\d+)(Low|Middle|High)$", pos.replace("Watermelon", ""))
            if m2:
                height, rot = WM_HEIGHTS[m2.group(2).lower()], int(m2.group(1))
            elif m:
                height, rot = WM_HEIGHTS[m.group(1).lower()], int(m.group(2) or 0)
            else:
                height, rot = "single", 0
            key = f"s{sid}_{height}_r{rot}_f{ghz:g}"
            scans[key] = pd.read_csv(csv, header=None).to_numpy(np.float32)
            rows.append(
                {
                    "item_id": f"wm{sid:02d}",
                    "sample": sid,
                    "height": height,
                    "rotation_deg": rot,
                    "freq_ghz": ghz,
                    "scan_key": key,
                    "csv": rel(csv),
                    **meta.get(sid, {}),
                }
            )
    pd.DataFrame(rows).to_csv(MAN / "a2sl_watermelon.csv", index=False)
    np.savez_compressed(A2SL / "scans_watermelon_2022.npz", **scans)
    print(f"watermelon_2022: {len(meta)} logged samples, {len(rows)} scan grids")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-convert", action="store_true", help="skip HEIC -> JPEG conversion")
    args = ap.parse_args()
    if not A2SL.exists():
        sys.exit(f"{A2SL} not found; copy the three raw folders there first (see docs/data_a2sl.md)")
    MAN.mkdir(parents=True, exist_ok=True)
    ingest_avocado(convert=not args.no_convert)
    ingest_pilot(convert=not args.no_convert)
    ingest_watermelon()


if __name__ == "__main__":
    main()
