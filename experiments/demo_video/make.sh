#!/usr/bin/env bash
# Rebuild assets/demo.mp4 and assets/demo.gif from real runs. Needs ffmpeg and `pip install -e ".[dev]"`.
# Takes ~10 min on an M1 Pro: most of it is the full 500-iteration calibration re-run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/../.."; C="$HERE/_cache"
mkdir -p "$C"
(cd "$ROOT" && python lecture/neural_sde_demo.py) > "$C/demo_out.txt"
(cd "$ROOT" && python -m pytest 2>&1 | tail -5) > "$C/pytest_out.txt"
python "$HERE/record_calibration.py"
python "$HERE/render_demo.py" "$ROOT/assets/demo.mp4"
ffmpeg -loglevel error -y -i "$ROOT/assets/demo.mp4" -vf "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" "$ROOT/assets/demo.gif"
