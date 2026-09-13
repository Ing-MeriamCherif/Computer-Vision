#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/playwright install chromium
.venv/bin/hf download depth-anything/Depth-Anything-V2-Small-hf \
  --local-dir models/depth-anything-v2-small --max-workers 8
if [[ ! -f models/hand_landmarker.task ]]; then
  echo "Missing models/hand_landmarker.task (download the MediaPipe Hand Landmarker asset before live hand/XYZ modes)." >&2
  exit 2
fi
.venv/bin/python -m tools.cuda_smoke
