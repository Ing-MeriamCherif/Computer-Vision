#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-gpu-ui.txt
.venv/bin/playwright install chromium
.venv/bin/hf download depth-anything/Depth-Anything-V2-Small-hf \
  --local-dir models/depth-anything-v2-small --max-workers 8
.venv/bin/python -m tools.cuda_smoke

