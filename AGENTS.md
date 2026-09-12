# AGENTS.md

Real-time monocular depth; 30+ FPS target (baseline: V2-Small @ 420, ~31 FPS — see `depth/config.py`).

## Setup

```bash
pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q   # 15 tests, <2s, no GPU/camera/weights (fake backend)
```

No `pyproject.toml`, lint, or CI. Single test file: `tests/test_async_depth.py` (`python -m pytest tests/test_async_depth.py -v`).

## Entrypoints (`python -m ...`)

- `depth` / `depth.benchmark` — synthetic benchmark on random frames, no camera needed. `depth/__main__.py` calls `main()` at import (no `__main__` guard).
- `depth.camera_benchmark` — captures camera frames up front, then times inference only.
- `depth.camera_benchmark_live`, `depth.camera_live`, `depth.live_viewer` — need camera + display (`q` quits/skips).
- `depth.camera_preview` — no inference; use to find working `--camera-index` (`ls /dev/video*`).
- `depth.depth_matrix` — one frame, prints depth grid to terminal.
- `model_comparison.benchmark` — Small vs Base only (docstring says Large, `MODELS` dict has no Large).

`live_viewer` defaults to `DEPTH_CONFIG` (`depth/config.py`, frozen dataclass singleton — tests assert its values, don't change without updating tests).

## Architecture

- `DepthModel(backend, device, input_size, fp16)` in `depth/model.py` is the inference API; `infer(frame)` takes `(H, W, 3)` uint8 BGR, returns `DepthState` (`depth_map` float32 `(H, W)`, `timestamp`, `scale_mode`, `valid_mask`, `backend_name`, `inference_ms`).
- Backend protocol in `depth/backends/<name>.py`: `__init__(device, input_size, fp16)`, `predict(frame) -> (depth, mask)`, `warmup()`, class attr `scale_mode`; register in `BACKENDS` in `depth/backends/__init__.py` (only `depth_anything_v2_small` exists).
- Async path (`depth/async_depth.py`, used by `live_viewer`): `LatestFrameBuffer` → `DepthWorker` thread (always newest frame, drops stale) → `LatestDepthBuffer`. `validate_depth_state()` is the output-shape/range/NaN check helper.

## Gotchas

- Depth is relative, min-max normalized to [0,1] per frame and **inverted** (closer = 1.0/bright). Flat frames yield all zeros. `valid_mask` is always all-`True`.
- Backend resizes input to `input_size` square (`INTER_AREA`, aspect ratio not preserved) and resizes output back to input resolution (`INTER_LINEAR`).
- `fp16` applies on CUDA only (`fp16 and device.type == "cuda"`); HF pipeline `device` arg is `0`/`-1`, not `torch.device`. `--device cuda` falls back to CPU silently when CUDA is unavailable.
- HF pipeline lazy-loads once per backend instance (`self._pipe` guard); first `warmup()`/`predict()` downloads `depth-anything/Depth-Anything-V2-Small-hf` — needs network.
- Benchmarks must `torch.cuda.synchronize()` around timing, `reset_peak_memory_stats()` before, and `del model` + `gc.collect()` + `empty_cache()` between resolutions (see `depth/benchmark.py::benchmark_single`).
- All camera open paths force MJPG + 640x480 and discard ~30 warm-up frames (auto-exposure/blank frames). `camera_benchmark --show` visualizes outside the timed region.
- Result CSVs (`benchmark_results.csv`, `camera_*.csv`, `model_comparison/results.csv`) are gitignored; benchmark writes to CWD by default.
