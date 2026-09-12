# AGENTS.md

Real-time monocular depth; 30+ FPS target (benchmark reports fps per resolution).

## Setup

No `pyproject.toml`, tests, lint, or CI. Install deps via `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running

```bash
python -m depth --backend depth_anything_v2_small --device cuda --resolutions 336,420,518
python -m depth.benchmark --num-frames 50 --output benchmark_results.csv
```

- Entry chain: `python -m depth` → `depth/__main__.py` (calls `main()` at import, no `__main__` guard) → `depth/benchmark.py::main`. No root `main.py`.
- `DepthModel(backend, device, input_size, fp16)` in `depth/model.py` is the inference API; `infer(frame)` takes `(H, W, 3)` uint8 and returns `DepthState` (`depth_map` float32, `timestamp`, `scale_mode`, `valid_mask`, `backend_name`, `inference_ms`).
- To add a backend: subclass in `depth/backends/<name>.py` with `__init__(device, input_size, fp16)`, `predict(frame) -> (depth, mask)`, `warmup()`, class attr `scale_mode`, and register in `BACKENDS` in `depth/backends/__init__.py`.

## Gotchas

- Depth is relative, min-max normalized to [0, 1] per frame; flat frames yield all zeros. `valid_mask` is always all-`True`.
- Backend assumes BGR input (`cv2.cvtColor` to RGB for the HF pipeline), resizes input to `input_size` square before inference (aspect ratio not preserved — fine for benchmarking), and resizes output back to input resolution.
- `fp16` is passed as `torch_dtype` to the HF pipeline and only applies on CUDA (`fp16 and device.type == "cuda"`); pipeline `device` arg is `0`/`-1`, not a `torch.device`.
- The HF pipeline is lazy-loaded once per backend instance (guard on `self._pipe`); benchmark frees each model (`del` + `gc.collect()` + `torch.cuda.empty_cache()`) between resolutions.
- `--device cuda` falls back to CPU silently when CUDA is unavailable (`torch.device(...) if torch.cuda.is_available() else "cpu"`).
- First `warmup()`/`predict()` lazily builds the HF pipeline and downloads `depth-anything/Depth-Anything-V2-Small-hf`; needs network on first run.
- Benchmark timing wraps `model.infer` on random frames (not camera data) and writes CSV (`backend,resolution,device,fp16,median_ms,p95_ms,fps,vram_mb,num_frames`) to CWD.
