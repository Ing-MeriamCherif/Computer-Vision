# Geometry performance and migration notes

`tools/geometry_benchmark.py` runs warmups followed by repeated measurements
and reports mean, p50, p95, p99, and max for each stage.  Use JSON for CI or
machine-readable capture:

```bash
python3 -m tools.geometry_benchmark --width 320 --height 180 --warmups 5 --iterations 30 --json
python3 -m tools.geometry_benchmark --width 640 --height 360 --warmups 3 --iterations 10 --json
python3 -m tools.geometry_benchmark --width 1280 --height 720 --warmups 1 --iterations 3 --json
```

The benchmark covers backprojection, all normal modes, generic/depth-aware
warping, affine/scale-only alignment, fresh/history-only temporal updates, and
labels percentile confidence when a caller requests fewer than five or ten
samples, and includes quality fields (valid geometry/normals, alignment model,
residual, and sample counts). For stable tails use at least 30 samples at
320x180, 10 at 640x360, and 5 at 1280x720; small exploratory runs are marked
limited. `tools.geometry_stress` provides deterministic multi-rate and
fault-injection runs.

The optional reduced-resolution flow study is reproducible with
`python3 -m tools.flow_resolution_benchmark --iterations 3`; it reports DIS and
Farneback EPE (mean/median/p95) alongside runtime for scales 1.0, 0.5, and
0.25. On the checked-in synthetic texture, Farneback 0.5 was the best CPU
trade-off while DIS 1.0 had the lowest error but substantially higher cost.

The principal avoidable allocation removed in Phase 4 is per-frame camera-grid
construction: backprojection uses an eight-entry bounded cache of read-only
float32 normalized-ray fields keyed by camera intrinsics and resolution.
Alignment supports deterministic grid-stratified `max_samples` (the temporal
default is 10,000), and converts only selected fitting vectors to float64.
Multi-scale normal selection now streams one radius at a time while preserving
the acceptance-first/best-fallback result policy. Inputs are not mutated and
OpenCV conversion only makes a contiguous uint8 buffer when required by the
backend.

CPU NumPy/OpenCV remains a prototype, not a 60 FPS claim.  At 1280x720 the
temporal update is still on the order of seconds on this host; the dominant
future migration targets are optical flow, depth-aware warp, normals, and
temporal fusion/backprojection.  These operations are data-parallel and can
share device-resident float32 buffers.  NVIDIA Optical Flow, CUDA kernels,
TensorRT, and zero-copy interop are intentionally deferred.

The checked-in baseline snapshot (CPU, repeated measurements) is approximately:

| Resolution | Backprojection mean | Edge-aware normals mean | Depth-aware warp mean | Fresh temporal update mean | History-only mean |
|---|---:|---:|---:|---:|---:|
| 320x180 | 0.72 ms | 42.53 ms | 15.30 ms | 146.51 ms | 73.10 ms |
| 640x360 | 3.26 ms | 122.50 ms | 71.86 ms | 591.81 ms | 380.84 ms |
| 1280x720 | 8.05 ms | 470.72 ms | 280.09 ms | 2982.95 ms | 1812.50 ms |

The corresponding JSON contains p50/p95/p99/max, quality metrics, and iteration
counts; the
1280 row used two repeated measurements because each full update is expensive.
