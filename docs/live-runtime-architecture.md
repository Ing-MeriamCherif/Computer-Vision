# Live runtime architecture

`main.py` delegates to `tools/p123_live_app.py`. `P123LiveRuntime` captures a physical camera into a capacity-one latest-frame slot. Depth, CUDA geometry, temporal fusion, hand tracking, and HandXYZ run asynchronously; the UI renders the newest completed snapshot and exposes capture/inference/render FPS and latency.

Mode 7 consumes the shared `GeometryState` and HandXYZ contract. Backend selection is automatic by default: RTX + OptiX selects `RTX_OPTIX`, otherwise OpenGL raster is used, with CPU as an emergency fallback. Camera-source or invalid-geometry changes reset renderer history.
