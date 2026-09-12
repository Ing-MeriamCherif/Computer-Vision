# Upstream branch integrations

This directory contains the actual source trees imported from the colleague
repository, not a rewrite of their API:

| Source | Preserved files | Live use |
| --- | --- | --- |
| `main` `ee90d026` | hand tracker, EMA/One-Euro filters, vector math, validation renderer | `geometry.hand_control` and `geometry.lighting` |
| `talel-hand` `f6e65831` | `config.py`, `upstream_main.py`, `.env.example` with reacquire/hold/LK tuning | settings/provenance reference |
| `feature/depth` `f8ecd09d` | complete `depth/` and `model_comparison/` packages | `ColleagueDepthProvider` via `NRW_DEPTH_SOURCE=colleague` |

The host UI and renderer remain in `tools/`, but their hand stage now calls
the preserved colleague `create_tracker()` and its One-Euro filter, while the
light stage calls the preserved `palm_to_light()` math. The depth adapter maps
the preserved branch's `DepthState` into the shared contract and keeps frame
IDs so it can participate in the live latest-state pipeline.
