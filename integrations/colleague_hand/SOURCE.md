# Colleague hand/light implementation

These modules are imported from `Ing-MeriamCherif/Computer-Vision` and kept
under their original filenames so the merge is auditable:

- `main` @ `ee90d026f72a07ab03f93dea4178897157cb3c16` (hand/vector/rendering)
- `talel-hand` @ `f6e65831f27bbb7e1bb8e9330ee5db03a3130464` (reacquire/hold/LK fix)

The only edits after import are package-relative imports (`config` and
`utils`) so the source can be loaded without changing the algorithms.
`geometry.hand_control` calls the real `create_tracker()` implementation and
converts its `HandResult` into the shared geometry contract.
