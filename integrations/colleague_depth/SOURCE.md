# Colleague depth implementation

The `depth/` and `model_comparison/` trees are preserved from
`Ing-MeriamCherif/Computer-Vision` branch `feature/depth` at commit
`f8ecd09d546c2cbb0c9c1e11e9e151591e567fee`.

The upstream files remain source-compatible with their original
`python -m depth` entry point. The live app keeps its tested local-checkpoint
provider as the default; an adapter can opt into this branch without hiding
or rewriting the original implementation.
