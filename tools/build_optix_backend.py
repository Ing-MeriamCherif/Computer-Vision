"""Configure/build the optional native OptiX Mode 7 backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optix-root", default=None, help="NVIDIA OptiX SDK directory (or use OPTIX_ROOT)")
    parser.add_argument("--build-dir", default="build/optix")
    args = parser.parse_args()
    root = args.optix_root or __import__("os").environ.get("OPTIX_ROOT")
    if not root:
        raise SystemExit("Set OPTIX_ROOT or pass --optix-root <OptiX SDK directory>")
    source = Path(__file__).resolve().parents[1] / "native" / "optix_relight"
    build = Path(args.build_dir)
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), f"-DOPTIX_ROOT={root}"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "-j2"], check=True)
    print(f"Built optional OptiX module under {build}; copy _p123_optix into the active .venv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
