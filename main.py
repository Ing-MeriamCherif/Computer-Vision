"""Challenge entry point.

``python main.py`` launches the native live camera-first competition application.
``python main.py --smoke`` runs a deterministic headless smoke test for CI.

All standard CLI arguments are forwarded to ``tools.native_live_app``.

Quick start:
    python main.py                            # launch full native live app
    python main.py --smoke                    # headless 10-frame smoke test
    python main.py --camera 0 --quality high --fullscreen
    python main.py --depth-backend colleague  # use colleague depth model
    python main.py --help                     # all options
"""

from __future__ import annotations

import sys


def main() -> int:
    from tools.native_live_app import main as _native_main
    return _native_main()


if __name__ == "__main__":
    sys.exit(main())
