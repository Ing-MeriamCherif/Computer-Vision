"""Challenge entry point.

``python main.py`` launches the single P123 Material 3 live interface.

All standard CLI arguments are forwarded to ``tools.p123_live_app``.

Quick start:
    python main.py                            # launch the P123 Material 3 UI
    python main.py --camera 0 --mode 7 --lighting-quality high
    python main.py --depth-backend colleague  # use colleague depth model
    python main.py --help                     # all options
"""

from __future__ import annotations

import sys


def main() -> int:
    from tools.p123_live_app import main as _p123_main
    return _p123_main()


if __name__ == "__main__":
    sys.exit(main())
