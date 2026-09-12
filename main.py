"""Challenge entry point.

``python main.py`` launches the live webcam/WebRTC application required by the
challenge.  The deterministic geometry smoke test remains available with
``python main.py --smoke`` for headless CI and numeric validation.
"""

from __future__ import annotations

import sys

from tools.geometry_demo import main
from tools.webcam_geometry_app import main as live_main


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        sys.argv.remove("--smoke")
        raise SystemExit(main())
    raise SystemExit(live_main())
