"""Phase 1 geometry smoke entry point.

The complete challenge application will be integrated later. This entry point
keeps ``python main.py`` useful now by running the deterministic geometry demo.
"""

from tools.geometry_demo import main


if __name__ == "__main__":
    raise SystemExit(main())
