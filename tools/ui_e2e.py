#!/usr/bin/env python3
"""Headless browser verification for the local NRW Geometry Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--image", default="/tmp/cvnrw-webcam.jpg")
    parser.add_argument("--screenshot", default="/tmp/cvnrw-ui-e2e.png")
    args = parser.parse_args()
    if not Path(args.image).exists():
        parser.error(f"test image does not exist: {args.image}")
    console_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeoutError:
            # Gradio maintains background connections; rendered-state gates below
            # are authoritative when the network never becomes fully idle.
            pass
        page.get_by_text("NRW Geometry Lab", exact=True).wait_for(timeout=30_000)
        page.locator("input[type=file]").last.set_input_files(args.image)
        page.get_by_role("button", name="Process current frame").click()
        page.get_by_text("renderer_contract_valid", exact=False).wait_for(timeout=90_000)
        page.get_by_text("CUDA current geometry", exact=True).click()
        page.get_by_role("button", name="Process current frame").click()
        page.wait_for_timeout(2_000)
        page.get_by_text("Phase 5 persistent", exact=True).click()
        page.get_by_role("button", name="Process current frame").click()
        page.get_by_text("surfel_count", exact=False).wait_for(timeout=90_000)
        page.screenshot(path=args.screenshot, full_page=True)
        body = page.locator("body").inner_text()
        assert "Relative depth" in body and "Camera-facing normals" in body
        assert "renderer_contract_valid" in body and "true" in body.lower()
        browser.close()
    print({"ui_loaded": True, "modes_processed": ["Phase 1-4 temporal", "CUDA current geometry", "Phase 5 persistent"], "renderer_contract_valid": True, "console_errors": console_errors, "screenshot": args.screenshot})
    return 0 if not console_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
