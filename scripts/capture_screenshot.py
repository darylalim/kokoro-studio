"""Recapture `assets/screenshot-dark.png`, the README hero.

Playwright is deliberately *not* a project dependency — it would drag a browser
download into every `uv sync` for something run a few times a year. Start the
app, then run this through `uvx`:

    uv run streamlit run streamlit_app.py --server.headless true &
    uvx --with playwright python scripts/capture_screenshot.py

This is a script rather than a paragraph of instructions because all four ways
it goes wrong produce a *plausible but wrong image* instead of an error:

* `embed_options=dark_theme` is a no-op, so dark mode is forced by emulating the
  browser's colour scheme. That is also the honest path: the app ships no theme
  and follows the system setting, so this is what a dark-mode user really sees.
* Chrome's own `--headless --screenshot --virtual-time-budget` captures the
  skeleton loader — virtual time advances timers but never Streamlit's websocket
  handshake. Every wait below is therefore on a real selector.
* The clicked Play button keeps a focus state its five siblings lack and renders
  a brighter red, so it is blurred and the mouse parked before the shot.
* `.stMainBlockContainer`'s own bounding box overshoots the visible content by
  its bottom padding, so the crop height is measured off the last real elements.

`embed=true` drops the Deploy/⋮ toolbar; `show_padding` restores the top margin
that embed mode strips.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright  # ty: ignore[unresolved-import]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "assets" / "screenshot-dark.png"

URL = "http://localhost:8501/?embed=true&embed_options=show_padding"
WIDTH = 1600
HEIGHT = 1400
SCALE = 2

# Cropping anchors: the bottom-most real element in each column.
CROP_ANCHORS = ("Show all voices", "Raise stress:")
CROP_MARGIN = 48

GENERATION_TIMEOUT_MS = 240_000


def drive(page: Page) -> None:
    """Put the app into the mid-session state the hero is meant to show."""
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Show all voices", timeout=60_000)
    page.wait_for_timeout(1_500)

    page.get_by_role("button", name="Gatsby").click()
    page.wait_for_function(
        "document.querySelector('textarea').value.includes('younger')", timeout=30_000
    )

    page.get_by_role("button", name="Tokenize").click()
    page.wait_for_selector("text=Phonemes", timeout=60_000)
    page.wait_for_timeout(1_200)

    page.get_by_role("button", name="Play").first.click()
    page.wait_for_selector("audio", timeout=GENERATION_TIMEOUT_MS)
    page.wait_for_timeout(3_000)

    alerts = page.locator('[data-testid="stAlert"]').all_inner_texts()
    if alerts:
        raise RuntimeError(f"app raised an alert, refusing to ship the shot: {alerts}")

    # Drop the focus ring the click left behind, and park the cursor off-canvas
    # so nothing renders a hover state.
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(4, 4)
    page.wait_for_timeout(1_200)


def crop_height(page: Page) -> int:
    """Height that ends just below the last real element, not the container."""
    bottoms = []
    for anchor in CROP_ANCHORS:
        box = page.get_by_text(anchor).first.bounding_box()
        if box is None:
            raise RuntimeError(f"crop anchor not found, UI changed?: {anchor!r}")
        bottoms.append(box["y"] + box["height"])
    return round(max(bottoms) + CROP_MARGIN)


def optimise(path: Path) -> None:
    """Shrink in place with pngquant, which is lossless enough on flat UI colour."""
    if shutil.which("pngquant") is None:
        print("pngquant not found, leaving the PNG unoptimised", file=sys.stderr)
        return
    before = path.stat().st_size
    subprocess.run(
        [
            "pngquant",
            "--quality=88-100",
            "--speed",
            "1",
            "--strip",
            "--force",
            "--output",
            str(path),
            str(path),
        ],
        check=True,
    )
    print(f"pngquant: {before // 1024} KB -> {path.stat().st_size // 1024} KB")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=SCALE,
            color_scheme="dark",
        )
        drive(page)
        height = crop_height(page)
        page.screenshot(
            path=str(OUTPUT),
            clip={"x": 0, "y": 0, "width": WIDTH, "height": height},
        )
        browser.close()

    optimise(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} at {WIDTH * SCALE}x{height * SCALE}")


if __name__ == "__main__":
    main()
