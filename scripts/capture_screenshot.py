"""Recapture `assets/screenshot-dark.png`, the README hero.

Playwright is deliberately *not* a project dependency — it would drag a browser
download into every `uv sync` for something run a few times a year. Start the
app, then run this through `uvx`:

    uv run streamlit run streamlit_app.py --server.headless true &
    uvx --with playwright python scripts/capture_screenshot.py

This is a script rather than a paragraph of instructions because all eight ways
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
* Waiting on "Show all voices" proves the column exists, not that it is full: a
  capture once shipped with five of the six cards, the second slot simply absent
  and no step failing. `wait_for_cards` pins the count instead.
* Playwright's `text=` engine matches a case-insensitive *substring*, so waiting
  on "Phonemes" after Tokenize resolved instantly against the length caption
  ("275 phonemes — ideal"), which the sample text alone renders. The wait was a
  no-op and only a 1.2s sleep stood between a cold G2P load and a shot with the
  expander empty. Wait on the expander's own label instead.
* `crop_height` measures content, but a non-full-page screenshot cannot clip
  past the viewport: once the page grows taller than `HEIGHT`, Playwright
  silently returns a shot cut off at `HEIGHT` rather than erroring. Adding one
  element to either column was enough to get within 20px of that. `crop_height`
  now raises instead, so the failure is loud and `HEIGHT` is the one number to
  raise when the layout legitimately grows.
* That guard only covers the page outgrowing the *viewport*. The crop is taken
  at `CROP_ANCHORS`, so content added *below* an anchor is cropped away just as
  quietly and passes every check above — the anchor is still found and the
  height is still under `HEIGHT`. `crop_height` therefore measures the real
  content bottom (`content_bottom`) and refuses if anything sits under the crop
  line, which is also what catches an anchor going stale.

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
# The crop ceiling, not just the window: a non-full-page screenshot cannot clip
# past the viewport, so this must stay comfortably above whatever `crop_height`
# measures. Raise it when the layout grows — `crop_height` raises if it doesn't.
HEIGHT = 1800
SCALE = 2

# Cropping anchors: the bottom-most real element in each column.
CROP_ANCHORS = ("Show all voices", "Raise stress:")
CROP_MARGIN = 48

GENERATION_TIMEOUT_MS = 240_000

# Cards drawn outside the expander — `_split_voices_for_display`'s `top_n`.
EXPECTED_VISIBLE_CARDS = 6

# Leaves only, for the reason `content_bottom` gives. Zero-area and hidden
# elements are skipped so `<script>`, `<style>` and collapsed nodes don't
# report a bottom; `display:none` is already covered, since it zeroes the rect.
# The page is never scrolled, so viewport coordinates are document coordinates
# — the same frame `bounding_box()` reports the anchors in.
_CONTENT_BOTTOM = """
() => {
  const root = document.querySelector('.stMainBlockContainer') || document.body;
  let bottom = 0;
  for (const el of root.querySelectorAll('*')) {
    if (el.children.length > 0) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.opacity === '0') continue;
    bottom = Math.max(bottom, r.bottom);
  }
  return bottom;
}
"""

# Every card carries exactly one Play button, so counting them counts cards.
_PLAY_BUTTON_COUNT = (
    "n => [...document.querySelectorAll('button')]"
    ".filter(b => b.innerText.trim().endsWith('Play')).length === n"
)


def wait_for_cards(page: Page) -> None:
    """Block until the voice column is full, not merely present."""
    page.wait_for_function(
        _PLAY_BUTTON_COUNT, arg=EXPECTED_VISIBLE_CARDS, timeout=60_000
    )


def drive(page: Page) -> None:
    """Put the app into the mid-session state the hero is meant to show."""
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Show all voices", timeout=60_000)
    wait_for_cards(page)
    page.wait_for_timeout(1_500)

    page.get_by_role("button", name="Gatsby").click()
    page.wait_for_function(
        "document.querySelector('textarea').value.includes('younger')", timeout=30_000
    )

    page.get_by_role("button", name="Tokenize").click()
    # Not "text=Phonemes": that substring-matches the length caption, which is
    # already on screen. Wait for the expander, then for tokens inside it.
    page.wait_for_selector("text=Phoneme tokens", timeout=60_000)
    page.wait_for_selector('[data-testid="stCode"]', timeout=60_000)
    page.wait_for_timeout(1_200)

    page.get_by_role("button", name="Play").first.click()
    page.wait_for_selector("audio", timeout=GENERATION_TIMEOUT_MS)
    page.wait_for_timeout(3_000)

    # Match the error variants, not `stAlert`: Streamlit suffixes the *content*
    # testid with the severity (stAlertContentInfo / stAlertContentError) and
    # leaves the outer stAlert container severity-agnostic, so `stAlert` would
    # match any non-error alert the page happens to render and abort a perfectly
    # good capture. st.exception renders stException instead.
    failures = page.locator(
        '[data-testid="stAlertContentError"], [data-testid="stException"]'
    ).all_inner_texts()
    if failures:
        raise RuntimeError(
            f"app surfaced an error, refusing to ship the shot: {failures}"
        )

    # Drop the focus ring the click left behind, and park the cursor off-canvas
    # so nothing renders a hover state.
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(4, 4)
    page.wait_for_timeout(1_200)

    # Re-check after the Play rerun, immediately before the shot is measured.
    wait_for_cards(page)


def content_bottom(page: Page) -> float:
    """The true bottom of rendered content, measured off leaf elements.

    `.stMainBlockContainer`'s own box overshoots by its bottom padding, which is
    why the crop is anchored rather than measured off the container. A leaf has
    no children to pad around, so the lowest leaf bottom is the real one.
    """
    return page.evaluate(_CONTENT_BOTTOM)


def crop_height(page: Page) -> int:
    """Height that ends just below the last real element, not the container."""
    bottoms = []
    for anchor in CROP_ANCHORS:
        box = page.get_by_text(anchor).first.bounding_box()
        if box is None:
            raise RuntimeError(f"crop anchor not found, UI changed?: {anchor!r}")
        bottoms.append(box["y"] + box["height"])
    height = round(max(bottoms) + CROP_MARGIN)
    if height > HEIGHT:
        raise RuntimeError(
            f"content is {height}px tall but the viewport is {HEIGHT}px; a "
            "non-full-page screenshot clips to the viewport, so this would have "
            f"shipped a hero cut off at {HEIGHT}px with no error. Raise HEIGHT."
        )
    # The anchors say where the columns *used* to end. Anything rendered below
    # one of them is cropped away just as silently as overflowing the viewport
    # is, and no step above notices — the anchor is still found, the height is
    # still under HEIGHT. So measure the content independently and refuse.
    reached = content_bottom(page)
    if reached > height:
        raise RuntimeError(
            f"content reaches {round(reached)}px but the crop ends at {height}px, "
            f"so {round(reached) - height}px would be cut off with no error. One "
            f"of the crop anchors {CROP_ANCHORS} is no longer the bottom-most "
            "element in its column — repoint it at whatever now is."
        )
    return height


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
