"""Recapture `assets/screenshot-dark.png`, the README hero.

Playwright is deliberately *not* a project dependency — it would drag a browser
download into every `uv sync` for something run a few times a year. Start the
app, then run this through `uvx`:

    uv run streamlit run streamlit_app.py --server.headless true &
    uvx --with playwright python scripts/capture_screenshot.py

This is a script rather than a paragraph of instructions because all nine ways
it goes wrong produce a *plausible but wrong image* instead of an error:

* `embed_options=dark_theme` is a no-op, so dark mode is forced by emulating the
  browser's colour scheme. That is also the honest path: the app follows the
  system setting (only its dark theme is custom), so this is what a dark-mode
  user really sees.
* Chrome's own `--headless --screenshot --virtual-time-budget` captures the
  skeleton loader — virtual time advances timers but never Streamlit's websocket
  handshake. Every wait below is therefore on a real selector.
* The clicked Play button keeps a focus state its five siblings lack and renders
  a brighter red, so it is blurred and the mouse parked before the shot — in the
  main area's top-right margin, not the top-left corner it used to use: that is
  sidebar now, and hovering the sidebar draws its « collapse button.
* `.stMainBlockContainer` and the sidebar's user-content box both overshoot the
  visible content by their bottom padding (the sidebar's by 96px), so the crop
  height is measured off the last real elements.
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
  silently returns a shot cut off at `HEIGHT` rather than erroring. When the
  page was ~1330px tall, one added element came within 20px of that. `crop_height`
  now raises instead, so the failure is loud and `HEIGHT` is the one number to
  raise when the layout legitimately grows.
* That guard only covers the page outgrowing the *viewport*. The crop is taken
  at `CROP_ANCHORS`, so content added *below* an anchor is cropped away just as
  quietly and passes every check above — the anchor is still found and the
  height is still under `HEIGHT`. `crop_height` therefore measures the real
  content bottom (`content_bottom`) of the main area and the sidebar, and
  refuses if anything sits under the crop line, which is also what catches an
  anchor going stale.
* That measurement once read leaf *elements* only, which sees neither a
  paragraph's own text once it wraps past its last inline element nor any box's
  frame, and the sidebar layout exposed both. The narrow sidebar wraps the tips'
  "`[word](+2)` (works best on less-stressed, usually short words)" two lines
  past the code span, so the scan stopped at 595px with text reaching 646px.
  And an expander's border sits well under its last line of text: in a
  since-reverted layout that wrapped the phonemes, the crop without the phoneme
  anchor fell 13px short of that border and the refusal stayed silent. Text
  nodes, and any element that paints a
  bottom border or a background, are now measured too.

`embed=true` drops the Deploy/⋮ toolbar but keeps the sidebar, expanded at its
default 300px; `show_padding` restores the top margin that embed mode strips.
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

# Cropping anchors, one per region: text inside the region's bottom-most block,
# and the testid of that block. The crop is taken at the block's own bottom edge,
# not at the text line: the phoneme block has no fixed text to anchor on, and in
# the narrow sidebar the tips' last paragraph wraps two lines below the words
# that start it, so a line of text no longer marks where a region ends.
CROP_ANCHORS = (
    ("Phoneme tokens", "stExpander"),  # composer: the expanded phoneme block
    # Voice list: the "Show all voices" toggle under the top cards. st.toggle
    # has no testid of its own; it renders as a role="switch" stCheckbox.
    ("Show all voices", "stCheckbox"),
    ("Pronunciation tips", "stVerticalBlock"),  # sidebar: the tips' bordered box
)
CROP_MARGIN = 48

# Where `content_bottom` looks: the main area and the sidebar's user content.
# Not the whole sidebar — its resize handle is a leaf as tall as the viewport.
CONTENT_ROOTS = (".stMainBlockContainer", '[data-testid="stSidebarUserContent"]')

GENERATION_TIMEOUT_MS = 240_000

# Cards drawn above the "Show all voices" toggle — `_split_voices_for_display`'s
# `top_n`. A fresh session starts with the toggle off, so these are all of them.
EXPECTED_VISIBLE_CARDS = 6

# Leaf elements, painted boxes and text nodes, for the reasons `content_bottom`
# gives. Zero-area and hidden nodes are skipped so `<script>`, `<style>` and
# collapsed nodes don't report a bottom; `display:none` is already covered,
# since it zeroes the rect. A missing root is reported rather than skipped, or a
# renamed testid would silently drop the sidebar from the check. The page is
# never scrolled, so viewport coordinates are document coordinates — the same
# frame `bounding_box()` reports the anchors in.
_CONTENT_BOTTOM = """
roots => {
  const missing = roots.filter(sel => !document.querySelector(sel));
  if (missing.length > 0) return {missing};
  const clear = 'rgba(0, 0, 0, 0)';
  const shown = s => s.visibility !== 'hidden' && s.opacity !== '0';
  const painted = s =>
    s.backgroundColor !== clear ||
    (parseFloat(s.borderBottomWidth) > 0 && s.borderBottomStyle !== 'none' &&
      s.borderBottomColor !== clear);
  const range = document.createRange();
  let bottom = 0;
  const reach = r => { bottom = Math.max(bottom, r.bottom); };
  const hasArea = r => r.width > 0 && r.height > 0;
  for (const sel of roots) {
    const root = document.querySelector(sel);
    for (const el of root.querySelectorAll('*')) {
      const r = el.getBoundingClientRect();
      if (!hasArea(r)) continue;
      const s = getComputedStyle(el);
      if (!shown(s)) continue;
      if (el.children.length === 0 || painted(s)) reach(r);
    }
    const texts = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let t = texts.nextNode(); t; t = texts.nextNode()) {
      if (t.data.trim() === '') continue;
      range.selectNodeContents(t);
      const r = range.getBoundingClientRect();
      if (hasArea(r) && shown(getComputedStyle(t.parentElement))) reach(r);
    }
  }
  return {bottom};
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
    # A collapsed sidebar is still laid out, just off-canvas, so every anchor is
    # still found; demand it open rather than let a narrower WIDTH fold it away.
    page.wait_for_selector(
        '[data-testid="stSidebar"][aria-expanded="true"]', timeout=10_000
    )
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

    # Drop the focus ring the click left behind, and park the cursor in the main
    # area's empty top-right margin so nothing renders a hover state. Not the
    # top-left corner: that is inside the sidebar, which draws its collapse
    # button while hovered.
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(WIDTH - 4, 4)
    page.wait_for_timeout(1_200)

    # Re-check after the Play rerun, immediately before the shot is measured.
    wait_for_cards(page)


def content_bottom(page: Page) -> float:
    """The true bottom of rendered content across `CONTENT_ROOTS`.

    Both roots overshoot by their bottom padding, which is why the crop is
    anchored rather than measured off a container. A leaf element has no
    children to pad around, a text node's range is exactly its line boxes, and a
    box that paints a bottom border or a background shows its bottom edge, so
    the lowest of those is the real bottom. Leaves alone are not enough: a
    paragraph whose text wraps past its last inline child is not a leaf, and a
    bordered block's frame sits below every leaf inside it. The roots are not
    their own descendants, so their padding stays out.
    """
    reached = page.evaluate(_CONTENT_BOTTOM, list(CONTENT_ROOTS))
    if "missing" in reached:
        raise RuntimeError(
            f"content root not found, Streamlit changed?: {reached['missing']}"
        )
    return reached["bottom"]


def anchor_bottom(page: Page, text: str, testid: str) -> float:
    """Bottom edge of the nearest `testid` block enclosing `text`."""
    block = page.get_by_text(text).locator(
        f'xpath=ancestor::*[@data-testid="{testid}"][1]'
    )
    # Exactly one, so an ambiguous anchor fails rather than `.first` quietly
    # picking whichever block comes first in the DOM.
    if block.count() != 1:
        raise RuntimeError(
            f"crop anchor {text!r} matched {block.count()} {testid} blocks, UI changed?"
        )
    box = block.bounding_box()
    if box is None:
        raise RuntimeError(f"crop anchor not visible, UI changed?: {text!r}")
    return box["y"] + box["height"]


def crop_height(page: Page) -> int:
    """Height that ends just below the last real element, not the container."""
    bottoms = [anchor_bottom(page, text, testid) for text, testid in CROP_ANCHORS]
    height = round(max(bottoms) + CROP_MARGIN)
    if height > HEIGHT:
        raise RuntimeError(
            f"content is {height}px tall but the viewport is {HEIGHT}px; a "
            "non-full-page screenshot clips to the viewport, so this would have "
            f"shipped a hero cut off at {HEIGHT}px with no error. Raise HEIGHT."
        )
    # The anchors say where the regions *used* to end. Anything rendered below
    # one of them is cropped away just as silently as overflowing the viewport
    # is, and no step above notices — the anchor is still found, the height is
    # still under HEIGHT. So measure the content independently and refuse.
    reached = content_bottom(page)
    if reached > height:
        raise RuntimeError(
            f"content reaches {round(reached)}px but the crop ends at {height}px, "
            f"so {round(reached) - height}px would be cut off with no error. One "
            f"of the crop anchors {CROP_ANCHORS} is no longer the bottom-most "
            "block in its region — repoint it at whatever now is."
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
