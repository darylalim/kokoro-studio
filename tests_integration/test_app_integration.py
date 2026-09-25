import re
from pathlib import Path

from huggingface_hub import snapshot_download
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.proto.LabelVisibility_pb2 import LabelVisibility
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Block, Button, Markdown, Toggle

# The initial run touches the local model snapshot and lazily loads a misaki
# tokenizer when Tokenize is clicked, so allow generous headroom.
DEFAULT_TIMEOUT = 60

# Absolute so the suite is independent of both the working directory and of
# AppTest's own relative-path base, which streamlit 1.61 changed from the
# working directory to the directory of the file calling from_file().
APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def _run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=DEFAULT_TIMEOUT).run()


def _voice_titles(at: AppTest) -> list[str]:
    return [
        m.value
        for m in at.markdown
        if m.value and ("(female)" in m.value or "(male)" in m.value)
    ]


SHOW_ALL_VOICES = "Show all voices"


def _show_all_toggle(at: AppTest) -> Toggle:
    """The "Show all voices" toggle, found by its label as a user finds it.

    Page-wide, like Tokenize in TestSidebar: a misplaced toggle still gets
    clicked, and the placement asserts are what fail. By label rather than by
    key, so a regression that drops the key fails the element-id test on the
    id itself instead of on this lookup.
    """
    (toggle,) = [t for t in at.toggle if t.label.startswith(SHOW_ALL_VOICES)]
    return toggle


def _show_all_voices(at: AppTest, on: bool) -> None:
    """Flip "Show all voices" the way a click does, then rerun.

    `set_value` sends the new value under the element id the last run drew the
    toggle with, as a browser does, and writes nothing into session_state
    behind the widget's back. So if the id moves between runs, the click is
    lost here exactly as it would be in a browser.
    """
    _show_all_toggle(at).set_value(on).run()


def _tail_size(at: AppTest) -> int:
    """How many more voices the toggle's label promises: 14 in "(14 more)"."""
    label = _show_all_toggle(at).label
    match = re.fullmatch(rf"{SHOW_ALL_VOICES} \((\d+) more\)", label)
    assert match, f"unexpected toggle label {label!r}"
    return int(match[1])


def _voices_on_disk(prefix: str) -> int:
    """Voice files in the local snapshot whose id starts with `prefix`.

    A count the app cannot influence: the toggle's label and its tail both come
    from the app's own `hidden` list, so checking one against the other passes
    even when that list silently drops voices.
    """
    snapshot = Path(
        snapshot_download("mlx-community/Kokoro-82M-bf16", local_files_only=True)
    )
    return len(list((snapshot / "voices").glob(f"{prefix}*.safetensors")))


def _block_shape(node: object) -> object:
    """A card's nesting of blocks and element types, blind to its content."""
    if not isinstance(node, Block):
        return type(node).__name__
    children = tuple(_block_shape(node.children[i]) for i in sorted(node.children))
    return (node.proto.SerializeToString(deterministic=True), children)


def _speed_keys(root: AppTest | Block) -> set[str]:
    """Keys of the per-card speed selectboxes this run actually drew in `root`."""
    return {s.key for s in root.selectbox if s.key and s.key.startswith("speed_")}


TIPS_SYNTAX = "[word](/phonemes/)"
TIPS_HEADING = "Pronunciation tips"


def _markdown_containing(root: Block, needle: str) -> list[str]:
    return [m.value for m in root.markdown if needle in (m.value or "")]


# Common to the estimate ("**~9 phonemes** — …") and the exact count, and absent
# from every other caption — the orientation line has an em dash but no count.
LENGTH_CAPTION = " phonemes** — "


def _length_captions(root: Block) -> list[str]:
    return [c.value for c in root.caption if LENGTH_CAPTION in (c.value or "")]


def _composer_and_card_buttons(root: Block) -> list[str]:
    """Labels of the buttons that belong in main: samples, Tokenize, every Play."""
    return [
        b.label
        for b in root.button
        if b.label in ("Tokenize", "Play") or (b.key or "").startswith("sample_")
    ]


class TestInitialRender:
    def test_has_no_exception(self) -> None:
        at = _run_app()
        assert not at.exception

    def test_title_and_language_default(self) -> None:
        at = _run_app()
        assert at.title[0].value == "Kokoro Studio"
        assert at.selectbox(key="language").value == "American English"

    def test_gender_filter_defaults_to_all(self) -> None:
        at = _run_app()
        assert at.segmented_control(key="gender").value == "All"

    def test_tokenize_disabled_when_text_empty(self) -> None:
        at = _run_app()
        tokenize = next(b for b in at.button if b.label == "Tokenize")
        assert tokenize.disabled is True

    def test_play_buttons_disabled_when_text_empty(self) -> None:
        at = _run_app()
        play_buttons = [b for b in at.button if b.label == "Play"]
        assert play_buttons
        assert all(b.disabled for b in play_buttons)


class TestSidebar:
    """The sidebar holds the settings and the reference note; main holds the work.

    Language and gender decide which voices main shows, and the tips are
    consulted rather than read in sequence, so those three live in the sidebar.
    The composer and the voice cards are the content and stay in main — on a
    phone the sidebar starts hidden. Placement is read from `at.sidebar` and
    `at.main`, never from the page as a whole: `at.selectbox(key=...)` searches
    both roots and so passes wherever the widget lands.
    """

    def test_language_and_gender_filters_render_in_the_sidebar(self) -> None:
        at = _run_app()
        assert [s.key for s in at.sidebar.selectbox] == ["language"]
        assert "language" not in [s.key for s in at.main.selectbox]
        assert [s.key for s in at.sidebar.segmented_control] == ["gender"]
        assert not at.main.segmented_control

    def test_filter_labels_are_visible(self) -> None:
        # The gender control used to sit label-collapsed directly above the cards
        # it filters. In the sidebar it is away from them, so it names itself.
        at = _run_app()
        language = at.sidebar.selectbox(key="language")
        gender = at.sidebar.segmented_control(key="gender")
        assert language.label == "Language"
        assert gender.label == "Voice gender"
        visible = LabelVisibility.LabelVisibilityOptions.VISIBLE
        assert language.proto.label_visibility.value == visible
        assert gender.proto.label_visibility.value == visible

    def test_pronunciation_tips_render_once_in_the_sidebar(self) -> None:
        at = _run_app()
        assert len(_markdown_containing(at.sidebar, TIPS_SYNTAX)) == 1
        assert len(_markdown_containing(at.sidebar, TIPS_HEADING)) == 1
        assert not _markdown_containing(at.main, TIPS_SYNTAX)
        assert not _markdown_containing(at.main, TIPS_HEADING)
        # Heading directly above the body in one bordered box — the grouping
        # that stands in for st.info without its live region.
        boxes = [
            node
            for node in at.sidebar
            if isinstance(node, Block)
            and any(
                isinstance(c, Markdown) and TIPS_SYNTAX in c.value
                for c in node.children.values()
            )
        ]
        assert len(boxes) == 1
        (box,) = boxes
        assert box.proto.flex_container.border
        heading, body = list(box.children.values())[:2]
        assert isinstance(heading, Markdown) and TIPS_HEADING in heading.value
        assert isinstance(body, Markdown) and TIPS_SYNTAX in body.value

    def test_composer_and_voice_cards_render_in_main(self) -> None:
        at = _run_app()
        assert [t.key for t in at.main.text_area] == ["text_input"]
        assert not at.sidebar.text_area
        assert not _composer_and_card_buttons(at.sidebar)
        labels = _composer_and_card_buttons(at.main)
        assert labels.count("Tokenize") == 1
        assert labels.count("Play") == 6
        samples = [b for b in at.main.button if (b.key or "").startswith("sample_")]
        assert len(samples) == 3
        # The tail cards too, and the toggle that reveals them: both belong to
        # main's voice column.
        assert [t.label for t in at.main.toggle] == [_show_all_toggle(at).label]
        assert not at.sidebar.toggle
        _show_all_voices(at, True)
        assert not _composer_and_card_buttons(at.sidebar)
        plays = [b for b in at.button if b.label == "Play"]
        assert len(plays) > 6
        assert _composer_and_card_buttons(at.main).count("Play") == len(plays)

    def test_tokenize_row_and_phoneme_readout_render_in_main(self) -> None:
        # Drawn by helpers called under `input_col`, so the text area staying in
        # main says nothing about them. Tokenize is found page-wide on purpose:
        # a misplaced button still gets clicked, and the asserts below fail.
        at = _run_app()
        at.text_area(key="text_input").input("hello world").run()
        next(b for b in at.button if b.label == "Tokenize").click().run()
        assert not at.exception
        phonemes = at.session_state["last_phonemes"][2]
        assert phonemes
        assert "Tokenize" not in [b.label for b in at.sidebar.button]
        assert not _length_captions(at.sidebar)
        assert "Phoneme tokens" not in [e.label for e in at.sidebar.expander]
        assert phonemes not in [c.value for c in at.sidebar.code]
        # One horizontal row in main holds the button and the caption, and the
        # caption is this run's exact count, not the "~" estimate: its slot is
        # filled only after the click has tokenized.
        (row,) = [
            node
            for node in at.main
            if isinstance(node, Block)
            and any(
                isinstance(c, Button) and c.label == "Tokenize"
                for c in node.children.values()
            )
        ]
        horizontal = BlockProto.FlexContainer.Direction.HORIZONTAL
        assert row.proto.flex_container.direction == horizontal
        assert _length_captions(at.main) == _length_captions(row)
        (caption,) = _length_captions(row)
        assert f"**{len(phonemes)} phonemes**" in caption
        (tokens,) = [e for e in at.main.expander if e.label == "Phoneme tokens"]
        assert [c.value for c in tokens.code] == [phonemes]


class TestPronunciationNote:
    def test_note_is_not_an_alert(self) -> None:
        # st.info renders role="status" -- an ARIA live region -- which asks a
        # screen reader to announce permanent reference text as a status update.
        at = _run_app()
        assert not at.info

    def test_note_renders_once_with_the_pronunciation_syntax(self) -> None:
        at = _run_app()
        bodies = [
            m.value for m in at.markdown if "[word](/phonemes/)" in (m.value or "")
        ]
        assert len(bodies) == 1


class TestSampleButtons:
    def test_clicking_sample_populates_text_area(self) -> None:
        at = _run_app()
        assert at.text_area[0].value == ""
        at.button(key="sample_a_gatsby").click().run()
        text = at.session_state["text_input"]
        assert text
        assert len(text) > 50


class TestTextInputEnablesActions:
    def test_typing_text_enables_play_buttons(self) -> None:
        at = _run_app()
        at.text_area[0].input("hello world").run()
        play_buttons = [b for b in at.button if b.label == "Play"]
        assert play_buttons
        assert not any(b.disabled for b in play_buttons)


class TestTokenizeFlow:
    def test_clicking_tokenize_renders_phoneme_expander(self) -> None:
        at = _run_app()
        at.text_area[0].input("hello world").run()
        tokenize = next(b for b in at.button if b.label == "Tokenize")
        tokenize.click().run()
        text, lang, phonemes = at.session_state["last_phonemes"]
        assert text == "hello world"
        assert lang == "a"
        assert phonemes
        expander_labels = [e.label for e in at.expander]
        assert "Phoneme tokens" in expander_labels
        assert phonemes in [c.value for c in at.code]


class TestGenderFilter:
    def test_female_filter_shows_only_female_voices(self) -> None:
        at = _run_app()
        at.segmented_control(key="gender").set_value("Female").run()
        titles = _voice_titles(at)
        assert titles
        assert all("(female)" in t for t in titles)

    def test_male_filter_shows_only_male_voices(self) -> None:
        at = _run_app()
        at.segmented_control(key="gender").set_value("Male").run()
        titles = _voice_titles(at)
        assert titles
        assert all("(male)" in t for t in titles)


class TestLanguageSwitching:
    def test_switching_to_japanese_changes_voice_list(self) -> None:
        # Driven through the sidebar, where the Language selectbox lives.
        at = _run_app()
        american_titles = set(_voice_titles(at))
        at.sidebar.selectbox(key="language").select("Japanese").run()
        japanese_titles = set(_voice_titles(at))
        assert japanese_titles
        assert japanese_titles != american_titles


class TestVoiceCards:
    def test_each_visible_voice_has_speed_selectbox_default_1x(self) -> None:
        at = _run_app()
        # Top-grade American English voices, all inside the visible top 6.
        # (am_adam is a tail voice, drawn only once "Show all voices" is on.)
        for voice in ("af_heart", "af_bella", "af_nicole"):
            assert at.selectbox(key=f"speed_{voice}").value == 1.0

    def test_each_speed_selectbox_is_labelled_for_its_voice(self) -> None:
        # The label is collapsed on screen but is the accessible name, so every
        # card's must differ; a shared "Speed" gave twenty identical controls.
        at = _run_app()
        labels = {
            s.key: s.label for s in at.selectbox if s.key and s.key.startswith("speed_")
        }
        assert labels["speed_af_heart"] == "Speed for Heart (female)\u00a0—\u00a0A"
        assert len(set(labels.values())) == len(labels) == 6

    def test_changing_card_speed_reruns_cleanly(self) -> None:
        # The per-card speed selectbox must rerun the app cleanly and reflect the
        # new value. (AppTest can't observe fragment-scoped reruns, so this
        # asserts a healthy rerun, not the isolation itself.)
        at = _run_app()
        at.selectbox(key="speed_af_heart").select(1.5).run()
        assert not at.exception
        assert at.selectbox(key="speed_af_heart").value == 1.5

    def test_tail_voice_speed_survives_collapse_and_reopen(self) -> None:
        # Streamlit discards the state of any widget a run did not draw, so
        # turning "Show all voices" off drops `speed_{voice}` for every tail
        # card. Without persist_state="session" the choice snapped back to 1.0x
        # when the tail came back — and because _cache_key includes the speed,
        # that also demoted the voice's generated clip to a "speed changed"
        # stale preview.
        at = _run_app()
        visible = _speed_keys(at)
        _show_all_voices(at, True)
        tail = sorted(_speed_keys(at) - visible)
        assert tail, "expected voices behind the toggle"
        voice_key = tail[0]

        at.selectbox(key=voice_key).select(1.3).run()
        assert at.selectbox(key=voice_key).value == 1.3

        _show_all_voices(at, False)
        assert voice_key not in _speed_keys(at)
        _show_all_voices(at, True)
        assert at.selectbox(key=voice_key).value == 1.3

    def test_play_buttons_enabled_after_typing(self) -> None:
        at = _run_app()
        at.text_area[0].input("hello world").run()
        play_buttons = [b for b in at.button if b.label == "Play"]
        assert len(play_buttons) >= 3  # one per visible American English voice
        assert not any(b.disabled for b in play_buttons)


class TestShowAllVoices:
    """The voices past the top six sit behind a toggle drawn into the list.

    It is a keyed st.toggle between the top six and the rest, inside the same
    container, so every card is laid out alike. Keyed, its element id comes from
    the key alone, so the tail size in its label can change without remounting
    it; persisted, it stays on through a language or gender filter that leaves
    too few voices to draw it. It is found by label and flipped with
    `set_value`, as a user would.
    """

    def test_off_by_default_and_the_tail_is_not_built(self) -> None:
        # Lazy: the tail's cards — 14 for American English, each a session_state
        # scan and two widgets — are built only while the toggle is on, rather
        # than on every rerun of the page.
        at = _run_app()
        assert _show_all_toggle(at).value is False
        assert _tail_size(at) > 0
        assert len(_speed_keys(at)) == 6
        assert len(_voice_titles(at)) == 6
        assert len([b for b in at.button if b.label == "Play"]) == 6

    def test_turning_it_on_reveals_every_remaining_voice(self) -> None:
        # As many new cards as the label promised, the top six still there, and
        # every American English voice on disk accounted for.
        at = _run_app()
        top = _speed_keys(at)
        promised = _tail_size(at)
        _show_all_voices(at, True)
        assert not at.exception
        shown = _speed_keys(at)
        assert top < shown
        assert len(shown - top) == promised
        assert len(shown) == _voices_on_disk("a")
        assert len(_voice_titles(at)) == len(shown)

    def test_every_click_flips_it(self) -> None:
        # Several clicks in a row, each sent under the id the previous run drew.
        # A toggle whose id moved from run to run would start the next run from
        # its default instead, and swallow the click.
        at = _run_app()
        for on in (True, False, True, False, True):
            _show_all_voices(at, on)
            assert _show_all_toggle(at).value is on
            assert (len(_speed_keys(at)) > 6) is on

    def test_its_element_id_survives_clicks_and_a_count_change(self) -> None:
        # A new element id remounts the widget in the browser: keyboard focus
        # falls to the page, and a click sent under the old id before the new
        # one arrives is lost. The label carries the tail size, which the gender
        # filter changes, so the id must not hash the label. A keyed toggle's id
        # comes from its key alone; unkeyed, or keyed by its label, the id would
        # follow the count.
        at = _run_app()
        seen = [_show_all_toggle(at)]
        for on in (True, False, True):
            _show_all_voices(at, on)
            seen.append(_show_all_toggle(at))
        at.segmented_control(key="gender").set_value("Male").run()
        seen.append(_show_all_toggle(at))
        assert len({t.label for t in seen}) == 2, "the filter should move the count"
        assert len({t.id for t in seen}) == 1

    def test_stays_on_across_a_filter_change(self) -> None:
        at = _run_app()
        _show_all_voices(at, True)
        before = _show_all_toggle(at).label
        at.segmented_control(key="gender").set_value("Male").run()
        toggle = _show_all_toggle(at)
        assert toggle.label != before
        assert toggle.value is True
        titles = _voice_titles(at)
        assert all("(male)" in t for t in titles)
        assert len(titles) == 6 + _tail_size(at) == _voices_on_disk("am")

    def test_stays_on_across_a_filter_that_never_draws_it(self) -> None:
        # A gender filter hides the toggle the same way a small language does:
        # British English has eight voices, so its toggle offers two more, but
        # Female leaves four, and a run that does not draw the toggle would
        # discard its state without persist_state="session".
        at = _run_app()
        at.sidebar.selectbox(key="language").select("British English").run()
        _show_all_voices(at, True)
        opened = _speed_keys(at)
        assert len(opened) == _voices_on_disk("b") > 6

        at.sidebar.segmented_control(key="gender").set_value("Female").run()
        assert not at.toggle

        at.sidebar.segmented_control(key="gender").set_value("All").run()
        assert _show_all_toggle(at).value is True
        assert _speed_keys(at) == opened

    def test_stays_on_across_a_language_that_never_draws_it(self) -> None:
        # Japanese has five voices: no tail, so no toggle, and Streamlit discards
        # the state of any widget a run does not draw. Without
        # persist_state="session", American English came back with its tail
        # hidden.
        at = _run_app()
        _show_all_voices(at, True)
        opened = _speed_keys(at)

        at.sidebar.selectbox(key="language").select("Japanese").run()
        assert not at.toggle
        assert 0 < len(_speed_keys(at)) <= 6

        at.sidebar.selectbox(key="language").select("American English").run()
        assert _show_all_toggle(at).value is True
        assert _speed_keys(at) == opened

    def test_tail_cards_share_one_container_with_the_top_six(self) -> None:
        # The structural reason every card measures the same width: the tail is
        # drawn straight into the container that holds the top six, with the
        # toggle between them. Anything wrapped around the tail gives it a block
        # of its own — an expander's border and padding made each tail card
        # 34 px narrower than the six above it.
        at = _run_app()
        _show_all_voices(at, True)
        (voice_list,) = [
            node
            for node in at.main
            if isinstance(node, Block)
            and any(
                isinstance(c, Toggle) and c.label.startswith(SHOW_ALL_VOICES)
                for c in node.children.values()
            )
        ]
        rows = [voice_list.children[i] for i in sorted(voice_list.children)]
        # One row per card, each holding exactly one speed selectbox; a wrapper
        # around the tail would be a single row holding all of them.
        shape = [
            len(_speed_keys(row)) if isinstance(row, Block) else type(row).__name__
            for row in rows
        ]
        assert shape == [1] * 6 + ["Toggle"] + [1] * _tail_size(at)
        assert _speed_keys(voice_list) == _speed_keys(at)
        # And each card is built alike, top six and tail: a bordered wrapper
        # around every tail card would keep the one-selectbox-per-row shape above
        # while bringing back the narrower, inset tail.
        cards = [row for row in rows if isinstance(row, Block)]
        assert len({_block_shape(card) for card in cards}) == 1
