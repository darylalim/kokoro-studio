from pathlib import Path

from streamlit.testing.v1 import AppTest

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


def _set_expander(at: AppTest, is_open: bool) -> None:
    """Open or close "Show all voices" the way a real click would.

    A click sends only the widget value, never the app's `_show_all_voices_pref`
    mirror, so only the widget key is set here. Writing the mirror as well is
    what hid a lag in it that swallowed every second click in a real browser:
    left to the app, `test_tail_voice_speed_survives_collapse_and_reopen` fails
    against the lagging version.
    """
    at.session_state["show_all_voices"] = is_open


def _speed_keys(at: AppTest) -> set[str]:
    """Keys of the per-card speed selectboxes this run actually drew."""
    return {s.key for s in at.selectbox if s.key and s.key.startswith("speed_")}


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
        at = _run_app()
        american_titles = set(_voice_titles(at))
        at.selectbox(key="language").select("Japanese").run()
        japanese_titles = set(_voice_titles(at))
        assert japanese_titles
        assert japanese_titles != american_titles


class TestVoiceCards:
    def test_each_visible_voice_has_speed_selectbox_default_1x(self) -> None:
        at = _run_app()
        # Top-grade American English voices, all inside the visible top 6.
        # (am_adam used to pass here only because the expander rendered its whole
        # body while collapsed; it is a tail voice and is no longer drawn.)
        for voice in ("af_heart", "af_bella", "af_nicole"):
            assert at.selectbox(key=f"speed_{voice}").value == 1.0

    def test_changing_card_speed_reruns_cleanly(self) -> None:
        # The per-card speed selectbox must rerun the app cleanly and reflect the
        # new value. (AppTest can't observe fragment-scoped reruns, so this
        # asserts a healthy rerun, not the isolation itself.)
        at = _run_app()
        at.selectbox(key="speed_af_heart").select(1.5).run()
        assert not at.exception
        assert at.selectbox(key="speed_af_heart").value == 1.5

    def test_collapsed_expander_does_not_build_the_tail_voice_cards(self) -> None:
        # "Show all voices" is lazy (on_change="rerun" + .open). Left at the
        # default an expander computes its body while collapsed, which for
        # American English's 20 voices meant 14 surplus cards — each a
        # session_state scan and three widgets — on every rerun.
        at = _run_app()
        # Setting on_change makes the expander a widget rather than a plain
        # block, so it lands in session_state and NOT in at.expander. Reading it
        # here is what pins the lazy behaviour: no key means no `.open` gating.
        assert at.session_state["show_all_voices"] is False
        assert len(_voice_titles(at)) == 6
        assert len([b for b in at.button if b.label == "Play"]) == 6

    def test_opening_the_expander_reveals_the_remaining_voices(self) -> None:
        at = _run_app()
        _set_expander(at, True)
        at.run()
        assert not at.exception
        assert len(_voice_titles(at)) > 6

    def test_every_click_toggles_the_expander(self) -> None:
        # The expander's element id hashes `expanded`. Fed a mirror one run
        # behind, the id changed on every second click and that click was lost.
        at = _run_app()
        for is_open in (True, False, True, False):
            _set_expander(at, is_open)
            at.run()
            plays = len([b for b in at.button if b.label == "Play"])
            assert (plays > 6) is is_open

    def test_a_click_keeps_the_expanders_element_id(self) -> None:
        # A new element id remounts the expander in the browser: keyboard focus
        # fell to the page after every Enter, and a click sent under the old id
        # before the new one arrived (a double click within ~50 ms) was dropped.
        # Only a label change or a restore may re-seed it.
        at = _run_app()
        ids = [s.proto.id for s in at.status if "Show all voices" in s.label]
        for is_open in (True, False, True):
            _set_expander(at, is_open)
            at.run()
            ids += [s.proto.id for s in at.status if "Show all voices" in s.label]
        assert len(ids) == 4
        assert len(set(ids)) == 1

    def test_a_filter_change_reseeds_the_expander(self) -> None:
        # The label carries the tail size, so a filter change moves the element
        # id, and a new id starts from `expanded`: without the re-seed the
        # expander shut on every filter change. A browser resends the expander's
        # value with every rerun, so restate it here; left out, AppTest drops the
        # key and the collected-state re-seed hides a missing label re-seed.
        at = _run_app()
        _set_expander(at, True)
        at.run()
        _set_expander(at, True)
        at.segmented_control(key="gender").set_value("Male").run()
        assert len([b for b in at.button if b.label == "Play"]) == 9

    def test_expander_stays_open_across_a_language_with_no_hidden_voices(self) -> None:
        # Keying the expander turned its open state into ordinary widget state,
        # which Streamlit collects on any run that does not draw it. Japanese has
        # five voices, so `hidden` is empty, the expander is never created, and
        # `show_all_voices` disappears — English then came back collapsed. The
        # plain `_show_all_voices_pref` mirror is what survives that round trip.
        at = _run_app()
        _set_expander(at, True)
        at.run()
        opened = len([b for b in at.button if b.label == "Play"])
        assert opened > 6

        at.selectbox(key="language").select("Japanese").run()
        assert "show_all_voices" not in at.session_state

        at.selectbox(key="language").select("American English").run()
        assert len([b for b in at.button if b.label == "Play"]) == opened

    def test_tail_voice_speed_survives_collapse_and_reopen(self) -> None:
        # Streamlit discards the state of any widget a run did not draw, so
        # collapsing the expander drops `speed_{voice}` for every tail card.
        # Without persist_state="session" the choice snapped back to 1.0x on
        # reopen — and because _cache_key includes the speed, that also demoted
        # the voice's generated clip to a "speed changed" stale preview.
        at = _run_app()
        visible = _speed_keys(at)
        _set_expander(at, True)
        at.run()
        tail = sorted(_speed_keys(at) - visible)
        assert tail, "expected voices behind the expander"
        voice_key = tail[0]

        _set_expander(at, True)
        at.selectbox(key=voice_key).select(1.3).run()
        assert at.selectbox(key=voice_key).value == 1.3

        _set_expander(at, False)
        at.run()
        _set_expander(at, True)
        at.run()
        assert at.selectbox(key=voice_key).value == 1.3

    def test_play_buttons_enabled_after_typing(self) -> None:
        at = _run_app()
        at.text_area[0].input("hello world").run()
        play_buttons = [b for b in at.button if b.label == "Play"]
        assert len(play_buttons) >= 3  # one per visible American English voice
        assert not any(b.disabled for b in play_buttons)
