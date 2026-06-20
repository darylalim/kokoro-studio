from streamlit.testing.v1 import AppTest

# The initial run touches the local model snapshot and lazily loads a misaki
# tokenizer when Tokenize is clicked, so allow generous headroom.
DEFAULT_TIMEOUT = 60


def _run_app() -> AppTest:
    return AppTest.from_file("streamlit_app.py", default_timeout=DEFAULT_TIMEOUT).run()


def _voice_titles(at: AppTest) -> list[str]:
    return [
        m.value
        for m in at.markdown
        if m.value and ("(female)" in m.value or "(male)" in m.value)
    ]


class TestInitialRender:
    def test_has_no_exception(self) -> None:
        at = _run_app()
        assert not at.exception

    def test_title_and_language_default(self) -> None:
        at = _run_app()
        assert at.title[0].value == "Kokoro Studio"
        assert at.selectbox(key="language").value == "American English"

    def test_gender_checkboxes_start_unchecked(self) -> None:
        at = _run_app()
        assert at.checkbox(key="female").value is False
        assert at.checkbox(key="male").value is False

    def test_tokenize_disabled_when_text_empty(self) -> None:
        at = _run_app()
        tokenize = next(b for b in at.button if b.label == "Tokenize")
        assert tokenize.disabled is True

    def test_play_buttons_disabled_when_text_empty(self) -> None:
        at = _run_app()
        play_buttons = [b for b in at.button if b.label == "▶ Play"]
        assert play_buttons
        assert all(b.disabled for b in play_buttons)


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
        play_buttons = [b for b in at.button if b.label == "▶ Play"]
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
        assert "Phoneme Tokens" in expander_labels
        assert phonemes in [c.value for c in at.code]


class TestGenderFilter:
    def test_female_filter_shows_only_female_voices(self) -> None:
        at = _run_app()
        at.checkbox(key="female").check().run()
        titles = _voice_titles(at)
        assert titles
        assert all("(female)" in t for t in titles)

    def test_male_filter_shows_only_male_voices(self) -> None:
        at = _run_app()
        at.checkbox(key="male").check().run()
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
        # American English voices, best grade first.
        for voice in ("af_heart", "af_bella", "am_adam"):
            assert at.selectbox(key=f"speed_{voice}").value == 1.0

    def test_changing_card_speed_reruns_cleanly(self) -> None:
        # The per-card speed selectbox must rerun the app cleanly and reflect the
        # new value. (AppTest can't observe fragment-scoped reruns, so this
        # asserts a healthy rerun, not the isolation itself.)
        at = _run_app()
        at.selectbox(key="speed_af_heart").select(1.5).run()
        assert not at.exception
        assert at.selectbox(key="speed_af_heart").value == 1.5

    def test_play_buttons_enabled_after_typing(self) -> None:
        at = _run_app()
        at.text_area[0].input("hello world").run()
        play_buttons = [b for b in at.button if b.label == "▶ Play"]
        assert len(play_buttons) >= 3  # one per visible American English voice
        assert not any(b.disabled for b in play_buttons)
