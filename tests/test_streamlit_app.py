from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import streamlit as st

from streamlit_app import (
    AUDIO_CACHE_LIMIT,
    DEFAULT_SPEED_INDEX,
    ESPEAK_LANGUAGES,
    LANGUAGES,
    PRONUNCIATION_TIPS,
    REPO_ID,
    SAMPLE_RATE,
    SPEED_OPTIONS,
    _audio_to_wav_bytes,
    _cache_key,
    _create_g2p,
    _evict_old_audio,
    _filter_voices_by_gender,
    _find_stale_cached_audio,
    _format_voice,
    _gender_code_from_checkboxes,
    _render_persistent_phonemes,
    _split_voices_for_display,
    ensure_repo_downloaded,
    generate_one,
    generate_speech,
    get_voices,
    load_pipeline,
    load_tokenizer,
    render_phonemes,
    render_voice_card,
    tokenize_text,
)
from voice_grades import _GRADE_RANK, VOICE_GRADES, _grade_rank

EXPECTED_LANGUAGES = [
    "American English",
    "Brazilian Portuguese",
    "British English",
    "French",
    "Hindi",
    "Italian",
    "Japanese",
    "Mandarin Chinese",
    "Spanish",
]

EXPECTED_CODES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}


class TestLanguages:
    def test_all_languages_present(self) -> None:
        assert sorted(LANGUAGES.keys()) == EXPECTED_LANGUAGES

    def test_language_codes(self) -> None:
        codes = set(LANGUAGES.values())
        assert codes == EXPECTED_CODES

    def test_language_count(self) -> None:
        assert len(LANGUAGES) == 9


class TestModelConstants:
    def test_sample_rate(self) -> None:
        assert SAMPLE_RATE == 24000

    def test_repo_id(self) -> None:
        assert REPO_ID == "mlx-community/Kokoro-82M-bf16"


class TestEspeakLanguages:
    def test_has_all_espeak_language_codes(self) -> None:
        assert set(ESPEAK_LANGUAGES.keys()) == {"e", "f", "h", "i", "p"}

    def test_maps_to_correct_espeak_codes(self) -> None:
        assert ESPEAK_LANGUAGES["e"] == "es"
        assert ESPEAK_LANGUAGES["f"] == "fr-fr"
        assert ESPEAK_LANGUAGES["h"] == "hi"
        assert ESPEAK_LANGUAGES["i"] == "it"
        assert ESPEAK_LANGUAGES["p"] == "pt-br"

    def test_covers_non_english_non_ja_non_zh_languages(self) -> None:
        en_ja_zh = {"a", "b", "j", "z"}
        espeak_codes = set(LANGUAGES.values()) - en_ja_zh
        assert set(ESPEAK_LANGUAGES.keys()) == espeak_codes


class TestGetVoices:
    def test_returns_voices_for_language(self) -> None:
        voices = get_voices("a")
        assert len(voices) > 0
        assert all(v[0] == "a" for v in voices)

    def test_returns_empty_for_unknown_language(self) -> None:
        voices = get_voices("x")
        assert voices == []

    def test_sorted_by_quality_grade(self) -> None:
        # af_heart (A) → af_bella (A-) → am_adam (F+)
        voices = get_voices("a")
        assert voices == ["af_heart", "af_bella", "am_adam"]


class TestLoadPipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = load_pipeline()
        assert pipeline is not None

    def test_called_with_repo_id(self) -> None:
        from mlx_audio.tts.utils import load_model

        load_pipeline()
        load_model.assert_called_with(REPO_ID)  # type: ignore[union-attribute]


class TestCreateG2p:
    def test_american_english_uses_en_g2p(self) -> None:
        from misaki import en

        _create_g2p("a")
        en.G2P.assert_called()  # type: ignore[union-attribute]
        call_kwargs = en.G2P.call_args[1]  # type: ignore[union-attribute]
        assert call_kwargs["british"] is False

    def test_british_english_uses_en_g2p_with_british(self) -> None:
        from misaki import en

        _create_g2p("b")
        call_kwargs = en.G2P.call_args[1]  # type: ignore[union-attribute]
        assert call_kwargs["british"] is True

    def test_japanese_uses_ja_g2p(self) -> None:
        from misaki import ja

        _create_g2p("j")
        ja.JAG2P.assert_called()  # type: ignore[union-attribute]

    def test_chinese_uses_zh_g2p(self) -> None:
        from misaki import zh

        _create_g2p("z")
        zh.ZHG2P.assert_called()  # type: ignore[union-attribute]

    def test_espeak_languages_use_espeak_g2p(self) -> None:
        from misaki import espeak

        for code, espeak_lang in ESPEAK_LANGUAGES.items():
            espeak.EspeakG2P.reset_mock()  # type: ignore[union-attribute]
            _create_g2p(code)
            espeak.EspeakG2P.assert_called_with(language=espeak_lang)  # type: ignore[union-attribute]


class TestLoadTokenizer:
    def test_returns_g2p_object(self) -> None:
        result = load_tokenizer("a")
        assert result is not None

    def test_returns_callable(self) -> None:
        result = load_tokenizer("a")
        assert callable(result)


class TestTokenizeText:
    def _mock_g2p(self, phonemes: str) -> MagicMock:
        from misaki import en

        mock_g2p = MagicMock(return_value=(phonemes, None))
        en.G2P.return_value = mock_g2p  # type: ignore[union-attribute]
        return mock_g2p

    def test_returns_phonemes(self) -> None:
        self._mock_g2p("hɛlˈoʊ wˈɜːld")

        result = tokenize_text("hello world", "a")

        assert result == "hɛlˈoʊ wˈɜːld"

    def test_single_word(self) -> None:
        self._mock_g2p("hɛlˈoʊ")

        result = tokenize_text("hello", "a")

        assert result == "hɛlˈoʊ"

    def test_returns_empty_for_empty_phonemes(self) -> None:
        self._mock_g2p("")

        result = tokenize_text("", "a")

        assert result == ""

    def test_returns_empty_for_none_phonemes(self) -> None:
        from misaki import en

        en.G2P.return_value = MagicMock(return_value=(None, None))  # type: ignore[union-attribute]

        result = tokenize_text("", "a")

        assert result == ""

    def test_british_english_uses_british_g2p(self) -> None:
        self._mock_g2p("hɛlˈəʊ")

        tokenize_text("hello", "b")

        from misaki import en

        call_kwargs = en.G2P.call_args[1]  # type: ignore[union-attribute]
        assert call_kwargs["british"] is True

    def test_japanese_uses_ja_g2p(self) -> None:
        from misaki import ja

        ja.JAG2P.reset_mock()  # type: ignore[union-attribute]
        ja.JAG2P.return_value = MagicMock(return_value=("konniʧiwa", None))  # type: ignore[union-attribute]

        result = tokenize_text("こんにちは", "j")

        assert result == "konniʧiwa"
        ja.JAG2P.assert_called_once()  # type: ignore[union-attribute]

    def test_spanish_uses_espeak_g2p(self) -> None:
        from misaki import espeak

        espeak.EspeakG2P.return_value = MagicMock(return_value=("ola", None))  # type: ignore[union-attribute]

        result = tokenize_text("hola", "e")

        assert result == "ola"
        espeak.EspeakG2P.assert_called_with(language="es")  # type: ignore[union-attribute]


class TestGenerateSpeech:
    def _mock_model(self, *, audio_length: int = 48000) -> MagicMock:
        model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.random.randn(audio_length).astype(np.float32)
        model.generate.return_value = [chunk]
        return model

    def test_yields_audio(self) -> None:
        model = self._mock_model()

        results = list(generate_speech("hello", "af_heart", model, lang_code="a"))

        assert len(results) == 1
        assert isinstance(results[0], np.ndarray)
        assert results[0].shape == (48000,)

    def test_calls_model_generate_with_correct_args(self) -> None:
        model = self._mock_model()

        list(generate_speech("test text", "af_heart", model, speed=1.5, lang_code="b"))

        model.generate.assert_called_once_with(
            text="test text", voice="af_heart", speed=1.5, lang_code="b"
        )

    def test_default_speed_and_lang_code(self) -> None:
        model = self._mock_model()

        list(generate_speech("test", "af_heart", model))

        model.generate.assert_called_once_with(
            text="test", voice="af_heart", speed=1.0, lang_code="a"
        )

    def test_yields_multiple_chunks(self) -> None:
        model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = np.ones(100, dtype=np.float32)
        chunk2 = MagicMock()
        chunk2.audio = np.zeros(200, dtype=np.float32)
        model.generate.return_value = [chunk1, chunk2]

        results = list(generate_speech("long text", "af_heart", model, lang_code="a"))

        assert len(results) == 2
        assert results[0].shape == (100,)
        assert results[1].shape == (200,)

    def test_output_is_float32(self) -> None:
        model = self._mock_model()

        results = list(generate_speech("test", "af_heart", model, lang_code="a"))

        assert results[0].dtype == np.float32

    def test_raises_on_empty_chunks(self) -> None:
        model = MagicMock()
        model.generate.return_value = []

        with pytest.raises(ValueError, match="No audio generated"):
            list(generate_speech("test", "af_heart", model, lang_code="a"))

    def test_skips_chunks_with_none_audio(self) -> None:
        model = MagicMock()
        chunk1 = MagicMock()
        chunk1.audio = None
        chunk2 = MagicMock()
        chunk2.audio = np.ones(100, dtype=np.float32)
        model.generate.return_value = [chunk1, chunk2]

        results = list(generate_speech("test", "af_heart", model, lang_code="a"))

        assert len(results) == 1
        assert results[0].shape == (100,)


class TestGenerateOne:
    def _model(self, audio_length: int = 100) -> MagicMock:
        model = MagicMock()
        chunk = MagicMock()
        chunk.audio = np.ones(audio_length, dtype=np.float32)
        model.generate.return_value = [chunk]
        return model

    def _mock_tokenizer(self, phonemes: str = "hɛlˈoʊ") -> None:
        from misaki import en

        en.G2P.return_value = MagicMock(return_value=(phonemes, None))  # type: ignore[union-attribute]

    def test_returns_voice_result(self) -> None:
        self._mock_tokenizer()
        model = self._model()
        result = generate_one("hi", "af_heart", model, 1.0, "a")
        assert result["voice"] == "af_heart"

    def test_phonemes_included(self) -> None:
        self._mock_tokenizer("test phonemes")
        model = self._model()
        result = generate_one("hi", "af_heart", model, 1.0, "a")
        assert result["phonemes"] == "test phonemes"

    def test_audio_concatenated(self) -> None:
        self._mock_tokenizer()
        model = MagicMock()
        c1, c2 = MagicMock(), MagicMock()
        c1.audio = np.ones(50, dtype=np.float32)
        c2.audio = np.zeros(30, dtype=np.float32)
        model.generate.return_value = [c1, c2]
        result = generate_one("hi", "af_heart", model, 1.0, "a")
        assert result["audio"].shape == (80,)

    def test_passes_speed_and_lang(self) -> None:
        self._mock_tokenizer()
        model = self._model()
        generate_one("hi", "af_heart", model, 1.5, "b")
        model.generate.assert_called_with(
            text="hi", voice="af_heart", speed=1.5, lang_code="b"
        )


class TestSplitVoicesForDisplay:
    LONG = [f"af_v{i}" for i in range(10)]  # 10 voices

    def test_empty_returns_two_empty_lists(self) -> None:
        assert _split_voices_for_display([], None) == ([], [])

    def test_at_or_below_top_n_all_visible(self) -> None:
        voices = ["af_heart", "af_bella", "am_adam"]
        visible, hidden = _split_voices_for_display(voices, None)
        assert visible == voices
        assert hidden == []

    def test_exactly_top_n_all_visible(self) -> None:
        voices = self.LONG[:6]
        visible, hidden = _split_voices_for_display(voices, None)
        assert visible == voices
        assert hidden == []

    def test_more_than_top_n_splits_at_six(self) -> None:
        visible, hidden = _split_voices_for_display(self.LONG, None)
        assert visible == self.LONG[:6]
        assert hidden == self.LONG[6:]

    def test_selected_in_top_n_unchanged(self) -> None:
        visible, hidden = _split_voices_for_display(self.LONG, "af_v2")
        assert visible == self.LONG[:6]
        assert hidden == self.LONG[6:]

    def test_selected_in_tail_pinned_to_visible(self) -> None:
        visible, hidden = _split_voices_for_display(self.LONG, "af_v8")
        assert visible == self.LONG[:6] + ["af_v8"]
        assert hidden == ["af_v6", "af_v7", "af_v9"]

    def test_selected_none_uses_default_split(self) -> None:
        visible, hidden = _split_voices_for_display(self.LONG, None)
        assert visible == self.LONG[:6]
        assert hidden == self.LONG[6:]

    def test_custom_top_n(self) -> None:
        visible, hidden = _split_voices_for_display(self.LONG, None, top_n=3)
        assert visible == self.LONG[:3]
        assert hidden == self.LONG[3:]

    def test_preserves_input_order(self) -> None:
        voices = ["c", "a", "b", "z", "y", "x", "m", "n"]
        visible, hidden = _split_voices_for_display(voices, None)
        assert visible == ["c", "a", "b", "z", "y", "x"]
        assert hidden == ["m", "n"]


class TestCacheKey:
    def test_starts_with_audio_prefix(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a").startswith("audio:")

    def test_includes_voice(self) -> None:
        assert "af_heart" in _cache_key("af_heart", "hi", 1.0, "a")

    def test_includes_lang_code(self) -> None:
        assert ":a:" in _cache_key("af_heart", "hi", 1.0, "a")

    def test_includes_speed(self) -> None:
        assert "1.5" in _cache_key("af_heart", "hi", 1.5, "a")

    def test_same_inputs_same_key(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") == _cache_key("af_heart", "hi", 1.0, "a")

    def test_distinguishes_text(self) -> None:
        assert _cache_key("af_heart", "hello", 1.0, "a") != _cache_key("af_heart", "world", 1.0, "a")

    def test_distinguishes_voice(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") != _cache_key("af_bella", "hi", 1.0, "a")

    def test_distinguishes_speed(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") != _cache_key("af_heart", "hi", 1.5, "a")

    def test_distinguishes_lang(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") != _cache_key("af_heart", "hi", 1.0, "b")


class TestSpeedOptions:
    def test_includes_default_one(self) -> None:
        assert 1.0 in SPEED_OPTIONS

    def test_min_is_zero_seven(self) -> None:
        assert min(SPEED_OPTIONS) == 0.7

    def test_max_is_one_five(self) -> None:
        assert max(SPEED_OPTIONS) == 1.5

    def test_sorted_ascending(self) -> None:
        assert SPEED_OPTIONS == sorted(SPEED_OPTIONS)

    def test_default_speed_index_points_to_one(self) -> None:
        assert SPEED_OPTIONS[DEFAULT_SPEED_INDEX] == 1.0


class TestAudioToWavBytes:
    def test_returns_bytes(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _audio_to_wav_bytes(audio)
        assert isinstance(result, bytes)

    def test_starts_with_riff_header(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = _audio_to_wav_bytes(audio)
        assert result[:4] == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_handles_nonzero_audio(self) -> None:
        audio = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        result = _audio_to_wav_bytes(audio)
        assert isinstance(result, bytes)
        assert len(result) > 1000  # header + payload


class TestEvictOldAudio:
    @staticmethod
    def _clear_audio_cache() -> None:
        for k in list(st.session_state):
            if isinstance(k, str) and k.startswith("audio:"):
                del st.session_state[k]

    @staticmethod
    def _fill_cache(n: int) -> None:
        for i in range(n):
            st.session_state[f"audio:v{i}:a:1.0:{i}"] = {
                "audio": np.zeros(1, dtype=np.float32),
                "voice": f"v{i}",
                "phonemes": "x",
            }

    @staticmethod
    def _count_audio_keys() -> int:
        return sum(
            1 for k in st.session_state if isinstance(k, str) and k.startswith("audio:")
        )

    def test_no_op_when_empty(self) -> None:
        self._clear_audio_cache()
        _evict_old_audio()
        assert self._count_audio_keys() == 0

    def test_no_op_when_under_limit(self) -> None:
        self._clear_audio_cache()
        self._fill_cache(AUDIO_CACHE_LIMIT - 1)
        _evict_old_audio()
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT - 1
        self._clear_audio_cache()

    def test_no_op_when_at_limit(self) -> None:
        self._clear_audio_cache()
        self._fill_cache(AUDIO_CACHE_LIMIT)
        _evict_old_audio()
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        self._clear_audio_cache()

    def test_evicts_oldest_when_over_limit(self) -> None:
        self._clear_audio_cache()
        self._fill_cache(AUDIO_CACHE_LIMIT + 1)
        _evict_old_audio()
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        # The oldest key (v0) should be gone; the newest (v20) should remain
        assert "audio:v0:a:1.0:0" not in st.session_state
        assert f"audio:v{AUDIO_CACHE_LIMIT}:a:1.0:{AUDIO_CACHE_LIMIT}" in st.session_state
        self._clear_audio_cache()

    def test_evicts_multiple_when_far_over_limit(self) -> None:
        self._clear_audio_cache()
        self._fill_cache(AUDIO_CACHE_LIMIT + 5)
        _evict_old_audio()
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        # First 5 keys evicted
        for i in range(5):
            assert f"audio:v{i}:a:1.0:{i}" not in st.session_state
        self._clear_audio_cache()

    def test_preserves_non_audio_session_keys(self) -> None:
        self._clear_audio_cache()
        st.session_state["language"] = "American English"
        st.session_state["female"] = False
        self._fill_cache(AUDIO_CACHE_LIMIT + 1)
        _evict_old_audio()
        assert st.session_state["language"] == "American English"
        assert st.session_state["female"] is False
        self._clear_audio_cache()


class TestFindStaleCachedAudio:
    @staticmethod
    def _clear_audio_cache() -> None:
        for k in list(st.session_state):
            if isinstance(k, str) and k.startswith("audio:"):
                del st.session_state[k]

    def test_returns_none_when_no_cache(self) -> None:
        self._clear_audio_cache()
        assert _find_stale_cached_audio("af_heart", "hello", "a") is None

    def test_finds_audio_at_different_speed(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 0.7, "a")
        payload = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        st.session_state[key] = payload
        assert _find_stale_cached_audio("af_heart", "hello", "a") is payload
        self._clear_audio_cache()

    def test_does_not_find_audio_for_different_text(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        assert _find_stale_cached_audio("af_heart", "world", "a") is None
        self._clear_audio_cache()

    def test_does_not_find_audio_for_different_voice(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        assert _find_stale_cached_audio("af_bella", "hello", "a") is None
        self._clear_audio_cache()

    def test_does_not_find_audio_for_different_lang(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        assert _find_stale_cached_audio("af_heart", "hello", "b") is None
        self._clear_audio_cache()

    def test_returns_most_recent_when_multiple_speeds_cached(self) -> None:
        self._clear_audio_cache()
        key_07 = _cache_key("af_heart", "hello", 0.7, "a")
        key_15 = _cache_key("af_heart", "hello", 1.5, "a")
        st.session_state[key_07] = {
            "audio": np.zeros(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        st.session_state[key_15] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        result = _find_stale_cached_audio("af_heart", "hello", "a")
        assert result is not None
        assert result["audio"][0] == 1.0
        self._clear_audio_cache()


class TestRenderVoiceCard:
    @staticmethod
    def _reset_mocks() -> None:
        st.container.reset_mock()  # type: ignore[union-attribute]
        st.markdown.reset_mock()  # type: ignore[union-attribute]
        st.button.reset_mock()  # type: ignore[union-attribute]
        st.button.return_value = False  # type: ignore[union-attribute]
        st.selectbox.reset_mock()  # type: ignore[union-attribute]
        st.audio.reset_mock()  # type: ignore[union-attribute]
        st.caption.reset_mock()  # type: ignore[union-attribute]
        st.download_button.reset_mock()  # type: ignore[union-attribute]
        for k in list(st.session_state):
            if isinstance(k, str) and k.startswith("audio:"):
                del st.session_state[k]

    def test_renders_bordered_container(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.container.assert_called_once_with(border=True)  # type: ignore[union-attribute]

    def test_renders_formatted_title(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # type: ignore[union-attribute]

    def test_badge_when_cached_at_current_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**🔊 Heart (female) — A**")  # type: ignore[union-attribute]

    def test_badge_when_cached_at_other_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 0.7, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**🔊 Heart (female) — A**")  # type: ignore[union-attribute]

    def test_no_badge_when_cache_for_different_text(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "different_text", 1.0, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # type: ignore[union-attribute]

    def test_no_badge_when_cache_for_different_voice(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_bella", "hello", 1.0, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_bella",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # type: ignore[union-attribute]

    def test_play_button_key_is_voice_specific(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[1]["key"] == "play_af_heart"  # type: ignore[union-attribute]

    def test_play_button_uses_primary_type(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[1]["type"] == "primary"  # type: ignore[union-attribute]

    def test_play_button_label(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[0][0] == "▶ Play"  # type: ignore[union-attribute]

    def test_play_disabled_when_text_empty(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "", "a")
        assert st.button.call_args[1]["disabled"] is True  # type: ignore[union-attribute]

    def test_play_enabled_when_text_nonempty(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[1]["disabled"] is False  # type: ignore[union-attribute]

    def test_renders_speed_selectbox(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        speed_call = next(
            (c for c in st.selectbox.call_args_list if c.args and c.args[0] == "Speed"),  # type: ignore[union-attribute]
            None,
        )
        assert speed_call is not None
        assert speed_call.kwargs["options"] == SPEED_OPTIONS
        assert speed_call.kwargs["key"] == "speed_af_heart"

    def test_speed_format_func_renders_x_suffix(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        speed_call = next(
            c for c in st.selectbox.call_args_list if c.args and c.args[0] == "Speed"  # type: ignore[union-attribute]
        )
        assert speed_call.kwargs["format_func"](1.0) == "1.0x"
        assert speed_call.kwargs["format_func"](0.7) == "0.7x"

    def test_renders_audio_when_cached(self) -> None:
        self._reset_mocks()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "hɛlˈoʊ",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_called_once()  # type: ignore[union-attribute]
        del st.session_state[key]

    def test_no_audio_when_not_cached(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "never_cached_for_this_test", "a")
        st.audio.assert_not_called()  # type: ignore[union-attribute]

    def test_audio_uses_correct_sample_rate_when_cached(self) -> None:
        self._reset_mocks()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        assert st.audio.call_args[1]["sample_rate"] == SAMPLE_RATE  # type: ignore[union-attribute]
        del st.session_state[key]

    def test_renders_stale_audio_with_caption_when_only_other_speed_cached(self) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # type: ignore[union-attribute]
        # Cache key uses speed=0.7, but the conftest selectbox mock returns 1.0
        old_key = _cache_key("af_heart", "hello", 0.7, "a")
        st.session_state[old_key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_called_once()  # type: ignore[union-attribute]
        st.caption.assert_called_once_with("Click Play to refresh (speed changed)")  # type: ignore[union-attribute]
        del st.session_state[old_key]

    def test_no_stale_caption_when_fresh_audio_cached(self) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # type: ignore[union-attribute]
        # Both fresh (1.0) and stale (0.7) cached
        fresh_key = _cache_key("af_heart", "hello", 1.0, "a")
        stale_key = _cache_key("af_heart", "hello", 0.7, "a")
        st.session_state[fresh_key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        st.session_state[stale_key] = {
            "audio": np.zeros(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_called_once()  # type: ignore[union-attribute]
        st.caption.assert_not_called()  # type: ignore[union-attribute]
        del st.session_state[fresh_key]
        del st.session_state[stale_key]

    def test_download_button_rendered_when_fresh_audio_cached(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.download_button.assert_called_once()  # type: ignore[union-attribute]

    def test_download_button_file_name_includes_voice_and_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.download_button.call_args[1]  # type: ignore[union-attribute]
        assert kwargs["file_name"] == "af_heart_1.0x.wav"
        assert kwargs["mime"] == "audio/wav"
        assert kwargs["key"] == "download_af_heart"

    def test_download_button_data_is_bytes(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.download_button.call_args[1]  # type: ignore[union-attribute]
        assert isinstance(kwargs["data"], bytes)
        assert kwargs["data"][:4] == b"RIFF"

    def test_no_download_button_for_stale_audio(self) -> None:
        self._reset_mocks()
        # Cached at speed 0.7 — stale relative to default speed 1.0
        st.session_state[_cache_key("af_heart", "hello", 0.7, "a")] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.download_button.assert_not_called()  # type: ignore[union-attribute]

    def test_no_download_button_when_no_cached_audio(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.download_button.assert_not_called()  # type: ignore[union-attribute]

    def test_click_populates_cache_and_calls_generate(self) -> None:
        self._reset_mocks()
        st.button.return_value = True  # type: ignore[union-attribute]
        fake_result = {
            "audio": np.ones(50, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "hɛlˈoʊ",
        }
        with (
            patch("streamlit_app.load_pipeline", return_value="fake_pipeline") as mock_load,
            patch("streamlit_app.generate_one", return_value=fake_result) as mock_gen,
        ):
            render_voice_card("af_heart", "hello", "a")
        mock_load.assert_called_once()
        mock_gen.assert_called_once_with(
            "hello", "af_heart", "fake_pipeline", 1.0, "a"
        )
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        assert st.session_state[expected_key] is fake_result
        del st.session_state[expected_key]

    def test_click_handles_generate_error(self) -> None:
        self._reset_mocks()
        st.button.return_value = True  # type: ignore[union-attribute]
        st.exception.reset_mock()  # type: ignore[union-attribute]
        with (
            patch("streamlit_app.load_pipeline"),
            patch("streamlit_app.generate_one", side_effect=RuntimeError("boom")),
        ):
            render_voice_card("af_heart", "hello", "a")
        st.exception.assert_called_once()  # type: ignore[union-attribute]
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        assert expected_key not in st.session_state

    def test_click_triggers_eviction(self) -> None:
        self._reset_mocks()
        st.button.return_value = True  # type: ignore[union-attribute]
        fake_result = {
            "audio": np.ones(50, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        with (
            patch("streamlit_app.load_pipeline"),
            patch("streamlit_app.generate_one", return_value=fake_result),
            patch("streamlit_app._evict_old_audio") as mock_evict,
        ):
            render_voice_card("af_heart", "hello", "a")
        mock_evict.assert_called_once()
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        del st.session_state[expected_key]

    def test_no_stale_audio_when_text_differs(self) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # type: ignore[union-attribute]
        # Stale audio exists but for different text
        old_key = _cache_key("af_heart", "different_text", 0.7, "a")
        st.session_state[old_key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_not_called()  # type: ignore[union-attribute]
        st.caption.assert_not_called()  # type: ignore[union-attribute]
        del st.session_state[old_key]


class TestRenderPhonemes:
    def test_renders_expander_and_code(self) -> None:
        st.expander.reset_mock()  # type: ignore[union-attribute]
        st.code.reset_mock()  # type: ignore[union-attribute]
        render_phonemes("hɛlˈoʊ")
        st.expander.assert_called_once_with("Phoneme Tokens", expanded=False)  # type: ignore[union-attribute]
        st.code.assert_called_once_with("hɛlˈoʊ")  # type: ignore[union-attribute]

    def test_expanded_flag_forwarded(self) -> None:
        st.expander.reset_mock()  # type: ignore[union-attribute]
        render_phonemes("x", expanded=True)
        st.expander.assert_called_once_with("Phoneme Tokens", expanded=True)  # type: ignore[union-attribute]


class TestRenderPersistentPhonemes:
    @staticmethod
    def _reset_mocks() -> None:
        st.expander.reset_mock()  # type: ignore[union-attribute]
        st.code.reset_mock()  # type: ignore[union-attribute]
        st.session_state.pop("last_phonemes", None)

    def test_no_render_when_nothing_cached(self) -> None:
        self._reset_mocks()
        _render_persistent_phonemes("hello", "a")
        st.expander.assert_not_called()  # type: ignore[union-attribute]

    def test_renders_when_text_and_lang_match(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("hello", "a")
        st.expander.assert_called_once_with("Phoneme Tokens", expanded=True)  # type: ignore[union-attribute]
        st.code.assert_called_once_with("hɛlˈoʊ")  # type: ignore[union-attribute]

    def test_no_render_when_text_differs(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("world", "a")
        st.expander.assert_not_called()  # type: ignore[union-attribute]

    def test_no_render_when_lang_differs(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("hello", "b")
        st.expander.assert_not_called()  # type: ignore[union-attribute]

    def test_renders_again_when_returning_to_matched_text(self) -> None:
        # User typed "hello", tokenized, switched text, switched back
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("hello", "a")
        st.expander.assert_called_once()  # type: ignore[union-attribute]


class TestPronunciationTips:
    def test_is_nonempty_string(self) -> None:
        assert isinstance(PRONUNCIATION_TIPS, str) and len(PRONUNCIATION_TIPS) > 0

    def test_contains_custom_pronunciation_syntax(self) -> None:
        assert "[word](/phonemes/)" in PRONUNCIATION_TIPS

    def test_contains_intonation_info(self) -> None:
        assert "Intonation" in PRONUNCIATION_TIPS

    def test_contains_stress_adjustment(self) -> None:
        assert "[word](-1)" in PRONUNCIATION_TIPS
        assert "[word](+1)" in PRONUNCIATION_TIPS

    def test_no_leading_trailing_whitespace(self) -> None:
        assert PRONUNCIATION_TIPS == PRONUNCIATION_TIPS.strip()


class TestFormatVoice:
    def test_american_female(self) -> None:
        assert _format_voice("af_heart") == "Heart (female) — A"

    def test_american_male(self) -> None:
        assert _format_voice("am_adam") == "Adam (male) — F+"

    def test_british_female(self) -> None:
        assert _format_voice("bf_alice") == "Alice (female) — D"

    def test_japanese_female(self) -> None:
        assert _format_voice("jf_alpha") == "Alpha (female) — C+"

    def test_title_cases_name(self) -> None:
        assert _format_voice("af_bella") == "Bella (female) — A-"

    def test_ungraded_voice_omits_grade_suffix(self) -> None:
        # Spanish/Portuguese voices have no published grades
        assert _format_voice("ef_dora") == "Dora (female)"

    def test_multi_underscore_name_keeps_all_parts(self) -> None:
        assert _format_voice("af_some_long_name") == "Some Long Name (female)"

    def test_unknown_gender_char_returns_name_only(self) -> None:
        assert _format_voice("ax_mystery") == "Mystery"

    def test_no_underscore_returns_raw(self) -> None:
        assert _format_voice("af") == "af"


class TestGenderCodeFromCheckboxes:
    def test_both_checked_returns_none(self) -> None:
        assert _gender_code_from_checkboxes(True, True) is None

    def test_neither_checked_returns_none(self) -> None:
        assert _gender_code_from_checkboxes(False, False) is None

    def test_only_female_returns_f(self) -> None:
        assert _gender_code_from_checkboxes(True, False) == "f"

    def test_only_male_returns_m(self) -> None:
        assert _gender_code_from_checkboxes(False, True) == "m"


class TestFilterVoicesByGender:
    VOICES = ["af_bella", "af_heart", "am_adam", "am_echo"]

    def test_all_returns_unchanged(self) -> None:
        assert _filter_voices_by_gender(self.VOICES, None) == self.VOICES

    def test_female_filters_to_f(self) -> None:
        assert _filter_voices_by_gender(self.VOICES, "f") == ["af_bella", "af_heart"]

    def test_male_filters_to_m(self) -> None:
        assert _filter_voices_by_gender(self.VOICES, "m") == ["am_adam", "am_echo"]

    def test_empty_input_returns_empty(self) -> None:
        assert _filter_voices_by_gender([], "f") == []

    def test_no_matches_returns_empty(self) -> None:
        assert _filter_voices_by_gender(["af_bella"], "m") == []

    def test_preserves_input_order(self) -> None:
        voices = ["af_heart", "am_adam", "af_bella"]
        assert _filter_voices_by_gender(voices, "f") == ["af_heart", "af_bella"]


class TestGradeRank:
    def test_grade_a_ranks_lower_than_grade_a_minus(self) -> None:
        assert _grade_rank("af_heart") < _grade_rank("af_bella")

    def test_grade_a_minus_ranks_lower_than_grade_f_plus(self) -> None:
        assert _grade_rank("af_bella") < _grade_rank("am_adam")

    def test_ungraded_voice_ranks_last(self) -> None:
        # Spanish voices have no published grade
        assert _grade_rank("ef_dora") > _grade_rank("am_adam")

    def test_unknown_voice_ranks_last(self) -> None:
        assert _grade_rank("xx_unknown") > _grade_rank("am_adam")

    def test_known_specific_ranks(self) -> None:
        # A = 2, A- = 3, F+ = 13, ungraded = 99
        assert _grade_rank("af_heart") == 2
        assert _grade_rank("af_bella") == 3
        assert _grade_rank("am_adam") == 13
        assert _grade_rank("ef_dora") == 99


class TestEnsureRepoDownloaded:
    def test_returns_path_string(self) -> None:
        result = ensure_repo_downloaded()
        assert isinstance(result, str)
        assert result  # non-empty

    def test_calls_snapshot_download(self) -> None:
        from huggingface_hub import snapshot_download

        ensure_repo_downloaded()
        snapshot_download.assert_called()  # type: ignore[union-attribute]


class TestVoiceGrades:
    def test_all_keys_match_voice_id_pattern(self) -> None:
        for voice in VOICE_GRADES:
            assert "_" in voice, f"{voice!r} missing underscore"
            assert len(voice) >= 4, f"{voice!r} too short"
            assert voice[1] in ("f", "m"), f"{voice!r} has invalid gender code"

    def test_all_grades_are_known(self) -> None:
        for voice, grade in VOICE_GRADES.items():
            assert grade in _GRADE_RANK, f"{voice!r} has unknown grade {grade!r}"
