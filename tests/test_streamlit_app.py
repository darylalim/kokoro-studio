from pathlib import Path
from typing import Any
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
    SAMPLE_BUTTONS,
    SAMPLE_RATE,
    SPEED_OPTIONS,
    _PHONEME_MULTIPLIERS,
    _audio_to_wav_bytes,
    _cache_key,
    _create_g2p,
    _estimate_phonemes,
    _evict_old_audio,
    _filter_voices_by_gender,
    _find_stale_cached_audio,
    _format_voice,
    _gender_code_from_selection,
    _load_sample,
    _next_audio_seq,
    _phoneme_band,
    _pick_sample,
    _render_length_caption,
    _render_persistent_phonemes,
    _render_sample_buttons,
    _set_text_from_sample,
    _split_voices_for_display,
    _text_digest,
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

    @pytest.mark.parametrize(
        ("code", "espeak_lang"),
        [
            ("e", "es"),
            ("f", "fr-fr"),
            ("h", "hi"),
            ("i", "it"),
            ("p", "pt-br"),
        ],
        ids=["spanish", "french", "hindi", "italian", "portuguese"],
    )
    def test_maps_to_correct_espeak_codes(self, code: str, espeak_lang: str) -> None:
        assert ESPEAK_LANGUAGES[code] == espeak_lang

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
        load_model.assert_called_with(REPO_ID)  # ty: ignore[unresolved-attribute]


class TestCreateG2p:
    def test_american_english_uses_en_g2p(self) -> None:
        from misaki import en

        _create_g2p("a")
        en.G2P.assert_called()  # ty: ignore[unresolved-attribute]
        call_kwargs = en.G2P.call_args[1]  # ty: ignore[unresolved-attribute]
        assert call_kwargs["british"] is False

    def test_british_english_uses_en_g2p_with_british(self) -> None:
        from misaki import en

        _create_g2p("b")
        call_kwargs = en.G2P.call_args[1]  # ty: ignore[unresolved-attribute]
        assert call_kwargs["british"] is True

    def test_japanese_uses_ja_g2p(self) -> None:
        from misaki import ja

        _create_g2p("j")
        ja.JAG2P.assert_called()  # ty: ignore[unresolved-attribute]

    def test_chinese_uses_zh_g2p(self) -> None:
        from misaki import zh

        _create_g2p("z")
        zh.ZHG2P.assert_called()  # ty: ignore[unresolved-attribute]

    def test_espeak_languages_use_espeak_g2p(self) -> None:
        from misaki import espeak

        for code, espeak_lang in ESPEAK_LANGUAGES.items():
            espeak.EspeakG2P.reset_mock()  # ty: ignore[unresolved-attribute]
            _create_g2p(code)
            espeak.EspeakG2P.assert_called_with(language=espeak_lang)  # ty: ignore[unresolved-attribute]


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
        en.G2P.return_value = mock_g2p  # ty: ignore[unresolved-attribute]
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

        en.G2P.return_value = MagicMock(return_value=(None, None))  # ty: ignore[unresolved-attribute]

        result = tokenize_text("", "a")

        assert result == ""

    def test_british_english_uses_british_g2p(self) -> None:
        self._mock_g2p("hɛlˈəʊ")

        tokenize_text("hello", "b")

        from misaki import en

        call_kwargs = en.G2P.call_args[1]  # ty: ignore[unresolved-attribute]
        assert call_kwargs["british"] is True

    def test_japanese_uses_ja_g2p(self) -> None:
        from misaki import ja

        ja.JAG2P.reset_mock()  # ty: ignore[unresolved-attribute]
        ja.JAG2P.return_value = MagicMock(return_value=("konniʧiwa", None))  # ty: ignore[unresolved-attribute]

        result = tokenize_text("こんにちは", "j")

        assert result == "konniʧiwa"
        ja.JAG2P.assert_called_once()  # ty: ignore[unresolved-attribute]

    def test_spanish_uses_espeak_g2p(self) -> None:
        from misaki import espeak

        espeak.EspeakG2P.return_value = MagicMock(return_value=("ola", None))  # ty: ignore[unresolved-attribute]

        result = tokenize_text("hola", "e")

        assert result == "ola"
        espeak.EspeakG2P.assert_called_with(language="es")  # ty: ignore[unresolved-attribute]


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

        en.G2P.return_value = MagicMock(return_value=(phonemes, None))  # ty: ignore[unresolved-attribute]

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

    def test_raises_on_empty_chunks(self) -> None:
        # Locks the end-to-end "No audio generated" propagation that the card's
        # ValueError -> st.error branch depends on (match pins the exact message,
        # so a refactor letting np.concatenate([]) raise instead would fail).
        self._mock_tokenizer()
        model = MagicMock()
        chunk = MagicMock()
        chunk.audio = None
        model.generate.return_value = [chunk]
        with pytest.raises(ValueError, match="No audio generated"):
            generate_one("hi", "af_heart", model, 1.0, "a")


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
        assert _cache_key("af_heart", "hi", 1.0, "a") == _cache_key(
            "af_heart", "hi", 1.0, "a"
        )

    def test_distinguishes_text(self) -> None:
        assert _cache_key("af_heart", "hello", 1.0, "a") != _cache_key(
            "af_heart", "world", 1.0, "a"
        )

    def test_distinguishes_voice(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") != _cache_key(
            "af_bella", "hi", 1.0, "a"
        )

    def test_distinguishes_speed(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") != _cache_key(
            "af_heart", "hi", 1.5, "a"
        )

    def test_distinguishes_lang(self) -> None:
        assert _cache_key("af_heart", "hi", 1.0, "a") != _cache_key(
            "af_heart", "hi", 1.0, "b"
        )


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
                "seq": i,
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
        assert (
            f"audio:v{AUDIO_CACHE_LIMIT}:a:1.0:{AUDIO_CACHE_LIMIT}" in st.session_state
        )
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

    def test_evicts_lowest_seq_not_insertion_order(self) -> None:
        # seq, not st.session_state iteration order, decides the eviction victim.
        # Insert so the oldest (lowest seq) is the LAST key inserted.
        self._clear_audio_cache()
        n = AUDIO_CACHE_LIMIT + 1
        for i in range(n):
            st.session_state[f"audio:v{i}:a:1.0:{i}"] = {
                "audio": np.zeros(1, dtype=np.float32),
                "voice": f"v{i}",
                "phonemes": "x",
                "seq": n - i,  # last-inserted v{n-1} has the lowest seq (=1)
            }
        _evict_old_audio()
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        assert f"audio:v{n - 1}:a:1.0:{n - 1}" not in st.session_state  # lowest seq
        assert "audio:v0:a:1.0:0" in st.session_state  # highest seq survives
        self._clear_audio_cache()

    def test_protected_key_survives_even_when_lowest_seq(self) -> None:
        # A key a card is currently displaying (passed in `protect`) is never
        # evicted, even if it is the oldest — guards the fragment scenario where
        # one card's Play would otherwise orphan a sibling's on-screen player.
        self._clear_audio_cache()
        self._fill_cache(AUDIO_CACHE_LIMIT + 1)
        oldest = "audio:v0:a:1.0:0"  # seq=0, the normal eviction victim
        _evict_old_audio(protect=frozenset({oldest}))
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        assert oldest in st.session_state  # protected despite being oldest
        assert "audio:v1:a:1.0:1" not in st.session_state  # next-oldest evicted
        self._clear_audio_cache()

    def test_preserves_non_audio_session_keys(self) -> None:
        self._clear_audio_cache()
        st.session_state["language"] = "American English"
        st.session_state["gender"] = "All"
        self._fill_cache(AUDIO_CACHE_LIMIT + 1)
        _evict_old_audio()
        assert st.session_state["language"] == "American English"
        assert st.session_state["gender"] == "All"
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
        # Insert the higher-seq (most-recent) entry FIRST so insertion order
        # disagrees with seq order — this fails the old matches[-1] impl, which
        # would return the last-inserted (lower-seq) entry instead.
        self._clear_audio_cache()
        key_07 = _cache_key("af_heart", "hello", 0.7, "a")
        key_15 = _cache_key("af_heart", "hello", 1.5, "a")
        st.session_state[key_15] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 2,
        }
        st.session_state[key_07] = {
            "audio": np.zeros(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 1,
        }
        result = _find_stale_cached_audio("af_heart", "hello", "a")
        assert result is not None
        assert result["audio"][0] == 1.0  # higher-seq 1.5 entry, not last-inserted
        self._clear_audio_cache()

    def test_returns_highest_seq_regardless_of_insertion_order(self) -> None:
        # Recency must come from seq, not st.session_state iteration order. The
        # mock session_state is a plain (insertion-ordered) dict, so insert the
        # higher-seq entry FIRST: the buggy matches[-1] then returns the
        # last-inserted lower-seq entry (0.0) and this assertion fails.
        self._clear_audio_cache()
        key_07 = _cache_key("af_heart", "hello", 0.7, "a")
        key_15 = _cache_key("af_heart", "hello", 1.5, "a")
        st.session_state[key_07] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 2,
        }
        st.session_state[key_15] = {
            "audio": np.zeros(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 1,
        }
        result = _find_stale_cached_audio("af_heart", "hello", "a")
        assert result is not None
        assert result["audio"][0] == 1.0  # higher-seq 0.7 entry, not last-inserted
        self._clear_audio_cache()


class TestRenderVoiceCard:
    @staticmethod
    def _reset_mocks() -> None:
        st.container.reset_mock()  # ty: ignore[unresolved-attribute]
        st.markdown.reset_mock()  # ty: ignore[unresolved-attribute]
        st.button.reset_mock()  # ty: ignore[unresolved-attribute]
        st.button.return_value = False  # ty: ignore[unresolved-attribute]
        st.selectbox.reset_mock()  # ty: ignore[unresolved-attribute]
        st.audio.reset_mock()  # ty: ignore[unresolved-attribute]
        st.caption.reset_mock()  # ty: ignore[unresolved-attribute]
        st.download_button.reset_mock()  # ty: ignore[unresolved-attribute]
        for k in list(st.session_state):
            if isinstance(k, str) and k.startswith("audio:"):
                del st.session_state[k]

    def test_renders_bordered_container(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.container.assert_called_once_with(border=True)  # ty: ignore[unresolved-attribute]

    def test_renders_formatted_title(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # ty: ignore[unresolved-attribute]

    def test_badge_when_cached_at_current_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            ":material/volume_up: **Heart (female) — A**"
        )

    def test_badge_when_cached_at_other_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 0.7, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            ":material/volume_up: **Heart (female) — A**"
        )

    def test_no_badge_when_cache_for_different_text(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "different_text", 1.0, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # ty: ignore[unresolved-attribute]

    def test_no_badge_when_cache_for_different_voice(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_bella", "hello", 1.0, "a")] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_bella",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # ty: ignore[unresolved-attribute]

    def test_play_button_key_is_voice_specific(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[1]["key"] == "play_af_heart"  # ty: ignore[unresolved-attribute]

    def test_play_button_uses_primary_type(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[1]["type"] == "primary"  # ty: ignore[unresolved-attribute]

    def test_play_button_label(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[0][0] == "Play"  # ty: ignore[unresolved-attribute]

    def test_registers_current_key_as_displayed_when_cached(self) -> None:
        # When the current-speed take exists it is what's on screen, so it is the
        # key protected — even when a NEWER take exists at another speed (which
        # _stale_cached_key would otherwise prefer by seq). This discriminates the
        # `key if key in st.session_state` branch from a buggy `stale_key or key`.
        self._reset_mocks()
        st.session_state.pop("_displayed_card_keys", None)
        current = _cache_key("af_heart", "hello", 1.0, "a")  # conftest speed = 1.0
        st.session_state[current] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 1,
        }
        newer_other_speed = _cache_key("af_heart", "hello", 0.7, "a")
        st.session_state[newer_other_speed] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 2,  # higher seq than the current-speed take
        }
        render_voice_card("af_heart", "hello", "a")
        assert st.session_state["_displayed_card_keys"]["af_heart"] == current
        del st.session_state[current]
        del st.session_state[newer_other_speed]

    def test_registers_stale_key_as_displayed_when_only_other_speed_cached(
        self,
    ) -> None:
        # Regression: when only another speed is cached, the card shows that
        # stale-preview audio — so the protect map must register the STALE key, not
        # the (uncached) current-speed key, or a sibling's Play could evict the
        # audio this card is actively displaying.
        self._reset_mocks()
        st.session_state.pop("_displayed_card_keys", None)
        stale = _cache_key("af_heart", "hello", 0.7, "a")  # not the current 1.0
        st.session_state[stale] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 1,
        }
        render_voice_card("af_heart", "hello", "a")
        assert st.session_state["_displayed_card_keys"]["af_heart"] == stale
        del st.session_state[stale]

    def test_play_disabled_when_text_empty(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "", "a")
        assert st.button.call_args[1]["disabled"] is True  # ty: ignore[unresolved-attribute]

    def test_play_enabled_when_text_nonempty(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.button.call_args[1]["disabled"] is False  # ty: ignore[unresolved-attribute]

    def test_renders_speed_selectbox(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        speed_call = next(
            (c for c in st.selectbox.call_args_list if c.args and c.args[0] == "Speed"),  # ty: ignore[unresolved-attribute]
            None,
        )
        assert speed_call is not None
        assert speed_call.kwargs["options"] == SPEED_OPTIONS
        assert speed_call.kwargs["key"] == "speed_af_heart"

    def test_speed_format_func_renders_x_suffix(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        calls = st.selectbox.call_args_list  # ty: ignore[unresolved-attribute]
        speed_call = next(c for c in calls if c.args and c.args[0] == "Speed")
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
        st.audio.assert_called_once()  # ty: ignore[unresolved-attribute]
        del st.session_state[key]

    def test_no_audio_when_not_cached(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "never_cached_for_this_test", "a")
        st.audio.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_audio_uses_correct_sample_rate_when_cached(self) -> None:
        self._reset_mocks()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        assert st.audio.call_args[1]["sample_rate"] == SAMPLE_RATE  # ty: ignore[unresolved-attribute]
        del st.session_state[key]

    def test_renders_stale_audio_with_caption_when_only_other_speed_cached(
        self,
    ) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # ty: ignore[unresolved-attribute]
        # Cache key uses speed=0.7, but the conftest selectbox mock returns 1.0
        old_key = _cache_key("af_heart", "hello", 0.7, "a")
        st.session_state[old_key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_called_once()  # ty: ignore[unresolved-attribute]
        st.caption.assert_called_once_with("Click Play to refresh (speed changed)")  # ty: ignore[unresolved-attribute]
        del st.session_state[old_key]

    def test_no_stale_caption_when_fresh_audio_cached(self) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # ty: ignore[unresolved-attribute]
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
        st.audio.assert_called_once()  # ty: ignore[unresolved-attribute]
        st.caption.assert_not_called()  # ty: ignore[unresolved-attribute]
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
        st.download_button.assert_called_once()  # ty: ignore[unresolved-attribute]

    def test_download_button_file_name_includes_voice_and_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.download_button.call_args[1]  # ty: ignore[unresolved-attribute]
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
        kwargs = st.download_button.call_args[1]  # ty: ignore[unresolved-attribute]
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
        st.download_button.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_no_download_button_when_no_cached_audio(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.download_button.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_click_populates_cache_and_calls_generate(self) -> None:
        self._reset_mocks()
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        fake_result = {
            "audio": np.ones(50, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "hɛlˈoʊ",
        }
        with (
            patch(
                "streamlit_app.load_pipeline", return_value="fake_pipeline"
            ) as mock_load,
            patch("streamlit_app.generate_one", return_value=fake_result) as mock_gen,
        ):
            render_voice_card("af_heart", "hello", "a")
        mock_load.assert_called_once()
        mock_gen.assert_called_once_with("hello", "af_heart", "fake_pipeline", 1.0, "a")
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        assert st.session_state[expected_key] is fake_result
        del st.session_state[expected_key]

    def test_click_handles_generate_error(self) -> None:
        self._reset_mocks()
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        st.exception.reset_mock()  # ty: ignore[unresolved-attribute]
        with (
            patch("streamlit_app.load_pipeline"),
            patch("streamlit_app.generate_one", side_effect=RuntimeError("boom")),
        ):
            render_voice_card("af_heart", "hello", "a")
        st.exception.assert_called_once()  # ty: ignore[unresolved-attribute]
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        assert expected_key not in st.session_state

    def test_click_renders_clean_error_for_no_audio(self) -> None:
        # A benign "No audio generated" (ValueError) gets a friendly st.error,
        # not a raw developer traceback via st.exception.
        self._reset_mocks()
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        st.error.reset_mock()  # ty: ignore[unresolved-attribute]
        st.exception.reset_mock()  # ty: ignore[unresolved-attribute]
        with (
            patch("streamlit_app.load_pipeline"),
            patch(
                "streamlit_app.generate_one",
                side_effect=ValueError("No audio generated. Check your input text."),
            ),
        ):
            render_voice_card("af_heart", "hello", "a")
        st.error.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            "No audio generated. Check your input text."
        )
        st.exception.assert_not_called()  # ty: ignore[unresolved-attribute]
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        assert expected_key not in st.session_state

    def test_click_unexpected_valueerror_uses_exception(self) -> None:
        # A ValueError that is NOT the benign "no audio" case keeps the
        # developer-facing st.exception, not a bare st.error.
        self._reset_mocks()
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        st.error.reset_mock()  # ty: ignore[unresolved-attribute]
        st.exception.reset_mock()  # ty: ignore[unresolved-attribute]
        with (
            patch("streamlit_app.load_pipeline"),
            patch(
                "streamlit_app.generate_one",
                side_effect=ValueError("some cryptic internal failure"),
            ),
        ):
            render_voice_card("af_heart", "hello", "a")
        st.exception.assert_called_once()  # ty: ignore[unresolved-attribute]
        st.error.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_click_forwards_displayed_keys_to_eviction(self) -> None:
        # The Play handler must forward every on-screen card's displayed key (plus
        # the just-written key) to _evict_old_audio, or a sibling's visible audio
        # could be orphaned. Guards the registration->eviction seam directly.
        self._reset_mocks()
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        sibling_key = _cache_key("af_bella", "hello", 0.7, "a")
        st.session_state["_displayed_card_keys"] = {"af_bella": sibling_key}
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
        protect = mock_evict.call_args[0][0]
        expected_key = _cache_key("af_heart", "hello", 1.0, "a")
        assert sibling_key in protect  # other on-screen card's key is shielded
        assert expected_key in protect  # the just-written take is shielded
        del st.session_state[expected_key]

    def test_play_reregisters_new_take_as_displayed(self) -> None:
        # Regression: after Play generates a new take over a stale preview, the
        # protect map must point at the NEW key (what the card now shows), not the
        # pre-Play stale key registered at render time — else a sibling's eviction
        # could orphan the freshly generated take.
        self._reset_mocks()
        st.session_state.pop("_displayed_card_keys", None)
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        stale = _cache_key("af_heart", "hello", 0.7, "a")  # prior take, other speed
        st.session_state[stale] = {
            "audio": np.ones(10, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
            "seq": 1,
        }
        current = _cache_key("af_heart", "hello", 1.0, "a")  # conftest speed = 1.0
        fake_result = {
            "audio": np.ones(50, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        with (
            patch("streamlit_app.load_pipeline"),
            patch("streamlit_app.generate_one", return_value=fake_result),
            patch("streamlit_app._evict_old_audio"),
        ):
            render_voice_card("af_heart", "hello", "a")
        assert st.session_state["_displayed_card_keys"]["af_heart"] == current
        del st.session_state[stale]
        del st.session_state[current]

    def test_no_stale_audio_when_text_differs(self) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # ty: ignore[unresolved-attribute]
        # Stale audio exists but for different text
        old_key = _cache_key("af_heart", "different_text", 0.7, "a")
        st.session_state[old_key] = {
            "audio": np.ones(100, dtype=np.float32),
            "voice": "af_heart",
            "phonemes": "x",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_not_called()  # ty: ignore[unresolved-attribute]
        st.caption.assert_not_called()  # ty: ignore[unresolved-attribute]
        del st.session_state[old_key]


class TestRenderPhonemes:
    def test_renders_expander_and_code(self) -> None:
        st.expander.reset_mock()  # ty: ignore[unresolved-attribute]
        st.code.reset_mock()  # ty: ignore[unresolved-attribute]
        render_phonemes("hɛlˈoʊ")
        st.expander.assert_called_once_with("Phoneme Tokens", expanded=False)  # ty: ignore[unresolved-attribute]
        st.code.assert_called_once_with("hɛlˈoʊ")  # ty: ignore[unresolved-attribute]

    def test_expanded_flag_forwarded(self) -> None:
        st.expander.reset_mock()  # ty: ignore[unresolved-attribute]
        render_phonemes("x", expanded=True)
        st.expander.assert_called_once_with("Phoneme Tokens", expanded=True)  # ty: ignore[unresolved-attribute]


class TestRenderPersistentPhonemes:
    @staticmethod
    def _reset_mocks() -> None:
        st.expander.reset_mock()  # ty: ignore[unresolved-attribute]
        st.code.reset_mock()  # ty: ignore[unresolved-attribute]
        st.session_state.pop("last_phonemes", None)

    def test_no_render_when_nothing_cached(self) -> None:
        self._reset_mocks()
        _render_persistent_phonemes("hello", "a")
        st.expander.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_renders_when_text_and_lang_match(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("hello", "a")
        st.expander.assert_called_once_with("Phoneme Tokens", expanded=True)  # ty: ignore[unresolved-attribute]
        st.code.assert_called_once_with("hɛlˈoʊ")  # ty: ignore[unresolved-attribute]

    def test_no_render_when_text_differs(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("world", "a")
        st.expander.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_no_render_when_lang_differs(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("hello", "b")
        st.expander.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_renders_again_when_returning_to_matched_text(self) -> None:
        # User typed "hello", tokenized, switched text, switched back
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello", "a", "hɛlˈoʊ")
        _render_persistent_phonemes("hello", "a")
        st.expander.assert_called_once()  # ty: ignore[unresolved-attribute]


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
    @pytest.mark.parametrize(
        ("voice", "expected"),
        [
            ("af_heart", "Heart (female) — A"),
            ("am_adam", "Adam (male) — F+"),
            ("bf_alice", "Alice (female) — D"),
            ("jf_alpha", "Alpha (female) — C+"),
            ("af_bella", "Bella (female) — A-"),
            # Spanish/Portuguese voices have no published grades
            ("ef_dora", "Dora (female)"),
            ("af_some_long_name", "Some Long Name (female)"),
            ("ax_mystery", "Mystery"),
            ("af", "af"),
        ],
        ids=[
            "american-female",
            "american-male",
            "british-female",
            "japanese-female",
            "title-cases-name",
            "ungraded-omits-suffix",
            "multi-underscore-name",
            "unknown-gender-char",
            "no-underscore-raw",
        ],
    )
    def test_format_voice(self, voice: str, expected: str) -> None:
        assert _format_voice(voice) == expected


class TestGenderCodeFromSelection:
    @pytest.mark.parametrize(
        ("selected", "expected"),
        [
            ("All", None),
            (None, None),
            ("Female", "f"),
            ("Male", "m"),
        ],
        ids=["all", "none", "female", "male"],
    )
    def test_gender_code_from_selection(
        self, selected: str | None, expected: str | None
    ) -> None:
        assert _gender_code_from_selection(selected) == expected


class TestFilterVoicesByGender:
    @pytest.mark.parametrize(
        ("voices", "gender_code", "expected"),
        [
            (
                ["af_bella", "af_heart", "am_adam", "am_echo"],
                None,
                ["af_bella", "af_heart", "am_adam", "am_echo"],
            ),
            (
                ["af_bella", "af_heart", "am_adam", "am_echo"],
                "f",
                ["af_bella", "af_heart"],
            ),
            (
                ["af_bella", "af_heart", "am_adam", "am_echo"],
                "m",
                ["am_adam", "am_echo"],
            ),
            ([], "f", []),
            (["af_bella"], "m", []),
            (
                ["af_heart", "am_adam", "af_bella"],
                "f",
                ["af_heart", "af_bella"],
            ),
        ],
        ids=[
            "no-filter-returns-all",
            "filters-to-female",
            "filters-to-male",
            "empty-input",
            "no-matches",
            "preserves-input-order",
        ],
    )
    def test_filter_voices_by_gender(
        self, voices: list[str], gender_code: str | None, expected: list[str]
    ) -> None:
        assert _filter_voices_by_gender(voices, gender_code) == expected


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
        snapshot_download.assert_called()  # ty: ignore[unresolved-attribute]


class TestVoiceGrades:
    def test_all_keys_match_voice_id_pattern(self) -> None:
        for voice in VOICE_GRADES:
            assert "_" in voice, f"{voice!r} missing underscore"
            assert len(voice) >= 4, f"{voice!r} too short"
            assert voice[1] in ("f", "m"), f"{voice!r} has invalid gender code"

    def test_all_grades_are_known(self) -> None:
        for voice, grade in VOICE_GRADES.items():
            assert grade in _GRADE_RANK, f"{voice!r} has unknown grade {grade!r}"


class TestPhonemeMultipliersConfig:
    def test_covers_all_languages(self) -> None:
        assert set(_PHONEME_MULTIPLIERS.keys()) == set(LANGUAGES.values())

    def test_all_values_are_positive_floats(self) -> None:
        for code, mult in _PHONEME_MULTIPLIERS.items():
            assert isinstance(mult, float), (
                f"{code} multiplier is {type(mult).__name__}"
            )
            assert mult > 0, f"{code} multiplier {mult} must be positive"


class TestEstimatePhonemes:
    def test_empty_returns_zero(self) -> None:
        assert _estimate_phonemes("", "a") == 0

    def test_whitespace_only_returns_zero(self) -> None:
        assert _estimate_phonemes("   \n  ", "a") == 0

    def test_english_uses_low_multiplier(self) -> None:
        assert _estimate_phonemes("hello world", "a") == int(11 * 0.85)

    def test_british_english_same_as_american(self) -> None:
        text = "hello world"
        assert _estimate_phonemes(text, "b") == _estimate_phonemes(text, "a")

    def test_japanese_higher_than_english(self) -> None:
        text = "abcdefghij"
        assert _estimate_phonemes(text, "j") > _estimate_phonemes(text, "a")

    def test_mandarin_uses_highest_multiplier(self) -> None:
        assert _estimate_phonemes("abcd", "z") == int(4 * 2.0)

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _estimate_phonemes("  hello  ", "a") == _estimate_phonemes("hello", "a")

    def test_unknown_lang_uses_default(self) -> None:
        assert _estimate_phonemes("hello", "x") == 5


class TestPhonemeBand:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, ("red", "very short")),
            (19, ("red", "very short")),
            (20, ("orange", "short")),
            (99, ("orange", "short")),
            (100, ("green", "ideal")),
            (399, ("green", "ideal")),
            (400, ("orange", "long")),
            (509, ("orange", "long")),
            (510, ("red", "will be chunked")),
            (10_000, ("red", "will be chunked")),
        ],
        ids=[
            "zero-very-short",
            "below-20-very-short",
            "lower-bound-short",
            "upper-bound-short",
            "lower-bound-ideal",
            "upper-bound-ideal",
            "lower-bound-long",
            "upper-bound-long",
            "lower-bound-chunked",
            "large-chunked",
        ],
    )
    def test_phoneme_band(self, n: int, expected: tuple[str, str]) -> None:
        assert _phoneme_band(n) == expected


class TestRenderLengthCaption:
    @staticmethod
    def _reset_mocks() -> None:
        st.caption.reset_mock()  # ty: ignore[unresolved-attribute]
        st.session_state.pop("last_phonemes", None)

    def test_empty_text_renders_nothing(self) -> None:
        self._reset_mocks()
        _render_length_caption("", "a")
        st.caption.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_whitespace_only_renders_nothing(self) -> None:
        self._reset_mocks()
        _render_length_caption("   \n  ", "a")
        st.caption.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_estimate_used_when_no_cached_phonemes(self) -> None:
        self._reset_mocks()
        _render_length_caption("hello world", "a")
        st.caption.assert_called_once()  # ty: ignore[unresolved-attribute]
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "~" in arg
        assert "9 phonemes" in arg
        assert "very short" in arg
        assert ":red[" in arg

    def test_exact_count_used_when_cached_phonemes_match(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello world", "a", "x" * 50)
        _render_length_caption("hello world", "a")
        st.caption.assert_called_once()  # ty: ignore[unresolved-attribute]
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "~" not in arg
        assert "50 phonemes" in arg
        assert "short" in arg
        assert ":orange[" in arg

    def test_estimate_used_when_cached_for_different_text(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("other text", "a", "x" * 200)
        _render_length_caption("hello world", "a")
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "~" in arg

    def test_estimate_used_when_cached_for_different_lang(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("hello world", "b", "x" * 200)
        _render_length_caption("hello world", "a")
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert "~" in arg

    def test_ideal_band_renders_green(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("xyz", "a", "x" * 250)
        _render_length_caption("xyz", "a")
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert ":green[" in arg
        assert "ideal" in arg

    def test_chunked_band_renders_red(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("xyz", "a", "x" * 600)
        _render_length_caption("xyz", "a")
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert ":red[" in arg
        assert "will be chunked" in arg

    def test_long_band_renders_orange(self) -> None:
        self._reset_mocks()
        st.session_state["last_phonemes"] = ("xyz", "a", "x" * 450)
        _render_length_caption("xyz", "a")
        arg = st.caption.call_args[0][0]  # ty: ignore[unresolved-attribute]
        assert ":orange[" in arg
        assert "long" in arg


class TestSampleButtonsConfig:
    def test_covers_all_languages(self) -> None:
        assert set(SAMPLE_BUTTONS.keys()) == set(LANGUAGES.values())

    def test_each_language_has_three_buttons(self) -> None:
        for lang, buttons in SAMPLE_BUTTONS.items():
            assert len(buttons) == 3, f"{lang} has {len(buttons)} buttons, expected 3"

    def test_each_language_has_exactly_one_random_button(self) -> None:
        for lang, buttons in SAMPLE_BUTTONS.items():
            random_count = sum(1 for _, _, is_random in buttons if is_random)
            assert random_count == 1, f"{lang} has {random_count} random buttons"

    def test_button_tuples_have_correct_shape(self) -> None:
        for buttons in SAMPLE_BUTTONS.values():
            for entry in buttons:
                assert len(entry) == 3
                label, filename, is_random = entry
                assert isinstance(label, str) and label
                assert isinstance(filename, str) and filename.endswith(".txt")
                assert isinstance(is_random, bool)

    def test_filenames_unique_within_language(self) -> None:
        for lang, buttons in SAMPLE_BUTTONS.items():
            filenames = [b[1] for b in buttons]
            assert len(set(filenames)) == len(filenames), (
                f"{lang} has duplicate filenames"
            )


class TestSampleFilesExist:
    def test_all_referenced_files_exist_and_nonempty(self) -> None:
        import streamlit_app

        samples_dir = Path(streamlit_app.__file__).parent / "samples"
        for lang, buttons in SAMPLE_BUTTONS.items():
            for _, filename, _ in buttons:
                path = samples_dir / lang / filename
                assert path.exists(), f"missing: {path}"
                assert path.stat().st_size > 0, f"empty: {path}"


class TestLoadSample:
    def test_returns_content_for_existing_file(self) -> None:
        content = _load_sample("a", "random.txt")
        assert content != ""
        assert isinstance(content, str)

    def test_returns_empty_for_missing_file(self) -> None:
        assert _load_sample("a", "definitely_does_not_exist.txt") == ""

    def test_returns_empty_for_unknown_lang(self) -> None:
        assert _load_sample("xx", "random.txt") == ""

    def test_strips_trailing_whitespace(self) -> None:
        content = _load_sample("a", "random.txt")
        assert content == content.strip()

    def test_handles_utf8_content(self) -> None:
        content = _load_sample("j", "kokoro.txt")
        assert "私" in content


class TestPickSample:
    def test_non_random_returns_full_content(self) -> None:
        full = _load_sample("a", "gatsby.txt")
        picked = _pick_sample("a", "gatsby.txt", is_random=False)
        assert picked == full

    def test_random_returns_one_line_from_pool(self) -> None:
        full = _load_sample("a", "random.txt")
        lines = [line for line in full.splitlines() if line.strip()]
        picked = _pick_sample("a", "random.txt", is_random=True)
        assert picked in lines

    def test_random_can_pick_multiple_distinct_lines(self) -> None:
        seen = set()
        for _ in range(50):
            seen.add(_pick_sample("a", "random.txt", is_random=True))
        full = _load_sample("a", "random.txt")
        n_lines = len([line for line in full.splitlines() if line.strip()])
        assert len(seen) >= min(5, n_lines // 2)

    def test_missing_file_returns_empty(self) -> None:
        assert _pick_sample("a", "nonexistent.txt", is_random=False) == ""
        assert _pick_sample("a", "nonexistent.txt", is_random=True) == ""

    def test_random_avoids_consecutive_repeats(self) -> None:
        for k in list(st.session_state):
            if isinstance(k, str) and k.startswith("_last_random_"):
                del st.session_state[k]
        prev = None
        for _ in range(20):
            pick = _pick_sample("a", "random.txt", is_random=True)
            if prev is not None:
                assert pick != prev, f"consecutive repeat: {pick!r}"
            prev = pick


class TestRenderSampleButtons:
    @staticmethod
    def _reset_mocks() -> None:
        st.button.reset_mock()  # ty: ignore[unresolved-attribute]
        st.button.return_value = False  # ty: ignore[unresolved-attribute]
        st.columns.reset_mock()  # ty: ignore[unresolved-attribute]
        st.rerun.reset_mock()  # ty: ignore[unresolved-attribute]
        st.session_state.pop("text_input", None)

    def test_renders_nothing_for_unknown_language(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("xx")
        st.columns.assert_not_called()  # ty: ignore[unresolved-attribute]
        st.button.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_creates_one_column_per_button(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        st.columns.assert_called_once_with(3)  # ty: ignore[unresolved-attribute]

    def test_renders_one_button_per_entry(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        assert st.button.call_count == 3  # ty: ignore[unresolved-attribute]

    def test_button_keys_are_language_and_filename_scoped(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        keys = [call.kwargs.get("key") for call in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
        assert "sample_a_random" in keys
        assert "sample_a_gatsby" in keys
        assert "sample_a_frankenstein" in keys

    def test_button_uses_on_click_callback(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        for call in st.button.call_args_list:  # ty: ignore[unresolved-attribute]
            assert call.kwargs.get("on_click") is _set_text_from_sample

    def test_button_args_match_entry(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        seen_args = [call.kwargs.get("args") for call in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
        expected = [
            ("a", fname, is_random) for _, fname, is_random in SAMPLE_BUTTONS["a"]
        ]
        assert seen_args == expected

    def test_renders_for_non_english_language(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("j")
        st.columns.assert_called_once_with(3)  # ty: ignore[unresolved-attribute]
        assert st.button.call_count == 3  # ty: ignore[unresolved-attribute]

    def test_button_labels_use_localized_text(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("j")
        labels = [call.args[0] for call in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
        assert any("こころ" in label for label in labels)


class TestSetTextFromSample:
    @staticmethod
    def _reset() -> None:
        st.session_state.pop("text_input", None)

    def test_non_random_loads_full_content(self) -> None:
        self._reset()
        _set_text_from_sample("a", "gatsby.txt", False)
        full = _load_sample("a", "gatsby.txt")
        assert st.session_state["text_input"] == full

    def test_random_loads_one_line_from_pool(self) -> None:
        self._reset()
        _set_text_from_sample("a", "random.txt", True)
        full = _load_sample("a", "random.txt")
        lines = [line for line in full.splitlines() if line.strip()]
        assert st.session_state["text_input"] in lines

    def test_missing_file_sets_empty_string(self) -> None:
        self._reset()
        _set_text_from_sample("a", "nonexistent.txt", False)
        assert st.session_state["text_input"] == ""

    def test_works_for_non_english(self) -> None:
        self._reset()
        _set_text_from_sample("j", "kokoro.txt", False)
        assert "私" in st.session_state["text_input"]


class TestTextDigest:
    def test_is_deterministic(self) -> None:
        assert _text_digest("hello world") == _text_digest("hello world")

    def test_differs_for_different_text(self) -> None:
        assert _text_digest("hello") != _text_digest("world")

    def test_is_16_hex_chars(self) -> None:
        digest = _text_digest("hello")
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)

    def test_known_value_is_stable_across_processes(self) -> None:
        # Hard-coded sha1("hello")[:16]; identical in every interpreter process,
        # unlike the previous hash(text). Catches any future digest change.
        assert _text_digest("hello") == "aaf4c61ddcc5e8a2"

    def test_handles_unicode(self) -> None:
        assert len(_text_digest("こんにちは")) == 16


class TestNextAudioSeq:
    def test_starts_at_one_when_unset(self) -> None:
        st.session_state.pop("_audio_seq", None)
        assert _next_audio_seq() == 1

    def test_increments_monotonically(self) -> None:
        st.session_state.pop("_audio_seq", None)
        assert _next_audio_seq() == 1
        assert _next_audio_seq() == 2
        assert _next_audio_seq() == 3


class TestThemeConfig:
    """Guard the shipped .streamlit/config.toml theme: a malformed file or a
    missing mode block breaks app startup or the light/dark toggle."""

    @staticmethod
    def _load_theme() -> dict[str, Any]:
        import tomllib

        import streamlit_app

        path = Path(streamlit_app.__file__).parent / ".streamlit" / "config.toml"
        assert path.exists(), f"missing: {path}"
        with path.open("rb") as f:
            config = tomllib.load(f)
        assert "theme" in config, "config.toml has no [theme] section"
        return config["theme"]

    def test_defines_primary_color(self) -> None:
        theme = self._load_theme()
        assert str(theme.get("primaryColor", "")).startswith("#")

    def test_defines_both_light_and_dark_modes(self) -> None:
        # Both blocks must exist for the toolbar light/dark toggle to appear.
        theme = self._load_theme()
        assert "light" in theme
        assert "dark" in theme

    def test_each_mode_defines_background_and_text(self) -> None:
        theme = self._load_theme()
        for mode in ("light", "dark"):
            block = theme[mode]
            assert str(block.get("backgroundColor", "")).startswith("#")
            assert str(block.get("textColor", "")).startswith("#")

    def test_each_mode_defines_caption_band_colors(self) -> None:
        # _render_length_caption emits :red[]/:orange[]/:green[] text, so both
        # modes must define those semantic colors.
        theme = self._load_theme()
        for mode in ("light", "dark"):
            block = theme[mode]
            for band in ("redColor", "orangeColor", "greenColor"):
                assert band in block, f"{mode} mode missing {band}"

    def test_h1_heading_is_extrabold(self) -> None:
        # Guard the deliberate refinement: h1 must be heavier than Streamlit's
        # default heading weight (700) — a [700, ...] list would be a silent no-op —
        # and the weight must actually be loaded in the body font to render.
        theme = self._load_theme()
        weights = theme.get("headingFontWeights")
        assert isinstance(weights, list) and len(weights) == 6
        assert weights[0] >= 800, "h1 should be extrabold, not the 700 default"
        assert "800" in str(theme.get("font", "")), "body font must load weight 800"


class TestProjectMetadata:
    """Keep the project description consistent across pyproject.toml (the source
    of truth), the README tagline, and the CLAUDE.md Project Overview."""

    EXPECTED_DESCRIPTION: str = (
        "Streamlit application for generating multilingual speech using "
        "Hexgrad Kokoro on Apple Silicon with MLX."
    )

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    def _pyproject_description(self) -> str:
        import tomllib

        path = self._repo_root() / "pyproject.toml"
        with path.open("rb") as f:
            return tomllib.load(f)["project"]["description"]

    def test_pyproject_description(self) -> None:
        assert self._pyproject_description() == self.EXPECTED_DESCRIPTION

    @pytest.mark.parametrize("filename", ["README.md", "CLAUDE.md"])
    def test_description_in_sync_with_docs(self, filename: str) -> None:
        import re

        text = (self._repo_root() / filename).read_text(encoding="utf-8")
        # The tagline is the line describing the app; strip markdown links so the
        # hyperlinked "[Hexgrad Kokoro](url)" compares equal to the plain text.
        line = next(ln for ln in text.splitlines() if "multilingual speech" in ln)
        unlinked = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line).strip()
        assert unlinked == self._pyproject_description()


class TestLicensing:
    """Keep the MIT license declared consistently across the LICENSE file,
    pyproject.toml (the source of truth), and the README."""

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    def _pyproject(self) -> dict[str, Any]:
        import tomllib

        with (self._repo_root() / "pyproject.toml").open("rb") as f:
            return tomllib.load(f)

    def test_license_file_is_mit(self) -> None:
        text = (self._repo_root() / "LICENSE").read_text(encoding="utf-8")
        # A real MIT body, not just a header line.
        assert text.startswith("MIT License")
        assert "Permission is hereby granted, free of charge" in text
        assert "WITHOUT WARRANTY OF ANY KIND" in text

    def test_pyproject_declares_mit(self) -> None:
        project = self._pyproject()["project"]
        assert project["license"] == "MIT"
        assert "LICENSE" in project["license-files"]

    def test_readme_documents_license(self) -> None:
        readme = (self._repo_root() / "README.md").read_text(encoding="utf-8")
        assert "## License" in readme
        assert "[MIT](LICENSE)" in readme

    def test_license_consistent_across_sources(self) -> None:
        # pyproject's SPDX id is the source of truth; the LICENSE header and the
        # README badge must agree, so changing the license can't silently desync
        # the three places it is declared.
        spdx = self._pyproject()["project"]["license"]
        license_text = (self._repo_root() / "LICENSE").read_text(encoding="utf-8")
        readme = (self._repo_root() / "README.md").read_text(encoding="utf-8")
        assert license_text.startswith(f"{spdx} License")
        assert f"[{spdx}](LICENSE)" in readme


class TestReleaseWorkflow:
    """Guard the assumption the release workflow (.github/workflows/release.yml)
    relies on when it checks tag/version drift."""

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    def test_release_workflow_present(self) -> None:
        path = self._repo_root() / ".github" / "workflows" / "release.yml"
        assert path.is_file()

    def test_pyproject_version_is_grep_extractable(self) -> None:
        import re
        import tomllib

        # release.yml extracts the version with: grep -m1 -E '^version = "' | sed.
        # Guard that pyproject keeps exactly one top-level `version = "X"` line and
        # that the shell-extracted value matches tomllib's parsed value, so the
        # workflow's drift check stays valid if pyproject is ever reformatted.
        root = self._repo_root()
        lines = (root / "pyproject.toml").read_text(encoding="utf-8").splitlines()
        version_lines = [ln for ln in lines if re.match(r'^version = "', ln)]
        assert len(version_lines) == 1, "expected exactly one top-level version line"
        match = re.match(r'^version = "([^"]+)"', version_lines[0])
        assert match is not None
        with (root / "pyproject.toml").open("rb") as f:
            parsed = tomllib.load(f)["project"]["version"]
        assert match.group(1) == parsed
