from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import streamlit as st
from conftest import DECORATOR_KWARGS

from streamlit_app import (
    _PHONEME_MULTIPLIERS,
    AUDIO_CACHE_LIMIT,
    DEFAULT_SPEED_INDEX,
    ESPEAK_LANGUAGES,
    LANGUAGES,
    PRONUNCIATION_TIPS,
    REPO_ID,
    SAMPLE_BUTTONS,
    SAMPLE_RATE,
    SPEED_OPTIONS,
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
    _tokenize_error_message,
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

# A cached clip is encoded WAV, not samples: both the player and the download
# want bytes, and st.audio would otherwise re-encode the array on every render.
_WAV = _audio_to_wav_bytes(np.ones(100, dtype=np.float32))
# A second, distinguishable clip, for tests that must tell two cache entries
# apart (recency by `seq`) now that the payload is opaque bytes.
_WAV_ALT = _audio_to_wav_bytes(np.zeros(100, dtype=np.float32))

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

    def test_pins_voice_repo_to_our_snapshot(self) -> None:
        # Without this, mlx-audio falls back to its hard-coded
        # `prince-canuma/Kokoro-82M` for voice tensors, re-downloading voices
        # already in our snapshot and breaking the offline guarantee.
        assert load_pipeline().repo_id == REPO_ID


class TestCacheSpinners:
    """Every `@st.cache_resource` site must choose its own `show_spinner`.

    The default renders ``Running `load_pipeline()`.`` — a Python identifier —
    and it does so at exactly the three moments the app makes someone wait.
    Nothing else can catch a dropped setting: the conftest shim has to swallow
    decorator kwargs to hand the function back, so it records them instead.
    """

    _RESOURCES: ClassVar[dict[str, dict[str, Any]]] = DECORATOR_KWARGS["cache_resource"]

    def test_covers_every_cached_resource(self) -> None:
        # A new @st.cache_resource site fails here until it is considered.
        assert set(self._RESOURCES) == {
            "ensure_repo_downloaded",
            "load_pipeline",
            "load_tokenizer",
        }

    def test_every_cached_resource_sets_show_spinner(self) -> None:
        for name, kwargs in self._RESOURCES.items():
            assert "show_spinner" in kwargs, (
                f"{name} takes the default spinner, which names the function"
            )

    def test_download_guard_suppresses_the_spinner(self) -> None:
        # False rather than a sentence: ensure_repo_downloaded renders its own
        # progress message, so a second one would stack on top of it.
        assert self._RESOURCES["ensure_repo_downloaded"]["show_spinner"] is False

    def test_waits_the_user_sees_are_worded_for_the_user(self) -> None:
        for name in ("load_pipeline", "load_tokenizer"):
            spinner = self._RESOURCES[name]["show_spinner"]
            assert isinstance(spinner, str) and spinner, f"{name} shows no message"
            assert name not in spinner, f"{name} leaks its identifier into the UI"


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

    def test_returns_voice_result(self) -> None:
        model = self._model()
        result = generate_one("hi", "af_heart", model, 1.0, "a")
        assert result["voice"] == "af_heart"

    def test_does_not_build_a_tokenizer(self) -> None:
        # generate_one must not run G2P. mlx-audio's pipeline runs its own inside
        # `generate`, so a pass here is a second one, and on a Play with no prior
        # Tokenize it builds this app's whole G2P stack on the click path
        # (measured ~1.4 s) to fill a field that was never read. Restoring the
        # `tokenize_text` call fails this.
        from misaki import en

        en.G2P.reset_mock()  # ty: ignore[unresolved-attribute]
        generate_one("hi", "af_heart", self._model(), 1.0, "a")
        en.G2P.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_audio_concatenated(self) -> None:
        model = MagicMock()
        c1, c2 = MagicMock(), MagicMock()
        c1.audio = np.ones(50, dtype=np.float32)
        c2.audio = np.zeros(30, dtype=np.float32)
        model.generate.return_value = [c1, c2]
        result = generate_one("hi", "af_heart", model, 1.0, "a")
        # The chunks are concatenated then encoded once, so assert on the WAV
        # the card will actually play rather than on an intermediate array.
        assert result["wav"] == _audio_to_wav_bytes(
            np.concatenate([c1.audio, c2.audio])
        )

    def test_passes_speed_and_lang(self) -> None:
        model = self._model()
        generate_one("hi", "af_heart", model, 1.5, "b")
        model.generate.assert_called_with(
            text="hi", voice="af_heart", speed=1.5, lang_code="b"
        )

    def test_raises_on_empty_chunks(self) -> None:
        # Locks the end-to-end "No audio generated" propagation that the card's
        # ValueError -> st.error branch depends on (match pins the exact message,
        # so a refactor letting np.concatenate([]) raise instead would fail).
        model = MagicMock()
        chunk = MagicMock()
        chunk.audio = None
        model.generate.return_value = [chunk]
        with pytest.raises(ValueError, match="No audio generated"):
            generate_one("hi", "af_heart", model, 1.0, "a")


class TestSplitVoicesForDisplay:
    LONG: ClassVar[list[str]] = [f"af_v{i}" for i in range(10)]  # 10 voices

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
                "wav": _WAV,
                "voice": f"v{i}",
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
                "wav": _WAV,
                "voice": f"v{i}",
                "seq": n - i,  # last-inserted v{n-1} has the lowest seq (=1)
            }
        _evict_old_audio()
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        assert f"audio:v{n - 1}:a:1.0:{n - 1}" not in st.session_state  # lowest seq
        assert "audio:v0:a:1.0:0" in st.session_state  # highest seq survives
        self._clear_audio_cache()

    def test_cap_still_holds_when_every_key_is_protected(self) -> None:
        # Reachable only once some language ships more than AUDIO_CACHE_LIMIT
        # voices, because a card protects exactly one key: American English has
        # 20 against a limit of 20, so today there is always one unprotected key
        # to drop. That is zero margin, and while `protect` was an absolute veto
        # a single voice added upstream would have left nothing evictable and
        # turned the documented bound into one clip per voice. The bound wins.
        self._clear_audio_cache()
        self._fill_cache(AUDIO_CACHE_LIMIT + 3)
        everything = frozenset(
            k for k in st.session_state if isinstance(k, str) and k.startswith("audio:")
        )
        _evict_old_audio(protect=everything)
        assert self._count_audio_keys() == AUDIO_CACHE_LIMIT
        # Oldest-first still picks which protected keys give way.
        for i in range(3):
            assert f"audio:v{i}:a:1.0:{i}" not in st.session_state
        assert (
            f"audio:v{AUDIO_CACHE_LIMIT}:a:1.0:{AUDIO_CACHE_LIMIT}" in st.session_state
        )
        self._clear_audio_cache()

    def test_protected_key_survives_even_when_lowest_seq(self) -> None:
        # A key a card is displaying (passed in `protect`) yields to any
        # unprotected key, however much newer — guarding the fragment scenario
        # where one card's Play would orphan a sibling's on-screen player. It is
        # a preference, not immunity: see the every-key-protected case above,
        # where the cap wins and the oldest displayed keys do go.
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
            "wav": _WAV,
            "voice": "af_heart",
        }
        st.session_state[key] = payload
        assert _find_stale_cached_audio("af_heart", "hello", "a") is payload
        self._clear_audio_cache()

    def test_does_not_find_audio_for_different_text(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        assert _find_stale_cached_audio("af_heart", "world", "a") is None
        self._clear_audio_cache()

    def test_does_not_find_audio_for_different_voice(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        assert _find_stale_cached_audio("af_bella", "hello", "a") is None
        self._clear_audio_cache()

    def test_does_not_find_audio_for_different_lang(self) -> None:
        self._clear_audio_cache()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "wav": _WAV,
            "voice": "af_heart",
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
            "wav": _WAV,
            "voice": "af_heart",
            "seq": 2,
        }
        st.session_state[key_07] = {
            "wav": _WAV_ALT,
            "voice": "af_heart",
            "seq": 1,
        }
        result = _find_stale_cached_audio("af_heart", "hello", "a")
        assert result is not None
        assert result["wav"] == _WAV  # higher-seq 1.5 entry, not last-inserted
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
            "wav": _WAV,
            "voice": "af_heart",
            "seq": 2,
        }
        st.session_state[key_15] = {
            "wav": _WAV_ALT,
            "voice": "af_heart",
            "seq": 1,
        }
        result = _find_stale_cached_audio("af_heart", "hello", "a")
        assert result is not None
        assert result["wav"] == _WAV  # higher-seq 0.7 entry, not last-inserted
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
        # Rendering a card writes both of these, so clearing them here is what
        # keeps the class order-independent: a test that reads the protect map
        # must see what its own run registered, not what a predecessor left.
        st.session_state.pop("_displayed_card_keys", None)
        st.session_state.pop("_audio_seq", None)

    def test_opens_its_own_containers_in_order(self) -> None:
        # Opened inside the fragment, not passed in: a shared container would
        # make every card one fragment instance. The whole *sequence* is
        # asserted rather than `assert_any_call(border=True)`, because `st`
        # here is one MagicMock: `st.container(border=True)` and the bare
        # `st.container()` return the identical object, so nothing downstream
        # can tell which is which. Call order is the only distinguishing
        # signal, which makes this equality the guard for two claims at once —
        # that the border lands on the card and not the title slot, and that
        # the title slot is reserved at all (its content is written last, see
        # test_badge_appears_on_the_run_that_generates). Both mutations pass
        # under `assert_any_call`.
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        assert st.container.call_args_list == [  # ty: ignore[unresolved-attribute]
            call(border=True),
            call(),
        ]

    def test_title_is_written_into_the_reserved_slot(self) -> None:
        # The sequence test above cannot see this one: dedenting the title out
        # of `with title_slot:` so it renders at the bottom of the card changes
        # no call argument and no call order, and the shared `st` MagicMock
        # hands back the same object for both containers. The only observable
        # difference is *which container is open* when `st.markdown` runs, so
        # this gives the containers a real enter/exit that pushes and pops a
        # stack and captures that frame as the title is written.
        self._reset_mocks()
        stack: list[MagicMock] = []
        opened: list[MagicMock] = []
        frames: list[tuple[str, tuple[MagicMock, ...]]] = []

        def _make_container(**_kw: Any) -> MagicMock:
            # Named, so a failure reads `(card,) != (card, title_slot)` rather
            # than a pair of MagicMock ids.
            names = ("card", "title_slot")
            n = len(opened)
            cm = MagicMock(name=names[n] if n < len(names) else f"container{n}")

            def _enter() -> MagicMock:
                stack.append(cm)
                return cm

            def _exit(*_a: Any) -> bool:
                stack.pop()
                return False

            cm.__enter__.side_effect = _enter
            cm.__exit__.side_effect = _exit
            opened.append(cm)
            return cm

        def _record(body: str, **_kw: Any) -> None:
            frames.append((body, tuple(stack)))

        with (
            patch.object(st, "container", side_effect=_make_container),
            patch.object(st, "markdown", side_effect=_record),
        ):
            render_voice_card("af_heart", "hello", "a")

        card, title_slot = opened
        assert frames == [("**Heart (female) — A**", (card, title_slot))]

    def test_renders_formatted_title(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # ty: ignore[unresolved-attribute]

    def test_badge_when_cached_at_current_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            "**Heart (female) — A** :green-badge[Cached]"
        )

    def test_badge_when_cached_at_other_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 0.7, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            "**Heart (female) — A** :green-badge[Cached]"
        )

    def test_no_badge_when_cache_for_different_text(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "different_text", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with("**Heart (female) — A**")  # ty: ignore[unresolved-attribute]

    def test_no_badge_when_cache_for_different_voice(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_bella", "hello", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_bella",
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
        current = _cache_key("af_heart", "hello", 1.0, "a")  # conftest speed = 1.0
        st.session_state[current] = {
            "wav": _WAV,
            "voice": "af_heart",
            "seq": 1,
        }
        newer_other_speed = _cache_key("af_heart", "hello", 0.7, "a")
        st.session_state[newer_other_speed] = {
            "wav": _WAV,
            "voice": "af_heart",
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
        stale = _cache_key("af_heart", "hello", 0.7, "a")  # not the current 1.0
        st.session_state[stale] = {
            "wav": _WAV,
            "voice": "af_heart",
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
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_called_once()  # ty: ignore[unresolved-attribute]
        del st.session_state[key]

    def test_no_audio_when_not_cached(self) -> None:
        self._reset_mocks()
        render_voice_card("af_heart", "never_cached_for_this_test", "a")
        st.audio.assert_not_called()  # ty: ignore[unresolved-attribute]

    def test_audio_is_played_from_the_cached_wav(self) -> None:
        # Passing the encoded clip rather than samples is what keeps st.audio
        # from re-encoding the array on every render; the rate is in the header.
        self._reset_mocks()
        key = _cache_key("af_heart", "hello", 1.0, "a")
        st.session_state[key] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        args, kwargs = st.audio.call_args  # ty: ignore[unresolved-attribute]
        assert args[0] == _WAV
        assert kwargs["format"] == "audio/wav"
        assert "sample_rate" not in kwargs
        del st.session_state[key]

    def test_renders_stale_audio_with_caption_when_only_other_speed_cached(
        self,
    ) -> None:
        self._reset_mocks()
        st.caption.reset_mock()  # ty: ignore[unresolved-attribute]
        # Cache key uses speed=0.7, but the conftest selectbox mock returns 1.0
        old_key = _cache_key("af_heart", "hello", 0.7, "a")
        st.session_state[old_key] = {
            "wav": _WAV,
            "voice": "af_heart",
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
            "wav": _WAV,
            "voice": "af_heart",
        }
        st.session_state[stale_key] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        st.audio.assert_called_once()  # ty: ignore[unresolved-attribute]
        st.caption.assert_not_called()  # ty: ignore[unresolved-attribute]
        del st.session_state[fresh_key]
        del st.session_state[stale_key]

    def test_speed_selectbox_persists_while_unrendered(self) -> None:
        """The speed must outlive a card that a run did not draw.

        Cards inside the collapsed "Show all voices" expander are not drawn, and
        Streamlit discards the state of any widget a run skipped. Losing it sends
        the card back to 1.0x on reopen and — since `_cache_key` includes the
        speed — demotes its clip to a stale preview. Only "session" holds the
        value; "page" was measured not to (its scope is page navigation).
        `tests_integration` exercises the real collapse/reopen cycle.
        """
        self._reset_mocks()
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.selectbox.call_args[1]  # ty: ignore[unresolved-attribute]
        assert kwargs["persist_state"] == "session"

    def test_download_button_rendered_when_fresh_audio_cached(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        st.download_button.assert_called_once()  # ty: ignore[unresolved-attribute]

    def test_download_button_file_name_includes_voice_and_speed(self) -> None:
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.download_button.call_args[1]  # ty: ignore[unresolved-attribute]
        assert kwargs["file_name"] == "af_heart_1.0x.wav"
        assert kwargs["mime"] == "audio/wav"
        assert kwargs["key"] == "download_af_heart"

    def test_download_serves_the_cached_wav_without_re_encoding(self) -> None:
        # The bytes come straight from the cache entry, so neither the player
        # nor the button re-runs sf.write on a rerun that merely draws the card.
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.download_button.call_args[1]  # ty: ignore[unresolved-attribute]
        assert kwargs["data"] is _WAV
        assert kwargs["data"][:4] == b"RIFF"

    def test_download_click_does_not_rerun_the_fragment(self) -> None:
        # Nothing on the page depends on the click, and the default "rerun"
        # re-renders the card while the browser is still fetching the file.
        self._reset_mocks()
        st.session_state[_cache_key("af_heart", "hello", 1.0, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
        }
        render_voice_card("af_heart", "hello", "a")
        kwargs = st.download_button.call_args[1]  # ty: ignore[unresolved-attribute]
        assert kwargs["on_click"] == "ignore"

    def test_no_download_button_for_stale_audio(self) -> None:
        self._reset_mocks()
        # Cached at speed 0.7 — stale relative to default speed 1.0
        st.session_state[_cache_key("af_heart", "hello", 0.7, "a")] = {
            "wav": _WAV,
            "voice": "af_heart",
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
            "wav": _WAV,
            "voice": "af_heart",
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

    def test_badge_appears_on_the_run_that_generates(self) -> None:
        # The title is written into a reserved slot after the Play handler
        # precisely so this holds. Emitted in place, the badge was computed
        # before generation, which left the one card that had just produced a
        # clip as the only card without a badge until some unrelated rerun
        # corrected it. Move the title write back above `if play_clicked:` and
        # this fails.
        self._reset_mocks()
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        fake_result = {"wav": _WAV, "voice": "af_heart"}
        with (
            patch("streamlit_app.load_pipeline"),
            patch("streamlit_app.generate_one", return_value=fake_result),
        ):
            render_voice_card("af_heart", "hello", "a")
        st.markdown.assert_called_once_with(  # ty: ignore[unresolved-attribute]
            "**Heart (female) — A** :green-badge[Cached]"
        )
        del st.session_state[_cache_key("af_heart", "hello", 1.0, "a")]

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
            "wav": _WAV,
            "voice": "af_heart",
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
        st.button.return_value = True  # ty: ignore[unresolved-attribute]
        stale = _cache_key("af_heart", "hello", 0.7, "a")  # prior take, other speed
        st.session_state[stale] = {
            "wav": _WAV,
            "voice": "af_heart",
            "seq": 1,
        }
        current = _cache_key("af_heart", "hello", 1.0, "a")  # conftest speed = 1.0
        fake_result = {
            "wav": _WAV,
            "voice": "af_heart",
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
            "wav": _WAV,
            "voice": "af_heart",
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
        st.expander.assert_called_once_with("Phoneme tokens", expanded=False)  # ty: ignore[unresolved-attribute]
        st.code.assert_called_once_with("hɛlˈoʊ")  # ty: ignore[unresolved-attribute]

    def test_expanded_flag_forwarded(self) -> None:
        st.expander.reset_mock()  # ty: ignore[unresolved-attribute]
        render_phonemes("x", expanded=True)
        st.expander.assert_called_once_with("Phoneme tokens", expanded=True)  # ty: ignore[unresolved-attribute]


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
        st.expander.assert_called_once_with("Phoneme tokens", expanded=True)  # ty: ignore[unresolved-attribute]
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
            random_count = sum(1 for b in buttons if b.is_random)
            assert random_count == 1, f"{lang} has {random_count} random buttons"

    def test_button_tuples_have_correct_shape(self) -> None:
        for buttons in SAMPLE_BUTTONS.values():
            for entry in buttons:
                assert len(entry) == 4
                label, icon, filename, is_random = entry
                assert isinstance(label, str) and label
                assert isinstance(icon, str) and icon
                assert isinstance(filename, str) and filename.endswith(".txt")
                assert isinstance(is_random, bool)

    def test_labels_carry_no_icon_of_their_own(self) -> None:
        # The icon belongs in st.button's `icon=`, not in the accessible name.
        for lang, buttons in SAMPLE_BUTTONS.items():
            for entry in buttons:
                assert entry.icon not in entry.label, (
                    f"{lang} label {entry.label!r} still embeds its icon"
                )

    def test_filenames_unique_within_language(self) -> None:
        for lang, buttons in SAMPLE_BUTTONS.items():
            filenames = [b.filename for b in buttons]
            assert len(set(filenames)) == len(filenames), (
                f"{lang} has duplicate filenames"
            )


class TestSampleFilesExist:
    def test_all_referenced_files_exist_and_nonempty(self) -> None:
        import streamlit_app

        samples_dir = Path(streamlit_app.__file__).parent / "samples"
        for lang, buttons in SAMPLE_BUTTONS.items():
            for entry in buttons:
                path = samples_dir / lang / entry.filename
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
        # Columns, not a horizontal container: the latter sizes children by
        # their label width, giving a visibly ragged row.
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
        keys = [c.kwargs.get("key") for c in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
        assert "sample_a_random" in keys
        assert "sample_a_gatsby" in keys
        assert "sample_a_frankenstein" in keys

    def test_button_uses_on_click_callback(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        for c in st.button.call_args_list:  # ty: ignore[unresolved-attribute]
            assert c.kwargs.get("on_click") is _set_text_from_sample

    def test_button_args_match_entry(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        seen_args = [c.kwargs.get("args") for c in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
        expected = [("a", b.filename, b.is_random) for b in SAMPLE_BUTTONS["a"]]
        assert seen_args == expected

    def test_button_labels_wrap(self) -> None:
        # Streamlit's default no longer wraps a button placed directly in a
        # column, so long labels ("Pride & Prejudice") were cut to "Pride & …".
        self._reset_mocks()
        _render_sample_buttons("b")
        wraps = [c.kwargs.get("wrap") for c in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
        assert wraps == [True] * len(SAMPLE_BUTTONS["b"])

    def test_button_icon_is_passed_separately_from_the_label(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("a")
        calls = st.button.call_args_list  # ty: ignore[unresolved-attribute]
        seen = [(c.args[0], c.kwargs.get("icon")) for c in calls]
        assert seen == [(b.label, b.icon) for b in SAMPLE_BUTTONS["a"]]

    def test_renders_for_non_english_language(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("j")
        st.columns.assert_called_once_with(3)  # ty: ignore[unresolved-attribute]
        assert st.button.call_count == 3  # ty: ignore[unresolved-attribute]

    def test_button_labels_use_localized_text(self) -> None:
        self._reset_mocks()
        _render_sample_buttons("j")
        labels = [c.args[0] for c in st.button.call_args_list]  # ty: ignore[unresolved-attribute]
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


class TestStreamlitConfig:
    """Guard the shipped .streamlit/config.toml — both the disabled file watcher
    and the deliberately absent [theme] section. This docstring is the one
    authoritative statement of the theming decision; config.toml and CLAUDE.md
    point here rather than restating it."""

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    def _load_config(self) -> dict[str, Any]:
        import tomllib

        path = self._repo_root() / ".streamlit" / "config.toml"
        assert path.exists(), f"missing: {path}"
        with path.open("rb") as f:
            return tomllib.load(f)

    def test_file_watcher_stays_disabled(self) -> None:
        # README's troubleshooting table tells users edits don't auto-reload
        # because of this; it is a server setting, not a theme one.
        config = self._load_config()
        assert config.get("server", {}).get("fileWatcherType") == "none"

    def test_usage_stats_stay_off(self) -> None:
        # Streamlit's frontend posts session telemetry to data.streamlit.io
        # unless this is false, and the option defaults to true — so the key
        # must be present, not merely absent. README's "no network calls" line
        # and the "Runs offline" badge are false without it.
        config = self._load_config()
        assert config.get("browser", {}).get("gatherUsageStats") is False

    def test_ships_no_custom_theme(self) -> None:
        # The app renders in Streamlit's stock light and dark themes, whose
        # defaults already supply both modes, the toolbar toggle, and the
        # per-mode red/orange/green that _render_length_caption's bands use.
        #
        # This asserts absence deliberately, in the same spirit as
        # TestReleaseWorkflow.test_tag_triggered_release_workflow_stays_retired:
        # the decision is that no theme ships, and a *partial* theme is the way
        # that decision breaks silently — Streamlit keeps the light/dark toggle
        # only when a custom theme defines both [theme.light] and [theme.dark],
        # so a bare [theme] block pins the app to one mode with no error. Any
        # deliberate return to theming updates this test, README's feature list,
        # and CLAUDE.md's Configuration section together.
        config = self._load_config()
        assert "theme" not in config, (
            "config.toml defines a custom [theme]; the app is meant to render "
            "in Streamlit's default light and dark themes"
        )


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
        # The third-party copyleft acknowledgement must not silently disappear.
        assert "GPLv3" in readme and "LGPL" in readme

    def test_license_consistent_across_sources(self) -> None:
        # pyproject's SPDX id is the source of truth; the LICENSE header, the
        # README badge, and the README section link must all agree, so a license
        # change can't silently desync them. (The header check assumes MIT's
        # "MIT License" header equals the SPDX id; switching to e.g. Apache-2.0
        # trips this intentionally, forcing a re-review of all three sources.)
        spdx = self._pyproject()["project"]["license"]
        license_text = (self._repo_root() / "LICENSE").read_text(encoding="utf-8")
        readme = (self._repo_root() / "README.md").read_text(encoding="utf-8")
        assert license_text.startswith(f"{spdx} License")
        assert f"[{spdx}](LICENSE)" in readme  # ## License section link
        assert f"License-{spdx}-" in readme  # shields.io badge path segment


class TestReleaseWorkflow:
    """Guard the assumptions the auto-release job in .github/workflows/ci.yml
    relies on when it turns a pyproject version bump into a published release."""

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    @classmethod
    def _ci_workflow(cls) -> str:
        return (cls._repo_root() / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )

    @classmethod
    def _release_job(cls) -> str:
        # Everything from the `release:` job key onward. It is the last job in the
        # file, so a plain partition is enough — no YAML parser needed.
        _, _, job = cls._ci_workflow().partition("\n  release:")
        assert job, "expected a `release:` job in ci.yml"
        return job

    @classmethod
    def _release_job_directives(cls) -> str:
        # The release job with comment lines removed. Assert against this, never
        # the raw job: ci.yml explains each of its own policies in prose right
        # above the line that implements it, so a substring check on the raw text
        # is satisfied by the explanation whether or not the policy survives.
        return "\n".join(
            ln
            for ln in cls._release_job().splitlines()
            if not ln.lstrip().startswith("#")
        )

    def test_release_job_present(self) -> None:
        assert self._release_job()

    def test_tag_triggered_release_workflow_stays_retired(self) -> None:
        # release.yml cut a release from a hand-pushed tag without ever consulting
        # CI. That job now lives in ci.yml gated on the check job; reviving the old
        # file would give one version two publish paths that could both cut it.
        path = self._repo_root() / ".github" / "workflows" / "release.yml"
        assert not path.exists(), (
            "release.yml is retired — ci.yml's release job is the only publish path"
        )

    def test_pyproject_version_is_grep_extractable(self) -> None:
        import re
        import tomllib

        # The release job extracts the version with: grep -m1 -E '^version = "' | sed.
        # Guard that pyproject keeps exactly one top-level `version = "X"` line and
        # that the shell-extracted value matches tomllib's parsed value, so the
        # workflow's tag derivation stays valid if pyproject is ever reformatted.
        root = self._repo_root()
        lines = (root / "pyproject.toml").read_text(encoding="utf-8").splitlines()
        version_lines = [ln for ln in lines if re.match(r'^version = "', ln)]
        assert len(version_lines) == 1, "expected exactly one top-level version line"
        match = re.match(r'^version = "([^"]+)"', version_lines[0])
        assert match is not None
        with (root / "pyproject.toml").open("rb") as f:
            parsed = tomllib.load(f)["project"]["version"]
        assert match.group(1) == parsed
        # The version becomes the tag verbatim, and the repo publishes final
        # releases only, so it must be exactly X.Y.Z — a 2-segment or PEP 440
        # prerelease such as "1.0.0rc1" would extract fine and then fail the
        # release job's own shape check.
        assert re.fullmatch(r"\d+\.\d+\.\d+", parsed), (
            "version must be X.Y.Z — the release job errors on any other shape"
        )

    def test_release_job_mirrors_version_guard(self) -> None:
        # Bind this test file to the workflow so the two version guards can't drift
        # apart: if the workflow's grep pattern or X.Y.Z check changes, the
        # pyproject-side check above silently stops mirroring it — fail loudly here.
        job = self._release_job()
        assert "grep -m1 -E '^version = \"'" in job
        assert "^[0-9]+\\.[0-9]+\\.[0-9]+$" in job
        # Comment-stripped: the checkout step's comment says "this job runs with
        # `contents: write`", which satisfies a raw substring check on its own.
        # Without stripping, deleting the permissions block -- or hoisting it to
        # the workflow level, handing a repo-write token to the job that runs
        # third-party code via `uv sync` -- passes this test unnoticed.
        assert "contents: write" in self._release_job_directives(), (
            "the write token must be granted on the release job, not workflow-wide"
        )

    def test_release_is_gated_on_the_check_job_and_main(self) -> None:
        # The reason the release lives in ci.yml at all: a red tree cannot ship,
        # and a fork PR never reaches the job holding the write token.
        job = self._release_job()
        assert "needs: lint-typecheck-test" in job
        assert "github.event_name == 'push'" in job
        assert "github.ref == 'refs/heads/main'" in job

    def test_release_job_checkout_is_sha_pinned(self) -> None:
        # Repo policy: a job holding `contents: write` pins actions/checkout to a
        # SHA, because a floating major tag is repointable upstream and would then
        # execute with a repo-write token. The read-only check job keeps @v7.
        # Comments are stripped first so the note explaining the policy can't
        # satisfy the check on its own.
        import re

        assert re.search(
            r"actions/checkout@[0-9a-f]{40}\b", self._release_job_directives()
        ), "a job with contents: write must pin actions/checkout to a full SHA"

    def test_release_is_drafted_before_it_is_published(self) -> None:
        # Draft-then-publish is deliberate: a draft is invisible to watchers and
        # notifies nobody, so the release only goes public once `gh release create`
        # has returned and the tag it verified is known to exist.
        import re

        directives = self._release_job_directives()
        create = directives.index("gh release create")
        edit = directives.index("gh release edit")
        assert create < edit, "the release must be created before it is published"
        # `--draft` has to be a bare flag on `gh release create`. A plain
        # `"--draft" in job` is also satisfied by the publish step's
        # `--draft=false`, so it cannot detect the flag being dropped from the
        # create -- which would publish the release the instant it is created,
        # silently, since `gh release edit --draft=false` then exits 0 as a no-op.
        assert re.search(r"--draft(?![=\w])", directives[create:edit]), (
            "gh release create must pass a bare --draft"
        )
        assert "--draft=false" in directives[edit:], (
            "gh release edit must clear the draft flag to publish"
        )

    def test_main_runs_are_never_cancelled_mid_publish(self) -> None:
        # cancel-in-progress must stay off for main: cancelling a run between the
        # tag push and the publish would strand a tag with no release behind it.
        assert (
            "cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}"
            in self._ci_workflow()
        )


class TestPythonVersionConsistency:
    """The target Python version is declared in four places that must agree:
    `.python-version` (what uv builds the venv from), `requires-python` (the
    supported floor), `[tool.ty.environment]` (what ty checks against), and the
    CI workflow. Drift here is silent — everything stays green while local dev,
    CI, and the type checker quietly target different versions."""

    EXPECTED: ClassVar[str] = "3.12"

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    def _pyproject(self) -> dict[str, Any]:
        import tomllib

        with (self._repo_root() / "pyproject.toml").open("rb") as f:
            return tomllib.load(f)

    def test_python_version_file(self) -> None:
        path = self._repo_root() / ".python-version"
        assert path.is_file(), ".python-version pins the interpreter for uv sync"
        assert path.read_text(encoding="utf-8").strip() == self.EXPECTED

    def test_requires_python_floor(self) -> None:
        assert (
            self._pyproject()["project"]["requires-python"] == f">={self.EXPECTED}"
        ), "requires-python floor must match the pinned interpreter"

    def test_ty_target(self) -> None:
        ty_env = self._pyproject()["tool"]["ty"]["environment"]
        assert ty_env["python-version"] == self.EXPECTED, (
            "ty must type-check against the declared floor"
        )

    def test_ci_workflow_matches(self) -> None:
        # CI pins the interpreter through setup-uv rather than reading
        # .python-version, so the two are independent declarations that can
        # drift. uv resolves `.python-version` ahead of the UV_PYTHON that
        # setup-uv exports, which means a mismatch would leave CI testing a
        # different version than it claims to.
        workflow = (self._repo_root() / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        assert f'python-version: "{self.EXPECTED}"' in workflow


class TestEspeakVendoring:
    """CI installs no system packages, which is only correct because espeak-ng
    ships prebuilt inside the espeakng-loader wheel and `misaki` points
    phonemizer straight at it. Nothing else in CI would notice if either half
    stopped being true: the unit suite mocks `misaki` wholesale and
    tests_integration/ is excluded by `testpaths`. These are the canary."""

    # espeak-ng stores its data dir in a fixed ~160-byte N_PATH_HOME buffer.
    # Past that the path is silently truncated, espeak falls back to the
    # build-machine path baked into the wheel, fails to find phontab, and calls
    # exit(1) directly -- no traceback, no pytest summary. See CLAUDE.md.
    DATA_PATH_BUDGET: ClassVar[int] = 159

    @staticmethod
    def _repo_root() -> Path:
        import streamlit_app

        return Path(streamlit_app.__file__).parent

    def test_library_ships_inside_the_wheel(self) -> None:
        import espeakng_loader

        lib = Path(espeakng_loader.get_library_path())
        assert lib.is_file(), (
            f"espeakng-loader shipped no library at {lib}. Installing a system "
            "espeak-ng would NOT fix this -- misaki binds phonemizer to this path."
        )

    def test_data_dir_ships_inside_the_wheel(self) -> None:
        import espeakng_loader

        data = Path(espeakng_loader.get_data_path())
        assert data.is_dir()
        # phontab is precisely the file espeak exit(1)s over when truncated.
        assert (data / "phontab").is_file()

    def test_data_path_fits_espeak_fixed_buffer(self) -> None:
        import espeakng_loader

        path = espeakng_loader.get_data_path()
        assert len(path) <= self.DATA_PATH_BUDGET, (
            f"espeak data path is {len(path)} chars, over the ~"
            f"{self.DATA_PATH_BUDGET} budget -- the app will die with exit code 1 "
            f"and no traceback. Move the checkout somewhere shorter: {path}"
        )

    def test_misaki_binds_phonemizer_to_the_bundled_library(self) -> None:
        # conftest replaces `misaki` with a MagicMock, so read the installed
        # source instead of importing. If a misaki upgrade drops these calls,
        # phonemizer falls through to ctypes.util.find_library('espeak-ng') and
        # a system install becomes load-bearing again -- which would silently
        # invalidate the "no system packages" claim in README and CLAUDE.md.
        import espeakng_loader

        site_packages = Path(espeakng_loader.__file__).parent.parent
        src = (site_packages / "misaki" / "espeak.py").read_text(encoding="utf-8")
        assert "EspeakWrapper.set_library(espeakng_loader.get_library_path())" in src
        assert "EspeakWrapper.set_data_path(espeakng_loader.get_data_path())" in src

    def test_ci_installs_no_system_packages(self) -> None:
        # Converts the "don't re-add brew install espeak-ng" decision from
        # prose into enforcement. If a future change genuinely needs a system
        # package, updating this test is the deliberate step that requires.
        workflow = (self._repo_root() / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        # Comments are stripped first: the file explains *why* there is no
        # `brew install espeak-ng`, so a naive substring check would match its
        # own documentation.
        directives = [
            ln for ln in workflow.splitlines() if not ln.lstrip().startswith("#")
        ]
        assert "brew install" not in "\n".join(directives)


class TestTokenizeErrorMessage:
    """The Tokenize handler catches G2P failures so they render in place. The
    missing-UniDic case is called out by name because the dictionary lives in
    the venv and vanishes on any venv rebuild — an error the user can fix."""

    def test_missing_unidic_gives_recovery_command(self) -> None:
        error = RuntimeError(
            "param.cpp(69) [ifs] no such file or directory: "
            "/repo/.venv/lib/python3.12/site-packages/unidic/dicdir/mecabrc"
        )
        message = _tokenize_error_message("j", error)
        assert "unidic download" in message
        assert "~1 GB" in message

    def test_other_japanese_error_falls_through(self) -> None:
        message = _tokenize_error_message("j", ValueError("something else"))
        assert "unidic download" not in message
        assert "something else" in message

    def test_non_japanese_unidic_mention_falls_through(self) -> None:
        # The recovery hint is Japanese-only; another language must not claim a
        # dictionary download would fix it.
        message = _tokenize_error_message("a", RuntimeError("unidic"))
        assert "unidic download" not in message

    def test_generic_error_includes_cause(self) -> None:
        message = _tokenize_error_message("z", RuntimeError("g2p exploded"))
        assert "g2p exploded" in message
