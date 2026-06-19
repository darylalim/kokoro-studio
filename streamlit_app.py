import io
import random
from collections.abc import Generator
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np
import soundfile as sf
import streamlit as st
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_audio.tts.utils import load_model

from voice_grades import VOICE_GRADES, _grade_rank


class VoiceResult(TypedDict):
    audio: np.ndarray
    voice: str
    phonemes: str


class SampleButton(NamedTuple):
    label: str
    filename: str
    is_random: bool


SAMPLE_RATE = 24000
REPO_ID = "mlx-community/Kokoro-82M-bf16"
PRONUNCIATION_TIPS = """\
**Custom pronunciation:** Use `[word](/phonemes/)` syntax, e.g. `[Kokoro](/kˈOkəɹO/)`

**Intonation:** Adjust with punctuation `;` `:` `,` `.` `!` `?` `—` `…` `"` `(` `)` `"` `"`

**Lower stress:** `[word](-1)` or `[word](-2)`

**Raise stress:** `[word](+1)` or `[word](+2)` (works best on less-stressed, usually short words)\
"""

LANGUAGES: dict[str, str] = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Japanese": "j",
    "Brazilian Portuguese": "p",
    "Mandarin Chinese": "z",
}

ESPEAK_LANGUAGES: dict[str, str] = {
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

_GENDER_LABELS: dict[str, str] = {"f": "female", "m": "male"}

SPEED_OPTIONS: list[float] = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
DEFAULT_SPEED_INDEX: int = SPEED_OPTIONS.index(1.0)

AUDIO_CACHE_LIMIT: int = 20

_PHONEME_MULTIPLIERS: dict[str, float] = {
    "a": 0.85, "b": 0.85,
    "e": 0.90, "f": 0.90, "p": 0.90,
    "h": 1.00, "i": 1.00,
    "j": 1.40,
    "z": 2.00,
}

SAMPLE_BUTTONS: dict[str, list[SampleButton]] = {
    "a": [
        SampleButton("🎲 Random Quote", "random.txt", True),
        SampleButton("📕 Gatsby", "gatsby.txt", False),
        SampleButton("📗 Frankenstein", "frankenstein.txt", False),
    ],
    "b": [
        SampleButton("🎲 Random Quote", "random.txt", True),
        SampleButton("📕 Pride & Prejudice", "pride.txt", False),
        SampleButton("📗 Sherlock Holmes", "sherlock.txt", False),
    ],
    "e": [
        SampleButton("🎲 Cita Aleatoria", "random.txt", True),
        SampleButton("📕 Don Quijote", "quijote.txt", False),
        SampleButton("📗 Bécquer", "becquer.txt", False),
    ],
    "f": [
        SampleButton("🎲 Citation", "random.txt", True),
        SampleButton("📕 Misérables", "miserables.txt", False),
        SampleButton("📗 Candide", "candide.txt", False),
    ],
    "h": [
        SampleButton("🎲 लोकोक्ति", "random.txt", True),
        SampleButton("📕 कबीर", "kabir.txt", False),
        SampleButton("📗 रहीम", "rahim.txt", False),
    ],
    "i": [
        SampleButton("🎲 Citazione", "random.txt", True),
        SampleButton("📕 Divina Commedia", "divina.txt", False),
        SampleButton("📗 Promessi Sposi", "promessi.txt", False),
    ],
    "j": [
        SampleButton("🎲 ことわざ", "random.txt", True),
        SampleButton("📕 こころ", "kokoro.txt", False),
        SampleButton("📗 坊っちゃん", "botchan.txt", False),
    ],
    "p": [
        SampleButton("🎲 Citação", "random.txt", True),
        SampleButton("📕 Brás Cubas", "bras_cubas.txt", False),
        SampleButton("📗 Iracema", "iracema.txt", False),
    ],
    "z": [
        SampleButton("🎲 古语", "random.txt", True),
        SampleButton("📕 唐诗", "tangshi.txt", False),
        SampleButton("📗 论语", "lunyu.txt", False),
    ],
}


@st.cache_resource
def ensure_repo_downloaded() -> str:
    try:
        return snapshot_download(REPO_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        with st.spinner("Downloading Kokoro model and voices (one-time, ~160 MB)..."):
            return snapshot_download(REPO_ID)


@st.cache_data
def get_voices(lang_code: str) -> list[str]:
    voices_dir = Path(ensure_repo_downloaded()) / "voices"
    voices = [
        p.stem
        for p in voices_dir.iterdir()
        if p.suffix == ".safetensors" and len(p.stem) >= 2 and p.stem[0] == lang_code
    ]
    return sorted(voices, key=lambda v: (_grade_rank(v), v))


@st.cache_resource
def load_pipeline() -> Any:
    return load_model(REPO_ID)  # type: ignore[arg-type]


def _create_g2p(lang_code: str) -> Any:
    if lang_code in ("a", "b"):
        from misaki import en, espeak as mespeak

        british = lang_code == "b"
        fallback = mespeak.EspeakFallback(british=british)
        return en.G2P(trf=False, british=british, fallback=fallback, unk="")
    if lang_code == "j":
        from misaki import ja

        return ja.JAG2P()
    if lang_code == "z":
        from misaki import zh

        return zh.ZHG2P()
    from misaki import espeak as mespeak

    return mespeak.EspeakG2P(language=ESPEAK_LANGUAGES[lang_code])


@st.cache_resource
def load_tokenizer(lang_code: str) -> Any:
    return _create_g2p(lang_code)


def tokenize_text(text: str, lang_code: str) -> str:
    phonemes, _ = load_tokenizer(lang_code)(text)
    return phonemes or ""


def _format_voice(voice: str) -> str:
    if "_" not in voice:
        return voice
    name = voice.split("_", 1)[1].replace("_", " ").title()
    gender = _GENDER_LABELS.get(voice[1], "")
    grade = VOICE_GRADES.get(voice, "")
    base = f"{name} ({gender})" if gender else name
    return f"{base} — {grade}" if grade else base


def _filter_voices_by_gender(voices: list[str], gender_code: str | None) -> list[str]:
    if gender_code is None:
        return voices
    return [v for v in voices if v[1] == gender_code]


def _gender_code_from_checkboxes(female: bool, male: bool) -> str | None:
    if female == male:
        return None
    return "f" if female else "m"


def _split_voices_for_display(
    voices: list[str], selected: str | None, top_n: int = 6
) -> tuple[list[str], list[str]]:
    top = voices[:top_n]
    tail = voices[top_n:]
    if selected and selected in tail:
        return top + [selected], [v for v in tail if v != selected]
    return top, tail


def _cache_key(voice: str, text: str, speed: float, lang_code: str) -> str:
    return f"audio:{voice}:{lang_code}:{speed}:{hash(text)}"


def _estimate_phonemes(text: str, lang_code: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    mult = _PHONEME_MULTIPLIERS.get(lang_code, 1.0)
    return int(len(stripped) * mult)


def _phoneme_band(n: int) -> tuple[str, str]:
    if n < 20:
        return ("red", "very short")
    if n < 100:
        return ("orange", "short")
    if n < 400:
        return ("green", "ideal")
    if n < 510:
        return ("orange", "long")
    return ("red", "will be chunked")


@st.cache_data
def _load_sample(lang_code: str, filename: str) -> str:
    path = Path(__file__).parent / "samples" / lang_code / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _pick_sample(lang_code: str, filename: str, is_random: bool) -> str:
    content = _load_sample(lang_code, filename)
    if not content:
        return ""
    if not is_random:
        return content
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return ""
    last_key = f"_last_random_{lang_code}_{filename}"
    last = st.session_state.get(last_key)
    pick = random.choice(lines)
    if pick == last and len(lines) > 1:
        pick = random.choice([line for line in lines if line != last])
    st.session_state[last_key] = pick
    return pick


def _set_text_from_sample(lang_code: str, filename: str, is_random: bool) -> None:
    st.session_state["text_input"] = _pick_sample(lang_code, filename, is_random)


def _render_sample_buttons(lang_code: str) -> None:
    buttons = SAMPLE_BUTTONS.get(lang_code, [])
    if not buttons:
        return
    cols = st.columns(len(buttons))
    for col, (label, filename, is_random) in zip(cols, buttons):
        with col:
            st.button(
                label,
                key=f"sample_{lang_code}_{Path(filename).stem}",
                use_container_width=True,
                on_click=_set_text_from_sample,
                args=(lang_code, filename, is_random),
            )


def _find_stale_cached_audio(
    voice: str, text: str, lang_code: str
) -> VoiceResult | None:
    prefix = f"audio:{voice}:{lang_code}:"
    suffix = f":{hash(text)}"
    matches = [
        k
        for k in st.session_state
        if isinstance(k, str) and k.startswith(prefix) and k.endswith(suffix)
    ]
    if not matches:
        return None
    return st.session_state[matches[-1]]


def _evict_old_audio() -> None:
    audio_keys = [
        k for k in st.session_state if isinstance(k, str) and k.startswith("audio:")
    ]
    overflow = len(audio_keys) - AUDIO_CACHE_LIMIT
    if overflow > 0:
        for k in audio_keys[:overflow]:
            del st.session_state[k]


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def generate_speech(
    text: str,
    voice: str,
    pipeline: Any,
    speed: float = 1.0,
    lang_code: str = "a",
) -> Generator[np.ndarray, None, None]:
    generated = False
    for result in pipeline.generate(
        text=text, voice=voice, speed=speed, lang_code=lang_code
    ):
        if result.audio is not None:
            generated = True
            yield np.asarray(result.audio, dtype=np.float32)
    if not generated:
        raise ValueError("No audio generated. Check your input text.")


def generate_one(
    text: str,
    voice: str,
    pipeline: Any,
    speed: float,
    lang_code: str,
) -> VoiceResult:
    phonemes = tokenize_text(text, lang_code)
    with st.status(f"Generating {voice}...", expanded=True) as status:
        chunks = []
        for i, chunk in enumerate(
            generate_speech(text, voice, pipeline, speed=speed, lang_code=lang_code), 1
        ):
            chunks.append(chunk)
            st.write(f"Chunk {i}...")
        status.update(label=f"{voice} complete!", state="complete")
    return {"audio": np.concatenate(chunks), "voice": voice, "phonemes": phonemes}


def render_voice_card(voice: str, text: str, lang_code: str) -> None:
    with st.container(border=True):
        cached = _find_stale_cached_audio(voice, text, lang_code)
        indicator = "🔊 " if cached is not None else ""
        st.markdown(f"**{indicator}{_format_voice(voice)}**")
        speed_col, play_col = st.columns([1, 1])
        with speed_col:
            card_speed = st.selectbox(
                "Speed",
                options=SPEED_OPTIONS,
                index=DEFAULT_SPEED_INDEX,
                key=f"speed_{voice}",
                label_visibility="collapsed",
                format_func=lambda x: f"{x}x",
            )
        with play_col:
            play_clicked = st.button(
                "▶ Play",
                key=f"play_{voice}",
                type="primary",
                use_container_width=True,
                disabled=not text.strip(),
            )
        key = _cache_key(voice, text, card_speed, lang_code)
        if play_clicked:
            try:
                pipeline = load_pipeline()
                st.session_state[key] = generate_one(
                    text, voice, pipeline, card_speed, lang_code
                )
                _evict_old_audio()
            except Exception as e:
                st.exception(e)
        if key in st.session_state:
            audio = st.session_state[key]["audio"]
            st.audio(audio, sample_rate=SAMPLE_RATE)
            st.download_button(
                label="Download",
                data=_audio_to_wav_bytes(audio),
                file_name=f"{voice}_{card_speed}x.wav",
                mime="audio/wav",
                key=f"download_{voice}",
            )
        elif cached is not None:
            st.caption("Click Play to refresh (speed changed)")
            st.audio(cached["audio"], sample_rate=SAMPLE_RATE)


def render_phonemes(phonemes: str, *, expanded: bool = False) -> None:
    with st.expander("Phoneme Tokens", expanded=expanded):
        st.code(phonemes)


def _render_length_caption(text: str, lang_code: str) -> None:
    if not text.strip():
        return
    saved = st.session_state.get("last_phonemes")
    if saved and saved[0] == text and saved[1] == lang_code:
        n = len(saved[2])
        prefix = ""
    else:
        n = _estimate_phonemes(text, lang_code)
        prefix = "~"
    color, label = _phoneme_band(n)
    st.caption(f":{color}[**{prefix}{n} phonemes** — {label}]")


def _render_persistent_phonemes(text: str, lang_code: str) -> None:
    saved = st.session_state.get("last_phonemes")
    if saved and saved[0] == text and saved[1] == lang_code:
        render_phonemes(saved[2], expanded=True)


st.title("Kokoro Studio")

try:
    ensure_repo_downloaded()
except Exception:
    st.error("Could not download the Kokoro model. Connect to the internet and reload the page.")
    st.stop()

language = st.selectbox(
    "Language",
    options=list(LANGUAGES.keys()),
    label_visibility="collapsed",
    key="language",
)
lang_code = LANGUAGES[language]

input_col, controls_col = st.columns(2)

with input_col:
    text_input = st.text_area(
        label="Text",
        placeholder="Start typing here or paste any text you want to turn into lifelike speech...",
        height=500,
        key="text_input",
        label_visibility="collapsed",
    )
    _render_sample_buttons(lang_code)
    tokenize_clicked = st.button("Tokenize", disabled=not text_input.strip())
    if tokenize_clicked:
        st.session_state["last_phonemes"] = (
            text_input,
            lang_code,
            tokenize_text(text_input, lang_code),
        )
    _render_length_caption(text_input, lang_code)
    _render_persistent_phonemes(text_input, lang_code)
    st.markdown("**Note:**")
    st.markdown(PRONUNCIATION_TIPS)

with controls_col:
    gcol_f, gcol_m = st.columns(2)
    with gcol_f:
        female_checked = st.checkbox("Female", value=False, key="female")
    with gcol_m:
        male_checked = st.checkbox("Male", value=False, key="male")
    gender_code = _gender_code_from_checkboxes(female_checked, male_checked)
    voices = _filter_voices_by_gender(get_voices(lang_code), gender_code)
    if voices:
        visible, hidden = _split_voices_for_display(voices, None)
        for voice in visible:
            render_voice_card(voice, text_input, lang_code)
        if hidden:
            with st.expander("Show All Voices"):
                for voice in hidden:
                    render_voice_card(voice, text_input, lang_code)
    else:
        st.info("No voices match this filter.")
