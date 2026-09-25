import hashlib
import io
import random
from collections.abc import Generator
from pathlib import Path
from typing import Any, NamedTuple, NotRequired, TypedDict

import numpy as np
import soundfile as sf
import streamlit as st
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from mlx_audio.tts.utils import load_model

from voice_grades import VOICE_GRADES, _grade_rank


class VoiceResult(TypedDict):
    """A generated clip, cached as encoded WAV rather than as samples.

    Both consumers — the inline player and the Download button — want bytes, and
    `st.audio` re-encodes a float32 array on *every* render (it copies to float64,
    converts to int16, writes a WAV and hashes it). Encoding once at generation
    keeps that off every rerun, and int16 WAV is smaller than the float32 array
    it replaces, so the cache holds `AUDIO_CACHE_LIMIT` clips in less memory.

    There is deliberately **no** `phonemes` field. It existed, was written on
    every generation, and was read by nothing: the card renders `wav` and the
    Download button serves `wav`. Filling it cost a whole second G2P stack on
    the click path — see `generate_one`. Phonemes reach the UI through the
    Tokenize button and `st.session_state["last_phonemes"]` instead.
    """

    wav: bytes
    voice: str
    seq: NotRequired[int]


class SampleButton(NamedTuple):
    """One sample-text button. `icon` is passed to st.button's own `icon=`.

    Keeping the emoji out of `label` is what leaves the label pure text for
    screen readers; concatenating the two made the icon part of the accessible
    name. Emoji rather than Material Symbols here on purpose — 📕/📗 tell the two
    literary excerpts apart by colour, which a monochrome symbol cannot.
    """

    label: str
    icon: str
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

# Fixed width of a voice card's speed box and Play button; the voice name takes
# whatever the row has left. Measured in the stock theme: Play is the tighter of
# the two, clipping to "P…" at 74 px and whole from 78 px ("1.0x" plus its
# chevron holds down to 70 and clips to "1.0" at 66). 96 leaves room for a
# slightly wider font; a test holds it at or above the 78 px floor.
CARD_CONTROL_WIDTH: int = 96

# Sized to balance the composer against the voice list beside it, not to hold
# long text (the box scrolls and has a resize handle). At the old 500 px it was
# mostly empty and pushed Tokenize to the fold at 1440x900; at 300 the composer
# *with* the self-opening phoneme expander ends about where the six one-row
# voice cards do, which is the state a working session sits in. 300 px still
# shows a dozen lines — the whole 100-400 phoneme sweet spot.
TEXT_AREA_HEIGHT: int = 300

_PHONEME_MULTIPLIERS: dict[str, float] = {
    "a": 0.85,
    "b": 0.85,
    "e": 0.90,
    "f": 0.90,
    "p": 0.90,
    "h": 1.00,
    "i": 1.00,
    "j": 1.40,
    "z": 2.00,
}

SAMPLE_BUTTONS: dict[str, list[SampleButton]] = {
    "a": [
        SampleButton("Random quote", "🎲", "random.txt", True),
        SampleButton("Gatsby", "📕", "gatsby.txt", False),
        SampleButton("Frankenstein", "📗", "frankenstein.txt", False),
    ],
    "b": [
        SampleButton("Random quote", "🎲", "random.txt", True),
        SampleButton("Pride & Prejudice", "📕", "pride.txt", False),
        SampleButton("Sherlock Holmes", "📗", "sherlock.txt", False),
    ],
    "e": [
        SampleButton("Cita aleatoria", "🎲", "random.txt", True),
        SampleButton("Don Quijote", "📕", "quijote.txt", False),
        SampleButton("Bécquer", "📗", "becquer.txt", False),
    ],
    "f": [
        SampleButton("Citation", "🎲", "random.txt", True),
        SampleButton("Misérables", "📕", "miserables.txt", False),
        SampleButton("Candide", "📗", "candide.txt", False),
    ],
    "h": [
        SampleButton("लोकोक्ति", "🎲", "random.txt", True),
        SampleButton("कबीर", "📕", "kabir.txt", False),
        SampleButton("रहीम", "📗", "rahim.txt", False),
    ],
    "i": [
        SampleButton("Citazione", "🎲", "random.txt", True),
        SampleButton("Divina Commedia", "📕", "divina.txt", False),
        SampleButton("Promessi Sposi", "📗", "promessi.txt", False),
    ],
    "j": [
        SampleButton("ことわざ", "🎲", "random.txt", True),
        SampleButton("こころ", "📕", "kokoro.txt", False),
        SampleButton("坊っちゃん", "📗", "botchan.txt", False),
    ],
    "p": [
        SampleButton("Citação", "🎲", "random.txt", True),
        SampleButton("Brás Cubas", "📕", "bras_cubas.txt", False),
        SampleButton("Iracema", "📗", "iracema.txt", False),
    ],
    "z": [
        SampleButton("古语", "🎲", "random.txt", True),
        SampleButton("唐诗", "📕", "tangshi.txt", False),
        SampleButton("论语", "📗", "lunyu.txt", False),
    ],
}


# show_spinner=False because this function renders its own, written for the
# moment it appears. Left at the default, Streamlit stacks a second spinner
# reading "Running `ensure_repo_downloaded()`." — the Python identifier — above
# the real one for the entire ~355 MB download, which is the longest wait the
# app ever shows and a first-time user's first impression of it.
@st.cache_resource(show_spinner=False)
def ensure_repo_downloaded() -> str:
    try:
        return snapshot_download(REPO_ID, local_files_only=True)
    except LocalEntryNotFoundError:
        with st.spinner("Downloading Kokoro model and voices (one-time, ~355 MB)..."):
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


# The default spinner would read "Running `load_pipeline()`." at the one moment
# the app makes the user wait without explanation — the first Play of a launch.
@st.cache_resource(show_spinner="Loading the Kokoro model (first play only)...")
def load_pipeline() -> Any:
    # The snapshot's Path, not REPO_ID. Handed a repo-id string, mlx-audio's
    # get_model_path runs an online snapshot_download, which asks huggingface.co
    # for the latest revision on the first Play of every launch; a Path skips it.
    # model_type is pinned because mlx-audio otherwise infers it from the path:
    # the directory after the first `hub` component, else the commit-hash
    # directory name. That yields kokoro only while the cache directory, as
    # snapshot_download resolves it, is itself named `hub`, which a custom
    # HF_HUB_CACHE or a `hub` symlink (resolved to its target) need not be.
    model = load_model(Path(ensure_repo_downloaded()), model_type="kokoro")
    # mlx-audio's Kokoro Model hard-codes `prince-canuma/Kokoro-82M` as the repo
    # it pulls voice tensors from, because its loader builds `Model(config)`
    # without forwarding a repo_id. Left alone, every Play re-downloads a voice
    # we already have in our own snapshot. `_get_pipeline` reads this attribute
    # when it first builds (and caches) a per-language pipeline, so setting it
    # here — before any Play click — keeps voice loading offline and local.
    model.repo_id = REPO_ID
    return model


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


# Building a language's G2P stack is the slow part of Tokenize (spaCy plus the
# espeak fallback for English, UniDic for Japanese), and the default spinner
# names the function and its argument rather than saying so.
@st.cache_resource(show_spinner="Preparing the tokenizer for this language...")
def load_tokenizer(lang_code: str) -> Any:
    return _create_g2p(lang_code)


def tokenize_text(text: str, lang_code: str) -> str:
    phonemes, _ = load_tokenizer(lang_code)(text)
    return phonemes or ""


def _tokenize_error_message(lang_code: str, error: Exception) -> str:
    """Friendly, in-place message for a tokenization failure.

    Tokenizing builds a per-language G2P stack, so it fails for environment
    reasons far more often than for bad input. The common one is the Japanese
    UniDic dictionary: it installs into the venv's `unidic` package directory,
    so any venv rebuild silently removes it and every Japanese request then
    raises. That case gets the recovery command instead of a raw error.
    """
    if lang_code == "j" and "unidic" in str(error).lower():
        return (
            "Japanese tokenization needs the UniDic dictionary, which is "
            "missing. Run `uv run python -m unidic download` (one-time, "
            "~1 GB), then reload this page."
        )
    return f"Could not tokenize this text: {error}"


def _format_voice(voice: str) -> str:
    if "_" not in voice:
        return voice
    name = voice.split("_", 1)[1].replace("_", " ").title()
    gender = _GENDER_LABELS.get(voice[1], "")
    grade = VOICE_GRADES.get(voice, "")
    base = f"{name} ({gender})" if gender else name
    # No-break spaces around the dash, so a title too long for its slot breaks
    # between the name and "(gender) — grade" rather than stranding the grade:
    # plain spaces left "Nicole (female) —" over a lone "B-" at ~1230-1300 px.
    return f"{base}\u00a0—\u00a0{grade}" if grade else base


def _filter_voices_by_gender(voices: list[str], gender_code: str | None) -> list[str]:
    if gender_code is None:
        return voices
    return [v for v in voices if v[1] == gender_code]


def _gender_code_from_selection(selected: str | None) -> str | None:
    # "All" (or no selection) → no filter; otherwise map the chosen gender.
    return {"Female": "f", "Male": "m"}.get(selected or "")


def _split_voices_for_display(
    voices: list[str], selected: str | None, top_n: int = 6
) -> tuple[list[str], list[str]]:
    top = voices[:top_n]
    tail = voices[top_n:]
    if selected and selected in tail:
        return top + [selected], [v for v in tail if v != selected]
    return top, tail


def _text_digest(text: str) -> str:
    # Stable across processes (unlike hash()), so cache keys are reproducible.
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _cache_key(voice: str, text: str, speed: float, lang_code: str) -> str:
    return f"audio:{voice}:{lang_code}:{speed}:{_text_digest(text)}"


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
    # A wrapping horizontal container, not st.columns. This row used to span half
    # a full-width page and wanted exact thirds; beside a sidebar the composer is
    # ~270-500 px, and thirds of that are narrower than the labels — at a 1024 px
    # window every label broke mid-word ("Rando/m quote", "Frank/enstei/n"). A
    # horizontal container sizes each button from its own label and wraps *whole
    # buttons* onto a new row when they stop fitting, so a label never breaks.
    # The cost is widths that track label length rather than even thirds.
    with st.container(horizontal=True):
        for entry in buttons:
            st.button(
                entry.label,
                icon=entry.icon,
                key=f"sample_{lang_code}_{Path(entry.filename).stem}",
                width="stretch",
                # Only reached if one label alone outgrows the column; it then
                # wraps inside its button instead of ellipsizing ("Pride & …").
                wrap=True,
                on_click=_set_text_from_sample,
                args=(lang_code, entry.filename, entry.is_random),
            )


def _next_audio_seq() -> int:
    seq = st.session_state.get("_audio_seq", 0) + 1
    st.session_state["_audio_seq"] = seq
    return seq


def _find_stale_cached_audio(
    voice: str, text: str, lang_code: str
) -> VoiceResult | None:
    key = _stale_cached_key(voice, text, lang_code)
    return st.session_state[key] if key is not None else None


def _stale_cached_key(voice: str, text: str, lang_code: str) -> str | None:
    # Most-recently-generated cache key for this (voice, text, lang) at ANY speed.
    # st.session_state iterates a hash-ordered set, not insertion order, so recency
    # must come from the stored seq, not position.
    prefix = f"audio:{voice}:{lang_code}:"
    suffix = f":{_text_digest(text)}"
    keys = [
        k
        for k in st.session_state
        if isinstance(k, str) and k.startswith(prefix) and k.endswith(suffix)
    ]
    if not keys:
        return None
    return max(keys, key=lambda k: st.session_state[k].get("seq", 0))


def _evict_old_audio(protect: frozenset[str] = frozenset()) -> None:
    audio_keys = [
        k for k in st.session_state if isinstance(k, str) and k.startswith("audio:")
    ]
    overflow = len(audio_keys) - AUDIO_CACHE_LIMIT
    if overflow <= 0:
        return
    # Drop the oldest by generation order (lowest seq); never trust iteration order.
    by_age = sorted(audio_keys, key=lambda k: st.session_state[k].get("seq", 0))
    # `protect` shields keys a card is currently displaying: render_voice_card is
    # an @st.fragment, so a Play in one card reruns only that card — evicting a
    # sibling's key here would orphan its still-visible player until a full rerun.
    #
    # It is a preference, not a veto, so the bound cannot quietly depend on how
    # many voices a language happens to ship. Treating protection as absolute is
    # only safe while every language has at most AUDIO_CACHE_LIMIT voices: beyond
    # that, every cached clip can be on display at once, nothing is evictable, and
    # the documented cap degrades into one clip per voice. American English has
    # exactly 20 against a limit of 20 — zero margin, so a single voice added
    # upstream would be enough. Unprotected keys go first, displayed ones only if
    # the cap still isn't met. The clip just generated carries the highest seq, so
    # it sorts last and is never the one dropped.
    ordered = [k for k in by_age if k not in protect]
    ordered += [k for k in by_age if k in protect]
    for k in ordered[:overflow]:
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
    # Deliberately does not tokenize. mlx-audio's KokoroPipeline builds its own
    # G2P and runs it inside `pipeline.generate`, so a pass here is a second one
    # over the same text — and on a Play with no prior Tokenize it also builds
    # this app's own G2P stack (spaCy + the espeak fallback) on the click path,
    # then holds that duplicate resident for the session. Measured ~1.4 s of the
    # first Play, spent to fill a `VoiceResult` field nothing read.
    with st.status(f"Generating {voice}...", expanded=True) as status:
        chunks = []
        for i, chunk in enumerate(
            generate_speech(text, voice, pipeline, speed=speed, lang_code=lang_code), 1
        ):
            chunks.append(chunk)
            st.write(f"Chunk {i}...")
        status.update(label=f"{voice} complete!", state="complete")
    return {
        "wav": _audio_to_wav_bytes(np.concatenate(chunks)),
        "voice": voice,
    }


@st.fragment
def render_voice_card(voice: str, text: str, lang_code: str) -> None:
    with st.container(border=True):
        # One row per voice — name, then speed and Play — so the cards read down
        # the page like a table: an idle card is one control-height tall rather
        # than a title line stacked over a control row, and Play sits in one
        # column down the list. A horizontal container rather than st.columns,
        # because this row is "fill plus fixed", not a proportional grid. Ratio
        # columns could not hold both ends: at a 1024 px window a card is under
        # 300 px, and [1.6, 1, 1] left speed and Play 66 px each, which
        # truncated them to "1.(" and "P…"; the same ratios at 1920 px blew the
        # two controls up to 190 px apiece. Here the controls keep a fixed width
        # and the name takes the rest, so their right edges line up across cards
        # of one width (tail cards, inset by the expander's padding, sit ~17 px
        # further in). And a horizontal container wraps by its *own* width,
        # where columns only stack by viewport (<=640 px): when a card is too
        # narrow for one line, the controls drop under the name as a unit — see
        # the inner container — rather than squeezing until their labels clip.
        with st.container(horizontal=True, vertical_alignment="center"):
            # The title is reserved here and written at the end of the fragment.
            # Its badge depends on whether this card has audio, and on the run
            # that generates some, it does not yet — so emitting the title in
            # place left the one card that had just produced a clip as the only
            # card without a badge, until some unrelated rerun corrected it.
            # `st.container`, not `st.empty`: a placeholder clears at the top of
            # each rerun, which would unmount and remount the title on every
            # fragment run. Stretch width (the default) is what makes the name
            # the row's flexible part, pushing the controls to the right edge.
            title_slot = st.container()
            # Grouped at content width so speed and Play wrap together. Loose
            # in the outer row they wrap one at a time, so a short name kept its
            # speed box on the first line and sent only Play to the second, and
            # neighbouring cards broke at different points.
            with st.container(horizontal=True, width="content", gap="xsmall"):
                card_speed = st.selectbox(
                    # Collapsed, but still the dropdown's accessible name, so it
                    # names its voice: a bare "Speed" was announced identically
                    # on every card, with nothing to say which voice it set.
                    f"Speed for {_format_voice(voice)}",
                    options=SPEED_OPTIONS,
                    index=DEFAULT_SPEED_INDEX,
                    key=f"speed_{voice}",
                    label_visibility="collapsed",
                    format_func=lambda x: f"{x}x",
                    width=CARD_CONTROL_WIDTH,
                    # Cards inside the collapsed "Show all voices" expander are
                    # not drawn, and Streamlit discards the state of any widget a
                    # run did not draw. Without this a tail voice's speed snapped
                    # back to 1.0x on reopen — and because _cache_key includes the
                    # speed, that also demoted its generated clip to a "speed
                    # changed" stale preview. "session", not "page": Streamlit
                    # documents "page" as covering a hidden widget too, and in a
                    # browser it holds here as well, but AppTest builds a fresh
                    # PagesManager per run, so every run reads as a page switch
                    # and "page" drops the value. Only "session" is visible to the
                    # guard test; in a single-page app they're the same.
                    persist_state="session",
                )
                play_clicked = st.button(
                    "Play",
                    icon=":material/play_arrow:",
                    key=f"play_{voice}",
                    type="primary",
                    width=CARD_CONTROL_WIDTH,
                    disabled=not text.strip(),
                )
        stale_key = _stale_cached_key(voice, text, lang_code)
        cached = st.session_state[stale_key] if stale_key is not None else None
        key = _cache_key(voice, text, card_speed, lang_code)
        # Register the key whose audio this card is actually showing so a Play in
        # any other fragment never evicts it: the current-speed take when it exists,
        # otherwise the stale-preview key rendered below (a different speed). Each
        # fragment rewrites its own entry every rerun, so the protect set self-heals
        # against per-card speed changes the main body can't observe.
        displayed_key = key if key in st.session_state else (stale_key or key)
        st.session_state.setdefault("_displayed_card_keys", {})[voice] = displayed_key
        if play_clicked:
            try:
                pipeline = load_pipeline()
                # The chunk-by-chunk status is progress, not a record: once the
                # clip is stored the player below says the same thing, and a
                # lingering "complete!" block was a third row in a card built to
                # be one row plus its player — one that then vanished on the
                # card's next rerun, jolting the player up. So it renders into a
                # placeholder cleared on success only; on failure the status (in
                # its error state) stays above the message as its context.
                progress = st.empty()
                with progress.container():
                    result = generate_one(text, voice, pipeline, card_speed, lang_code)
                result["seq"] = _next_audio_seq()
                st.session_state[key] = result
                # The card now displays this just-generated take, so re-register
                # it: the pre-Play registration above pointed at the prior
                # displayed key (e.g. a stale preview at another speed).
                st.session_state.setdefault("_displayed_card_keys", {})[voice] = key
                protect = frozenset(
                    st.session_state.get("_displayed_card_keys", {}).values()
                )
                _evict_old_audio(protect | {key})
                progress.empty()
            except ValueError as e:
                # Only the benign "no audio" case gets a friendly message;
                # unexpected ValueErrors keep the developer-facing traceback.
                if "No audio generated" in str(e):
                    st.error(str(e))
                else:
                    st.exception(e)
            except Exception as e:  # noqa: BLE001
                # Last-resort UI guard: any failure in one card must not take
                # down the sibling cards, so it is surfaced in-place instead.
                st.exception(e)
        # Written into the slot reserved above, so it renders in title position
        # while being computed after the Play handler — `key in st.session_state`
        # only becomes true once this run's generation has stored a clip.
        # Appended, not prefixed: an icon in front shifted the voice name right by
        # its own width the moment a card had audio, so a stacked column of six
        # cards lost its left edge during exactly the A/B comparison the cache
        # exists for. A badge also names the state rather than leaving a bare
        # speaker glyph to be guessed at. "Cached", not "Ready": it fires when
        # audio exists at *any* speed, including the case where the body below
        # reads "speed changed".
        has_audio = cached is not None or key in st.session_state
        badge = " :green-badge[Cached]" if has_audio else ""
        with title_slot:
            st.markdown(f"**{_format_voice(voice)}**{badge}")
        if key in st.session_state:
            wav = st.session_state[key]["wav"]
            # Player and Download share a row — the same stretch-plus-fit shape
            # as the controls row: the player takes whatever Download leaves.
            # Streamlit gives st.audio a ~14rem flex basis in a horizontal
            # container, so in a card too narrow for both, Download wraps under
            # the player rather than shrinking it to a stub with no seek bar.
            with st.container(horizontal=True, vertical_alignment="center"):
                st.audio(wav, format="audio/wav")
                st.download_button(
                    label="Download",
                    icon=":material/download:",
                    data=wav,
                    file_name=f"{voice}_{card_speed}x.wav",
                    mime="audio/wav",
                    key=f"download_{voice}",
                    # Nothing on the page depends on the click, and the default
                    # "rerun" would re-render this fragment while the browser is
                    # still fetching the file.
                    on_click="ignore",
                )
        elif cached is not None:
            st.caption("Click Play to refresh (speed changed)")
            st.audio(cached["wav"], format="audio/wav")


def render_phonemes(phonemes: str, *, expanded: bool = False) -> None:
    with st.expander("Phoneme tokens", expanded=expanded):
        # Unwrapped, deliberately, though at a 1440 px window the composer column
        # shows only ~50 of a typical 275 phonemes before scrolling. Misaki marks
        # stress with ˈ and ˌ *inside* words, and Unicode line breaking (UAX #14,
        # class BB) allows a break before both, so `wrap_lines=True` split words
        # at their stress marks — "fˈɑðəɹ" read as "f" / "ˈɑðəɹ" in 3 of 5 line
        # breaks at 1440 px.
        # This is the readout users copy `[word](/phonemes/)` overrides from; a
        # scroll hides tokens, a mid-word break misstates them.
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


def _render_tokenize_row(text: str, lang_code: str) -> None:
    # Tokenize and the length caption share a row: the caption is the button's
    # readout (an estimate before, the exact count after), so it sits beside it
    # rather than a row further down. A horizontal container because this is
    # fit-plus-fill: the button at its content width, the caption the rest.
    with st.container(horizontal=True, vertical_alignment="center"):
        clicked = st.button(
            "Tokenize",
            icon=":material/graphic_eq:",
            disabled=not text.strip(),
        )
        # Reserved and filled last, as the voice card does with its title: the
        # caption must follow this run's tokenization, but tokenizing *inside*
        # the row made `load_tokenizer`'s first-use spinner a flex item in it,
        # which knocked the caption onto a second line for the whole load.
        caption_slot = st.container()
    # Out here, so the spinner and any error span the column below the row.
    if clicked:
        try:
            st.session_state["last_phonemes"] = (
                text,
                lang_code,
                tokenize_text(text, lang_code),
            )
        except Exception as e:  # noqa: BLE001
            # Same UI guard as the per-card Play handler: a G2P failure must
            # degrade in place rather than replace the whole page with a
            # traceback. This is the only unwrapped path that loads a
            # tokenizer, so it is where a missing UniDic dictionary surfaces.
            st.error(_tokenize_error_message(lang_code, e))
    with caption_slot:
        _render_length_caption(text, lang_code)


# The sidebar keeps Streamlit's default width and "auto" state (hidden on
# phones). A narrower one was measured and rejected: 256 px gave the main area
# 44 px, but split the pronunciation tips' code spans mid-token ("[Kokoro](/k" /
# "ˈOkəɹO/)") — the one thing a syntax reference must not do — and the voice
# cards' controls wrap as a unit, so they never needed the room.
st.set_page_config(page_title="Kokoro Studio", page_icon="🎙️", layout="wide")
st.title("Kokoro Studio")

try:
    ensure_repo_downloaded()
except Exception:  # noqa: BLE001
    # Any failure here (network, HuggingFace, filesystem) means the app cannot
    # run, so all of them get the same friendly message rather than a traceback.
    st.error(
        "Could not download the Kokoro model. Connect to the internet and reload the page."
    )
    st.stop()

# Below the download guard on purpose: above it, this line would render over the
# one-time download and directly above its failure message, and both contradict
# the claim it makes. "Synthesis runs offline" stays true during that download in
# a way "everything runs locally" would not.
st.caption(
    "Press Play on any voice to hear your text — Play stays greyed out until you "
    "enter some. Synthesis runs offline on this Mac."
)

# The sidebar holds the two app-level filters and the reference note — the
# settings that decide *which* voices the main area shows, and a note consulted
# rather than read in sequence. Moving them out lets the composer and the voice
# list start directly under the caption. The text and the voices themselves stay
# in the main area: they are the content, and on a phone the sidebar starts
# hidden, where only settings can afford to be one tap away.
with st.sidebar:
    # Stretched to the sidebar now. The fixed 320 px width it used to carry only
    # kept a full-viewport dropdown under the title from reading as a banner.
    language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        key="language",
    )
    lang_code = LANGUAGES[language]
    gender_selection = st.segmented_control(
        "Voice gender",
        options=["All", "Female", "Male"],
        default="All",
        required=True,
        key="gender",
    )
    # A bordered container, not st.info. The alert component renders
    # role="status" for info and success, i.e. an ARIA live region, which asks a
    # screen reader to announce this as a status update; the note is permanent
    # reference content that belongs in the normal reading order. A bordered
    # container groups it just as well with no announcement semantics.
    with st.container(border=True):
        st.markdown(":material/lightbulb: **Pronunciation tips**")
        st.markdown(PRONUNCIATION_TIPS)

input_col, voices_col = st.columns(2)

with input_col:
    text_input = st.text_area(
        label="Text",
        placeholder="Start typing here or paste any text you want to turn into lifelike speech...",
        height=TEXT_AREA_HEIGHT,
        key="text_input",
        label_visibility="collapsed",
    )
    _render_sample_buttons(lang_code)
    _render_tokenize_row(text_input, lang_code)
    _render_persistent_phonemes(text_input, lang_code)

with voices_col:
    gender_code = _gender_code_from_selection(gender_selection)
    voices = _filter_voices_by_gender(get_voices(lang_code), gender_code)
    # Reset the protect map on every full rerun — even when the filter empties the
    # list — so voices dropped by a filter or language change don't linger. Each
    # card's fragment re-populates its own entry with the key it is displaying (see
    # render_voice_card). Only the cards this run actually draws register, so while
    # the expander is collapsed the map holds the visible ones alone; that is what
    # leaves _evict_old_audio enough unprotected keys to hold the cache limit.
    st.session_state["_displayed_card_keys"] = {}
    if voices:
        visible, hidden = _split_voices_for_display(voices, None)
        # A tighter gap than the column's default 1rem: an idle card is now only
        # 72 px tall, so the stock gap was nearly a quarter of one, and the list
        # read as six separate panels rather than one scannable table.
        with st.container(gap="xsmall"):
            for voice in visible:
                render_voice_card(voice, text_input, lang_code)
        if hidden:
            # `on_change="rerun"` is what makes `.open` meaningful. Left at the
            # default, an expander computes its whole body even while collapsed —
            # for American English that is 14 extra voice cards, each with a
            # session_state scan and two widgets (three once it has a Download),
            # rebuilt on every rerun.
            # Opening now costs one rerun; every other rerun stops paying for it.
            # `on_change` makes the expander a widget, so its open state is subject
            # to the same collection as any other: a language or filter leaving
            # six voices or fewer skips this branch entirely, Streamlit drops
            # `show_all_voices`, and coming back finds the expander shut. That is
            # the bug `persist_state` fixes for the speed selectbox, but
            # st.expander has no such option, so the state is mirrored by hand
            # into a plain key and fed back through `expanded`. The key's value
            # is copied into the mirror first whenever it survives, so the
            # mirror only decides on the runs that actually lost it.
            # The label carries the tail size because "Show all voices" says a
            # tail exists but not that it is 14 of 20. Note this makes the label
            # track the gender filter, and st.expander hashes `label` into its
            # element id (layouts.py computes it with key_as_main_identity=False
            # and both `label` and `expanded` as inputs), so the widget identity
            # now churns on filter changes too. `expanded` is in the same hash,
            # and a new id starts from `expanded`, not from the click the browser
            # sent under the old one. So `expanded` is re-seeded only on the runs
            # that start the widget afresh anyway (the label moved, which moves
            # the id, or its state was collected) and held still otherwise. Fed
            # last run's open state, the id moved on every second click and that
            # click was lost (1.61.1 and 1.64.0); fed this run's, it moved on
            # every click, so every toggle remounted the expander: keyboard focus
            # fell to the page, and a second click sent within ~50 ms, before the
            # browser had the new id, was lost.
            label = f"Show all voices ({len(hidden)} more)"
            st.session_state["_show_all_voices_pref"] = st.session_state.get(
                "show_all_voices", st.session_state.get("_show_all_voices_pref", False)
            )
            if (
                "show_all_voices" not in st.session_state
                or st.session_state.get("_show_all_voices_label") != label
            ):
                st.session_state["_show_all_voices_seed"] = st.session_state[
                    "_show_all_voices_pref"
                ]
                st.session_state["_show_all_voices_label"] = label
            more_voices = st.expander(
                label,
                icon=":material/library_music:",
                on_change="rerun",
                key="show_all_voices",
                expanded=st.session_state["_show_all_voices_seed"],
            )
            st.session_state["_show_all_voices_pref"] = bool(more_voices.open)
            if more_voices.open:
                with more_voices, st.container(gap="xsmall"):
                    for voice in hidden:
                        render_voice_card(voice, text_input, lang_code)
    else:
        st.caption("No voices match this filter.")
