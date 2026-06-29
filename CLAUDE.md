# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Generate multilingual speech with the [Hexgrad Kokoro model](https://huggingface.co/hexgrad/Kokoro-82M) on Apple Silicon with MLX.

## Installation

Requires `espeak-ng` system dependency.

```bash
uv sync --group dev
uv run python -m unidic download   # one-time, ~1 GB, only needed for Japanese
uv run streamlit run streamlit_app.py
```

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Unit tests**: `uv run pytest`
- **Integration tests**: `uv run pytest tests_integration/`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)
- When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Dependencies

**System:** `espeak-ng`

**Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `espeakng-loader`, `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `num2words`, `numpy`, `phonemizer-fork`, `soundfile`, `spacy`, `streamlit`. The English G2P stack (`spacy`, `num2words`, `phonemizer-fork`, `espeakng-loader`) is pulled in directly rather than via `misaki[en]` to skip its heavy ML extras (`torch`, `transformers`, `spacy-curated-transformers`). Japanese requires a one-time UniDic dictionary download (`uv run python -m unidic download`, ~1 GB).

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`, `testpaths`), ty (`python-version = "3.12"`).

`.streamlit/config.toml` — `[server] fileWatcherType = "none"` plus the "Kokoro indigo" theme: indigo `primaryColor` (`#4F46E5`, AA-contrast with white button text), `8px` radius, Inter body font (weights up to 800) with `headingFontWeights` (`[800, 600, …]`) rendering an extrabold h1 title above Streamlit's 700 default — h2–h6 keep the 600 default and headings reuse the body Inter, so no separate `headingFont` — and JetBrains Mono code font, and `[theme.light]`/`[theme.dark]` blocks defining both modes (with red/orange/green tuned to match the utterance-length caption bands). The app calls `st.set_page_config(page_title="Kokoro Studio", page_icon="🎙️", layout="wide")` as its first Streamlit command.

## Architecture

### Files

- `streamlit_app.py` — main app: language selector, text input + per-language sample buttons + Tokenize button + utterance-length caption + pronunciation note (left column), gender segmented control + per-card voice grid with per-card Play button + speed dropdown + inline audio playback + download button (right column)
- `voice_grades.py` — quality-grade table (`VOICE_GRADES`), rank table (`_GRADE_RANK`), and `_grade_rank` helper extracted from the Kokoro model card; consumed by the voice picker for sorting and labeling
- `samples/` — bundled public-domain sample text per language (9 directories × 3 files: `random.txt` quote pool plus two literary excerpts); referenced by `SAMPLE_BUTTONS` and read by `_load_sample`
- `.streamlit/config.toml` — server config (`fileWatcherType = "none"`) plus the "Kokoro indigo" `[theme]` with `[theme.light]`/`[theme.dark]` blocks (so the toolbar mode toggle appears); the only `.streamlit/` file checked in (a `.gitignore` exception)
- `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, and `huggingface_hub` for import; the `streamlit` mock provides identity-pass-through shims for `cache_resource`, `cache_data`, and `fragment` so decorated functions keep running their real bodies under test
- `tests/test_streamlit_app.py` — unit tests, including `seq`-ordered cache eviction/recency and `.streamlit/config.toml` theme validation
- `tests_integration/conftest.py` — clears `streamlit`/`misaki`/`mlx_audio`/`huggingface_hub` from `sys.modules` so AppTest gets the real modules; incompatible with `tests/conftest.py`'s mocks in one process, so `testpaths = ["tests"]` keeps the integration suite opt-in via an explicit `uv run pytest tests_integration/`
- `tests_integration/test_app_integration.py` — AppTest integration tests: initial render, sample buttons, Tokenize/Play enablement, gender filter, language switching, per-card speed controls

### Key Functions

- `ensure_repo_downloaded` — calls `huggingface_hub.snapshot_download` once per process; tries `local_files_only=True` first and only shows the download spinner when files are missing
- `get_voices` — walks the local snapshot's `voices/` directory and returns voice IDs for the given language, sorted by quality grade (best first, ties alphabetical)
- `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model`; called lazily on the first per-card Play click
- `load_tokenizer` — cached G2P tokenizer via direct `misaki` usage per language
- `_create_g2p` — creates language-specific misaki G2P object
- `tokenize_text` — returns phoneme string without running inference
- `generate_speech` — generator yielding audio arrays per chunk; takes `lang_code` parameter
- `generate_one` — runs `generate_speech` inside an `st.status` block, concatenates chunks, returns a single `VoiceResult` (`audio`, `voice`, `phonemes`; the caller adds a monotonic `seq`)
- `_format_voice` — formats a raw voice ID into a display label with optional grade suffix (e.g. `af_heart` → `"Heart (female) — A"`); ungraded voices show just `"Name (gender)"`. Used as the card title.
- `_grade_rank` — maps a voice ID to its numeric sort rank via `VOICE_GRADES` + `_GRADE_RANK` (both in `voice_grades.py`); ungraded voices get a sentinel rank that sorts last
- `_filter_voices_by_gender` — narrows a voice list to one gender (`"f"` or `"m"`), or returns unchanged for `None` (no filter)
- `_gender_code_from_selection` — maps the gender `st.segmented_control` value to a gender code: `"All"` or no selection → `None` (no filter); `"Female"` → `"f"`; `"Male"` → `"m"`
- `_split_voices_for_display` — splits a voice list into `(visible, hidden)` — top N (default 6) visible, rest hidden. If a selected voice would land in the tail, pins it into the visible section.
- `_text_digest` — stable 16-hex-char `hashlib.sha1` digest of the text; used in cache keys so they are reproducible across processes (unlike the previously used `hash()`, which is per-process randomized)
- `_cache_key` — builds the session-state key for a generated audio: `f"audio:{voice}:{lang_code}:{speed}:{_text_digest(text)}"`. Cache invalidates implicitly when any of voice/text/speed/lang changes.
- `_next_audio_seq` — returns a monotonically increasing counter from `st.session_state["_audio_seq"]`, stamped onto each `VoiceResult` at write time so cache ordering never relies on `st.session_state` iteration order (which is a hash-ordered set in real Streamlit, not insertion order)
- `_stale_cached_key` — returns the session-state key of the most recently generated cached audio for a `(voice, text, lang)` regardless of speed (highest `seq`), or `None`; used both for the stale-preview lookup and to register the displayed key in the protect map
- `_find_stale_cached_audio` — thin wrapper over `_stale_cached_key` returning that key's `VoiceResult` (or `None`); used to show a `:material/volume_up:` badge and a stale-preview player when the current speed has no cached audio
- `_evict_old_audio` — caps the session-state audio cache at `AUDIO_CACHE_LIMIT` (20) by deleting the lowest-`seq` (oldest-generated) entries; ordering comes from `seq`, not iteration order. Takes an optional `protect` set of keys never to evict — the main body resets `st.session_state["_displayed_card_keys"]` (a `voice → displayed key` dict) to `{}` each full rerun, and each card's fragment writes the key whose audio it is actually showing (the current-speed take if present, else the stale-preview key — a possibly different speed); the Play handler passes those values (plus the just-written key) so a fragment's eviction can't orphan a sibling card's visible player, including a stale-preview player at another speed
- `_audio_to_wav_bytes` — encodes a float32 audio array to WAV bytes via `soundfile` for the per-card download button
- `render_voice_card` — `@st.fragment`-wrapped; renders one bordered card per voice so a Play/speed interaction reruns only that card, not the whole script. Title via `_format_voice` (prefixed with a `:material/volume_up:` icon when cached audio exists), a 50/50 inner row with a speed selectbox (left) and `Play` button (`icon=":material/play_arrow:"`, right). It registers the key whose audio it is displaying (current-speed take if present, else the stale-preview key) into `st.session_state["_displayed_card_keys"]` on every rerun, then shows an `st.audio` player and Download button (`icon=":material/download:"`) when audio for `_cache_key(...)` is in session state, or a stale-preview player + "Click Play to refresh (speed changed)" caption when only another speed is cached. Play click runs `generate_one`, stamps `seq`, stores the result, and calls `_evict_old_audio`.
- `_render_persistent_phonemes` — re-renders the `Phoneme Tokens` expander (open) when `last_phonemes` matches the current `(text, lang_code)`, so tokenized phonemes persist across reruns
- `render_phonemes` — renders the `Phoneme Tokens` expander with `st.code`; `expanded` flag toggles open state
- `_estimate_phonemes` — cheap char-count × per-language multiplier (English 0.85, Romance/Portuguese 0.90, Hindi/Italian 1.00, Japanese 1.40, Mandarin 2.00); used by the length caption when no tokenization has run yet
- `_phoneme_band` — returns `(color, label)` for a phoneme count: `<20` red "very short", `20–99` orange "short", `100–399` green "ideal", `400–509` orange "long", `≥510` red "will be chunked"
- `_render_length_caption` — colored `st.caption` under the textarea showing exact phoneme count when `last_phonemes` matches current `(text, lang_code)`, else `~estimate`
- `_load_sample` — `@st.cache_data`, reads `samples/{lang_code}/{filename}` resolved relative to `streamlit_app.py`
- `_pick_sample` — returns the full sample text or a random non-empty line from a one-per-line pool, depending on `is_random`
- `_set_text_from_sample` — `on_click` callback that writes the chosen sample into `st.session_state["text_input"]`; must run between reruns to avoid Streamlit's "widget key already instantiated" error
- `_render_sample_buttons` — renders one row of buttons per language from `SAMPLE_BUTTONS[lang_code]`; each button wired via `on_click=_set_text_from_sample`

### Model

[Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) (`load_model` from `mlx_audio.tts.utils`), 82M params in bf16 precision. Sample rate: 24000 Hz. MLX backend for Apple Silicon.

### Supported Languages

a=American English, b=British English, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin Chinese — 9 languages

### Voice Discovery

On first launch, `ensure_repo_downloaded` calls `huggingface_hub.snapshot_download` to fetch the model and all voice files in one event (~355 MB), with a spinner shown only when the local HuggingFace cache is incomplete (detected via `snapshot_download(..., local_files_only=True)` raising `LocalEntryNotFoundError`). `get_voices` then walks the local snapshot's `voices/` directory and sorts results by quality grade (best first, ties broken alphabetically) using the `VOICE_GRADES` table in `voice_grades.py` (sourced from [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)). Voice files follow the naming convention `{lang}{gender}_{name}` (e.g. `af_heart` — American English, female, "heart") with `.safetensors` extension. Ungraded voices (Spanish, Brazilian Portuguese, or any future addition not in `VOICE_GRADES`) sort to the end. After the initial download the app is fully offline; voices added upstream require clearing the HuggingFace cache to pick up.

### Performance

- MLX backend runs natively on Apple Silicon (no PyTorch or MPS fallback needed)
- `@st.cache_resource` caches the model globally, tokenizers per language, and the snapshot path returned by `ensure_repo_downloaded`
- `@st.cache_data` caches voice lists per language code (local filesystem walk, no TTL needed)
- `ensure_repo_downloaded` first attempts `snapshot_download(..., local_files_only=True)` and only shows the spinner + downloads when the local cache is incomplete
- `load_pipeline()` is deferred until the first per-card Play click, so initial page render is not blocked by model load
- `generate_speech` uses `np.asarray(..., dtype=np.float32)` to avoid copying chunks that are already float32
- Per-card audio results are stored in `st.session_state` keyed by `_cache_key`, so re-renders triggered by other interactions don't regenerate audio
- `render_voice_card` is an `@st.fragment`, so a Play click or speed change reruns only that card instead of re-executing the whole script (each card opens its own `st.container`, giving it an independent fragment instance). One accepted trade-off: a sibling card's `:material/volume_up:` badge refreshes on the next full rerun, not instantly.
- The session-state audio cache is bounded at `AUDIO_CACHE_LIMIT` (20) via `_evict_old_audio`; eviction (oldest-generated first) and stale-preview selection (newest first) both order by the stored `seq`, never by `st.session_state` iteration order

### UI

**Layout (top to bottom):**
0. `st.set_page_config(layout="wide", page_title="Kokoro Studio", page_icon="🎙️")` (first command), then the `st.title("Kokoro Studio")` heading
1. Full-width `Language` selectbox at the top (hidden label)
2. Two-column split: input panel (left) + controls and voice cards (right). Generated audio renders inline inside each card — no separate output row.

**Left column (top to bottom):**
- `st.text_area` (500 px tall, no character cap, hidden label) with placeholder `"Start typing here or paste any text you want to turn into lifelike speech..."`
- Sample button row (`_render_sample_buttons`) — three per-language buttons from `SAMPLE_BUTTONS[lang_code]`: a random-quote button (picks a line from `random.txt`) plus two literary excerpts. Each button is wired via `on_click=_set_text_from_sample`, which runs between reruns and can safely set `st.session_state["text_input"]`.
- `Tokenize` button (`icon=":material/graphic_eq:"`), `disabled=not text.strip()`; clicking stores `(text, lang_code, phonemes)` in `st.session_state["last_phonemes"]`
- Utterance-length caption (`_render_length_caption`) — `st.caption` with exact phoneme count when `last_phonemes` matches, else `~estimate`; bands match VOICES.md guidance
- Persistent phoneme expander (`_render_persistent_phonemes`) — opens automatically when `last_phonemes` matches current `(text, lang_code)`
- `**Note:**` markdown heading followed by `PRONUNCIATION_TIPS` body — always visible, no expander

**Right column:**
- Gender filter: a single `st.segmented_control("Gender", ["All", "Female", "Male"], default="All", key="gender")` with collapsed label. `"All"` (or no selection) → `None` (no filter, show all); `"Female"`/`"Male"` filter to that gender. Translation handled by `_gender_code_from_selection`.
- Voice cards rendered via `render_voice_card`. Top 6 voices (by grade, via `_split_voices_for_display`) visible directly; the rest sit behind an `st.expander("Show all voices", icon=":material/library_music:")`
- When no voices match the gender filter, `st.caption("No voices match this filter.")` renders in place

**Voice card (per voice, via `render_voice_card`, `@st.fragment`):**
- `st.container(border=True)` frame (created inside the function, so each card is its own fragment instance — never pass a pre-built container in)
- Bold title via `_format_voice` (e.g. `**Heart (female) — A**`), prefixed with a `:material/volume_up:` icon when `_find_stale_cached_audio` finds any cached audio for this voice/text/lang
- 50/50 inner row via `st.columns([1, 1])`: speed selectbox on the left, Play button on the right
- Speed selectbox: `SPEED_OPTIONS` (`[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]`), default `1.0`, formatted as `"{x}x"` (e.g. `1.0x`), keyed `f"speed_{voice}"`, label hidden
- Play button: `Play` with `icon=":material/play_arrow:"`, `type="primary"`, `width="stretch"`, keyed `f"play_{voice}"`, `disabled=not text.strip()`. Click handler loads the model (cached), runs `generate_one`, stamps `seq` via `_next_audio_seq`, stores the result in `st.session_state[_cache_key(voice, text, speed, lang_code)]`, then calls `_evict_old_audio`
- When the current speed's cache key is present: an inline `st.audio` player plus a Download button (`f"download_{voice}"`, WAV via `_audio_to_wav_bytes`). Multiple voices' audios coexist for A/B comparison on the same text. When only another speed is cached: a "Click Play to refresh (speed changed)" caption above a stale-preview `st.audio` player (no download)

**Audio cache lifecycle:**
- Cache key includes voice, text, speed, lang_code (text via `_text_digest`) — changing any creates a new key, so the current-speed player disappears until Play is clicked again; a stale-preview player for a previously generated speed may still show via `_find_stale_cached_audio`
- Cached audios persist across reruns triggered by other interactions
- Each result carries a monotonic `seq`; `_evict_old_audio` caps the cache at `AUDIO_CACHE_LIMIT` (20), evicting the lowest-`seq` (oldest-generated) entries. Both eviction and stale-preview selection order by `seq`, never by `st.session_state` iteration order (a hash-ordered set at runtime)

**Behavior:**
- On initial render, `ensure_repo_downloaded` may show an `st.spinner` (`~355 MB`) for the one-time model + voices download when the local HuggingFace cache is incomplete; otherwise no spinner appears
- If the first-launch download fails (e.g. offline with no cache), the script shows `st.error(...)` and halts via `st.stop()` instead of leaking a Python traceback
- Chunk-by-chunk generation progress appears via `st.status` inside the active card during the Play-triggered run
- Expected/user-actionable per-card failures (`ValueError`, e.g. "No audio generated") surface as a clean `st.error(str(e))`; only genuinely unexpected errors fall through to `st.exception()`. Either way other cards remain functional

## Resources

- [MLX Model](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Original Model](https://github.com/hexgrad/kokoro)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)
