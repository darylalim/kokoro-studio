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

## Dependencies

**System:** `espeak-ng`

**Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `espeakng-loader`, `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `num2words`, `numpy`, `phonemizer-fork`, `soundfile`, `spacy`, `streamlit`. The English G2P stack (`spacy`, `num2words`, `phonemizer-fork`, `espeakng-loader`) is pulled in directly rather than via `misaki[en]` to skip its heavy ML extras (`torch`, `transformers`, `spacy-curated-transformers`). Japanese requires a one-time UniDic dictionary download (`uv run python -m unidic download`, ~1 GB).

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`, `testpaths`), ty (`python-version = "3.12"`).

## Architecture

### Files

- `streamlit_app.py` — main app: language selector, text input + per-language sample buttons + Tokenize button + utterance-length caption + pronunciation note (left column), gender checkboxes + per-card voice grid with per-card Play button + speed dropdown + inline audio playback + download button (right column)
- `voice_grades.py` — quality-grade table (`VOICE_GRADES`), rank table (`_GRADE_RANK`), and `_grade_rank` helper extracted from the Kokoro model card; consumed by the voice picker for sorting and labeling
- `samples/` — bundled public-domain sample text per language (9 directories × 3 files: `random.txt` quote pool plus two literary excerpts); referenced by `SAMPLE_BUTTONS` and read by `_load_sample`
- `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, and `huggingface_hub` for import
- `tests/test_streamlit_app.py` — unit tests
- `tests_integration/conftest.py` — clears `streamlit`/`misaki`/`mlx_audio`/`huggingface_hub` from `sys.modules` so AppTest gets the real modules; incompatible with `tests/conftest.py`'s mocks in one process, so `testpaths = ["tests"]` keeps the integration suite opt-in via an explicit `uv run pytest tests_integration/`
- `tests_integration/test_app_integration.py` — AppTest integration tests: initial render, sample buttons, Tokenize/Play enablement, gender filter, language switching

### Key Functions

- `ensure_repo_downloaded` — calls `huggingface_hub.snapshot_download` once per process; tries `local_files_only=True` first and only shows the download spinner when files are missing
- `get_voices` — walks the local snapshot's `voices/` directory and returns voice IDs for the given language, sorted by quality grade (best first, ties alphabetical)
- `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model`; called lazily on the first per-card Play click
- `load_tokenizer` — cached G2P tokenizer via direct `misaki` usage per language
- `_create_g2p` — creates language-specific misaki G2P object
- `tokenize_text` — returns phoneme string without running inference
- `generate_speech` — generator yielding audio arrays per chunk; takes `lang_code` parameter
- `generate_one` — runs `generate_speech` inside an `st.status` block, concatenates chunks, returns a single `VoiceResult`
- `_format_voice` — formats a raw voice ID into a display label with optional grade suffix (e.g. `af_heart` → `"Heart (female) — A"`); ungraded voices show just `"Name (gender)"`. Used as the card title.
- `_grade_rank` — maps a voice ID to its numeric sort rank via `VOICE_GRADES` + `_GRADE_RANK` (both in `voice_grades.py`); ungraded voices get a sentinel rank that sorts last
- `_filter_voices_by_gender` — narrows a voice list to one gender (`"f"` or `"m"`), or returns unchanged for `None` (no filter)
- `_gender_code_from_checkboxes` — maps Female/Male checkbox state to a gender code: both checked or both unchecked → `None` (no filter); only Female → `"f"`; only Male → `"m"`
- `_split_voices_for_display` — splits a voice list into `(visible, hidden)` — top N (default 6) visible, rest hidden. If a selected voice would land in the tail, pins it into the visible section.
- `_cache_key` — builds the session-state key for a generated audio: `f"audio:{voice}:{lang_code}:{speed}:{hash(text)}"`. Cache invalidates implicitly when any of voice/text/speed/lang changes.
- `render_voice_card` — renders one bordered card per voice: title via `_format_voice`, a 50/50 inner row with a speed selectbox on the left and a Play button on the right, and an `st.audio` player below when audio for `_cache_key(...)` is in session state. Play click runs `generate_one` and stores the result.
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

### UI

**Layout (top to bottom):**
1. Full-width `Language` selectbox at the top (hidden label)
2. Two-column split: input panel (left) + controls and voice cards (right). Generated audio renders inline inside each card — no separate output row.

**Left column (top to bottom):**
- `st.text_area` (500 px tall, no character cap, hidden label) with placeholder `"Start typing here or paste any text you want to turn into lifelike speech..."`
- Sample button row (`_render_sample_buttons`) — three per-language buttons from `SAMPLE_BUTTONS[lang_code]`: a random-quote button (picks a line from `random.txt`) plus two literary excerpts. Each button is wired via `on_click=_set_text_from_sample`, which runs between reruns and can safely set `st.session_state["text_input"]`.
- `Tokenize` button, `disabled=not text.strip()`; clicking stores `(text, lang_code, phonemes)` in `st.session_state["last_phonemes"]`
- Utterance-length caption (`_render_length_caption`) — `st.caption` with exact phoneme count when `last_phonemes` matches, else `~estimate`; bands match VOICES.md guidance
- Persistent phoneme expander (`_render_persistent_phonemes`) — opens automatically when `last_phonemes` matches current `(text, lang_code)`
- `**Note:**` markdown heading followed by `PRONUNCIATION_TIPS` body — always visible, no expander

**Right column:**
- Two checkboxes row: `Female` and `Male` in side-by-side columns via `st.columns(2)`, both default unchecked. Both-equal (both checked or both unchecked) → `None` (no filter, show all); only one checked filters to that gender. Translation handled by `_gender_code_from_checkboxes`.
- Voice cards rendered via `render_voice_card`. Top 6 voices (by grade, via `_split_voices_for_display`) visible directly; the rest sit behind an `st.expander("Show All Voices")`
- When no voices match the gender filter, `st.info("No voices match this filter.")` renders in place

**Voice card (per voice, via `render_voice_card`):**
- `st.container(border=True)` frame
- Bold title via `_format_voice` (e.g. `**Heart (female) — A**`)
- 50/50 inner row via `st.columns([1, 1])`: speed selectbox on the left, Play button on the right
- Speed selectbox: `SPEED_OPTIONS` (`[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]`), default `1.0`, formatted as `"{x}x"` (e.g. `1.0x`), keyed `f"speed_{voice}"`, label hidden
- Play button: `▶ Play`, `type="primary"`, `use_container_width=True`, keyed `f"play_{voice}"`, `disabled=not text.strip()`. Click handler loads the model (cached), runs `generate_one`, and stores the result in `st.session_state[_cache_key(voice, text, speed, lang_code)]`
- An `st.audio` player embeds inline below the row when that cache key is present in session state — multiple voices' audios coexist for A/B comparison on the same text

**Audio cache lifecycle:**
- Cache key includes voice, text, speed, lang_code — changing any creates a new key, so the audio player disappears until Play is clicked again (correct: different inputs → different audio)
- Cached audios persist across reruns triggered by other interactions
- No explicit eviction — the cache grows with each (voice, text, speed, lang) tuple played, which is fine for typical sessions

**Behavior:**
- On initial render, `ensure_repo_downloaded` may show an `st.spinner` for the one-time model + voices download when the local HuggingFace cache is incomplete; otherwise no spinner appears
- If the first-launch download fails (e.g. offline with no cache), the script shows `st.error(...)` and halts via `st.stop()` instead of leaking a Python traceback
- Chunk-by-chunk generation progress appears via `st.status` inside the active card during the Play-triggered run
- Per-card errors surface via `st.exception()` inside the card; other cards remain functional

## Resources

- [MLX Model](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Original Model](https://github.com/hexgrad/kokoro)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)
