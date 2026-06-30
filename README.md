# Kokoro Studio

[![CI](https://github.com/darylalim/kokoro-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/kokoro-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Streamlit application for generating multilingual speech using [Hexgrad Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) on Apple Silicon with MLX.

## Features

- 9 supported languages (American English, British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, Mandarin Chinese)
- One-time model + voice download on first launch (~355 MB); fully offline thereafter
- Per-language sample text buttons (🎲 Random Quote + two literary excerpts) to seed the textarea with public-domain reference text
- Voice cards sorted by quality grade (best first), with grade in each title (e.g. "Heart (female) — A"); top 6 visible, rest behind a "Show all voices" expander
- Gender filter via a single segmented control (All / Female / Male), defaulting to All
- Per-card Play button generates audio on demand and embeds an audio player inline — A/B-compare multiple voices on the same text. Each card reruns on its own (`st.fragment`), so playing one voice or changing its speed never reloads the whole page
- Per-card speed dropdown (0.7x–1.5x in 0.1 steps); a speaker icon marks voices with cached audio, and switching speed shows the previous take until you re-play
- Per-card Download button for the generated WAV
- Light and dark themes with a toolbar toggle
- Chunk-by-chunk generation progress via `st.status` inside the active card
- Phoneme token display via standalone Tokenize button (renders inline below the text input)
- Utterance-length caption under the textarea, color-coded against VOICES.md ideal bands (short / ideal / long / will-be-chunked)
- Pronunciation note always visible with Kokoro-specific syntax (custom phonemes, stress, intonation)

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)

## Installation

```bash
uv sync --group dev
uv run python -m unidic download   # one-time, ~1 GB, only needed for Japanese
uv run streamlit run streamlit_app.py
```

> **Note:** The spaCy model `en_core_web_sm` (required for English G2P) is installed automatically via `uv sync`.

## Development

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Unit tests**: `uv run pytest`
- **Integration tests**: `uv run pytest tests_integration/`

### Releasing

Pushing a `vX.Y.Z` tag publishes a GitHub Release automatically (via `.github/workflows/release.yml`), with notes generated from the commits since the previous tag:

```bash
# bump `version` in pyproject.toml (then `uv lock` to sync uv.lock), commit, then:
git tag -a v0.16.0 -m "v0.16.0"
git push origin v0.16.0
```

The workflow verifies the tag matches `pyproject.toml`'s `version` before publishing.

## License

[MIT](LICENSE) © 2026 Daryl Lim

### Third-party licenses & acknowledgements

This app is a thin Streamlit front-end. At runtime it downloads and depends on third-party components under their own licenses:

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** — the TTS model (Apache-2.0), downloaded on first launch and used unmodified; it is **not** redistributed in this repository.
- The English G2P path pulls in **espeak-ng** and **phonemizer-fork** (both **GPLv3**) and **num2words** (**LGPL**). Installing and running the app from source via `uv sync` is unaffected by these terms, but note that a *bundled, redistributed build* (e.g. a Docker image or standalone binary that vendors these dependencies) would be a combined work subject to **GPLv3**.

Bundled sample texts under `samples/` are public domain.
