# Kokoro Studio

*Private, on-device text-to-speech for Apple Silicon — nine languages, dozens of graded voices, fully offline after a one-time download.*

[![CI](https://github.com/darylalim/kokoro-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/kokoro-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
![Platform: Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black.svg)
![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)
![Runs offline](https://img.shields.io/badge/Runs-Offline-success.svg)

Streamlit application for generating multilingual speech using [Hexgrad Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) on Apple Silicon with MLX.

<p align="center">
  <img src="assets/screenshot-dark.png" width="100%" alt="Kokoro Studio in dark theme — a Gatsby excerpt in the text area with its phoneme tokens expanded below, beside a grade-sorted column of voice cards whose top card has finished generating and shows an audio player and a download button">
</p>

## Why Kokoro Studio

- **Fully offline & private** — after a one-time ~355 MB download, nothing leaves your Mac: no API keys, no network calls, no usage caps.
- **On-device Apple Silicon** — runs natively through Apple's [MLX](https://github.com/ml-explore/mlx), with no PyTorch or MPS fallback.
- **Nine languages, dozens of voices** — voices are sorted by quality grade so the strongest options surface first.
- **Built for comparison** — generate several voices on the same text and A/B them inline, each at its own playback speed.

> **Requires an Apple Silicon Mac (M1 or newer).** MLX is Apple-Silicon-only, so installation will fail to resolve on Intel macOS, Linux, or Windows.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Features

**Languages & voices**

- Nine languages: American & British English, Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese, and Mandarin Chinese.
- Voice cards sorted by quality grade (best first). The grade is shown in the title where the model card provides one (e.g. "Heart (female) — A"); ungraded voices (Spanish and Brazilian Portuguese) show just "Name (gender)" and sort after the graded ones. The top 6 are visible, the rest sit behind a "Show all voices (N more)" expander.
- Gender filter via a single segmented control (All / Female / Male), defaulting to All.

**Generation & playback**

- Per-card Play button generates audio on demand and embeds an inline player — play multiple cards to A/B-compare voices on the same text. Each card reruns on its own, so playing one voice or changing its speed never reloads the whole page.
- Per-card speed control (0.7x–1.5x in 0.1 steps); a green **Cached** badge marks voices with cached audio, and changing speed keeps the previous clip visible until you press Play again.
- Per-card Download button for the generated WAV.
- Chunk-by-chunk generation progress while a clip is being synthesized.
- Light and dark themes with a toolbar toggle.

**Privacy & offline**

- One-time model + voice download on first launch (~355 MB); fully offline thereafter.

**Text & pronunciation tools**

- Per-language sample buttons to seed the text box with public-domain reference text: a localized random-quote button (🎲 "Random quote" in English, 🎲 "古语" in Chinese) plus two literary excerpts, each 📕/📗 excerpt named for its source.
- Tokenize button to preview the phoneme tokens before synthesizing.
- Utterance-length caption under the text box, color-coded against [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) bands (very short / short / ideal / long / will-be-chunked).
- Always-visible pronunciation note with Kokoro-specific syntax (custom phonemes, stress, intonation).

## Requirements

- macOS with Apple Silicon (M1 or newer)
- Python 3.12+ — the repo ships a `.python-version` pinning **3.12** (matching CI), so `uv sync` builds the venv on 3.12 even if you have a newer Python installed
- [uv](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

No system packages are needed: [espeak-ng](https://github.com/espeak-ng/espeak-ng) ships as a prebuilt library inside the `espeakng-loader` wheel, which `uv sync` installs.

## Installation

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The app opens at <http://localhost:8501>. On first launch it downloads the model and voices once (~355 MB, shown with a spinner); the model itself also loads on your first Play. After the initial download it runs fully offline. Press `Ctrl+C` in the terminal to stop it.

> **Note:** Both G2P prerequisites are installed automatically by `uv sync` — the spaCy model `en_core_web_sm` (for English) and the bundled espeak-ng library (for English fallback and the espeak-backed languages). A separate `brew install espeak-ng` is **not** required: `misaki` calls `EspeakWrapper.set_library()` on the wheel's own `libespeak-ng.dylib` at import, so a system install is never consulted.

### Japanese support (optional)

Japanese G2P needs the UniDic dictionary, a one-time ~1 GB download. Skip this unless you plan to use Japanese:

```bash
uv run python -m unidic download
```

> **Note:** The dictionary installs *inside* the virtual environment, so anything that rebuilds the venv — changing `.python-version`, deleting `.venv`, switching interpreters — removes it and you will need to run the command again. The app detects this case and tells you, rather than failing with a traceback.

## Usage

1. Pick a language from the selector at the top.
2. Type or paste text into the box — or click a sample button (a random quote plus two literary excerpts) to seed it.
3. *(Optional)* Click **Tokenize** to preview the phoneme tokens. The colored caption tells you whether the text is too short, ideal length, or long enough to be chunked.
4. *(Optional)* Filter the voices by gender (All / Female / Male).
5. Pick a voice card — the top 6 by quality grade are shown directly, with the rest behind **Show all voices (N more)**, where N counts what the current gender filter leaves in the tail.
6. Choose a playback speed (0.7x–1.5x).
7. Click **Play**. Generation progress appears inline, and the audio player shows up in the card when it's done.
8. Play other cards to A/B-compare voices on the same text (a **Cached** badge marks cards that already have audio), and use **Download** to save any clip as a WAV.

> **Tip:** For custom pronunciation, use the in-app syntax — e.g. `[Kokoro](/kˈOkəɹO/)`. See the **Note** panel in the app for stress and intonation controls.

## How it works

Kokoro Studio is a thin Streamlit front-end over the [`mlx-community/Kokoro-82M-bf16`](https://huggingface.co/mlx-community/Kokoro-82M-bf16) MLX model (82M params, bf16, 24 kHz).

**Synthesis path.** Your raw text and the selected language code go straight into the mlx-audio Kokoro pipeline, which performs grapheme-to-phoneme (G2P) conversion and vocoding internally and streams audio chunks back:

```
text + language code  →  MLX Kokoro pipeline (internal G2P + vocoding)  →  audio chunks
```

The **Tokenize** button and the length caption run a *separate* `misaki` / `espeak-ng` G2P purely to **display** phoneme tokens and estimate length — those phonemes are not fed back into the model.

**Other notable pieces:**

- **Offline-first** — `snapshot_download` fetches the model and all voices once; afterward, voice discovery is just a local filesystem walk. `load_pipeline` loads the model from that snapshot's path rather than its repo id — handed a repo id, mlx-audio asks huggingface.co for the latest revision on every launch's first Play — and also pins the loaded model's `repo_id` to the same snapshot: mlx-audio otherwise falls back to a hard-coded `prince-canuma/Kokoro-82M` for voice tensors, which would re-download each voice on its first Play and fail outright when offline.
- **Fragment-scoped reruns** — each voice card is an `st.fragment`, so a Play or speed change reruns only that card. Generated audio is cached in session state (bounded, oldest-evicted) so unrelated interactions don't regenerate it.

For a full file-by-file map, a function reference, and design notes, see [CLAUDE.md](CLAUDE.md).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Could not download the Kokoro model" on first launch | The one-time ~355 MB fetch needs internet. Check your connection and reload. |
| Japanese errors or produces no audio | Run `uv run python -m unidic download` (one-time, ~1 GB). |
| Japanese worked before and now doesn't | The venv was rebuilt (interpreter change, deleted `.venv`), which removes the UniDic dictionary stored inside it. Re-run `uv run python -m unidic download`. |
| App dies on Play with exit code 1 and no traceback | espeak-ng truncates its data path at ~160 bytes. Move the checkout somewhere shorter, then rebuild the venv with `rm -rf .venv && uv sync`. A plain `uv sync` will **not** fix it — the console scripts keep an absolute shebang pointing at the old location, so you get a confusing "bad interpreter" error instead. Rebuilding also wipes UniDic, so re-run the Japanese download if you use it. |
| `uv sync` fails to resolve / won't install | You're not on Apple Silicon. MLX requires an Apple Silicon Mac; Intel macOS, Linux, and Windows are unsupported. |
| Port already in use | `uv run streamlit run streamlit_app.py --server.port 8502` |
| Source edits don't auto-reload | The file watcher is disabled (`fileWatcherType = "none"`); restart the app or use the toolbar's **Rerun**. |

## Development

```bash
uv sync --group dev          # install dev tooling (pytest, ruff, ty)
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ty check              # typecheck
uv run pytest                # unit tests
uv run pytest tests_integration/   # integration tests (opt-in)
```

**Contributing & CI.** CI is the merge gate on every push to `main` and every PR (runs on `macos-latest`): `ruff check`, `ruff format --check .`, `ty check`, and the unit test suite. `uv sync --locked` fails on lockfile drift, so re-run `uv lock` after changing dependencies. The integration suite is opt-in (`uv run pytest tests_integration/`) and needs the real ~355 MB download. Heads-up: CI gates on `ruff format --check .`, not the bare `ruff format .` above — format locally before pushing. A second job in the same workflow publishes a release when the version in `pyproject.toml` changes — see **Releasing** below.

<details>
<summary><strong>Releasing</strong> (maintainers)</summary>

Releases are cut automatically from the version in `pyproject.toml`. There is no tagging step:

```bash
# bump `version` in pyproject.toml, then `uv lock`, commit
git push origin main
```

When that lands on `main`, the `release` job in `.github/workflows/ci.yml` checks whether a `vX.Y.Z` tag for the new version exists. If not, it creates and pushes an annotated tag, drafts a GitHub Release, and publishes it.

- **It cannot ship a broken build** — the job `needs` the lint/type-check/test job, so a red tree blocks the release. (The old tag-triggered workflow never consulted CI.)
- **It's idempotent** — pushes that don't change the version are a no-op, and re-running a run that died between tagging and publishing finishes the job rather than duplicating it.
- **Final releases only** — the version must be exactly `X.Y.Z`; the job fails loudly on anything else.
- **To re-cut a release**, delete the GitHub Release and its tag, then re-run the CI run.

Release notes are built by `release_notes.py`, which groups commit subjects since the previous release under headings by [conventional-commit](https://www.conventionalcommits.org) type — `feat:` under **Features**, `fix:` under **Bug Fixes**, and so on, with anything marked `!` or carrying a `BREAKING CHANGE:` trailer promoted to the top. Write commit subjects in that form and the notes look after themselves; anything else still appears, verbatim, under **Other Changes**.

</details>

## License

[MIT](LICENSE) © 2026 Daryl Lim

### Third-party licenses & acknowledgements

This app is a thin Streamlit front-end. At runtime it downloads and depends on third-party components under their own licenses:

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by hexgrad — the upstream TTS model (Apache-2.0). At runtime the app downloads the [`mlx-community/Kokoro-82M-bf16`](https://huggingface.co/mlx-community/Kokoro-82M-bf16) MLX conversion (a bf16 derivative, also Apache-2.0); neither is redistributed in this repository.
- The G2P stack pulls in **espeak-ng** and **phonemizer-fork** (both **GPLv3**) and **num2words** (**LGPL**). These drive phonemization for English *and* the espeak-backed languages (Spanish, French, Hindi, Italian, Brazilian Portuguese), so the GPLv3 exposure is language-agnostic, not English-only. Note that espeak-ng arrives as a **prebuilt GPLv3 binary vendored inside the `espeakng-loader` wheel**, not as a separately-installed system package — so `uv sync` places it in your virtualenv either way. Installing and running the app from source is unaffected by these terms, but a *bundled, redistributed build* (e.g. a Docker image or standalone binary) would already be shipping that GPLv3 library and would be a combined work subject to **GPLv3**. num2words is LGPL, whose weaker terms don't impose GPLv3 on the larger work.

Bundled sample texts under `samples/` are public domain.
