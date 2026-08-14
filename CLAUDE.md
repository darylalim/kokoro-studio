# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit application for generating multilingual speech using [Hexgrad Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) on Apple Silicon with MLX.

## Installation

No system dependencies. espeak-ng is vendored as a prebuilt library inside the `espeakng-loader` wheel, so `uv sync` is sufficient (see Dependencies).

```bash
uv sync --group dev
uv run python -m unidic download   # one-time, ~1 GB, only needed for Japanese
uv run streamlit run streamlit_app.py
```

`.python-version` pins the interpreter to **3.12**, matching CI (`setup-uv` requests `python-version: "3.12"`), the `requires-python = ">=3.12"` floor, and `[tool.ty.environment]`. Without it `uv sync` falls back to uv's normal interpreter discovery and takes the first candidate satisfying `requires-python` — not necessarily the version CI uses — which silently drifts local dev off the version CI actually gates on — and since CI excludes `tests_integration/`, that suite would then never run on 3.12 anywhere. Keep `ty` targeting 3.12 even if a newer interpreter is used locally: type-checking against the declared floor is what catches accidental use of newer-only APIs.

Note that changing the pinned version makes the next `uv sync` rebuild the venv, which discards the ~1 GB UniDic dictionary (it installs into the venv's `unidic` package directory). Re-run `uv run python -m unidic download` afterwards if you use Japanese.

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Unit tests**: `uv run pytest`
- **Integration tests**: `uv run pytest tests_integration/`

**CI** (`.github/workflows/ci.yml`) has two jobs: `lint-typecheck-test` (the merge gate) and `release` (see Releases below).

`lint-typecheck-test` runs on push to `main` + every PR, on `macos-latest` only (mlx/mlx-metal are `darwin`-gated in `uv.lock`), `timeout-minutes: 20`, then `uv sync --locked --group dev` (fails on lockfile drift) → `ruff check .` → `ruff format --check .` → `ty check` → `pytest`. The integration suite is excluded (needs the real modules + the ~355 MB download). Note CI gates on `ruff format --check .`, not the bare `ruff format .` above. Steps omit `name:` so GitHub labels them with the command itself; keep `python-version: "3.12"` in the `setup-uv` block, which `TestPythonVersionConsistency.test_ci_workflow_matches` asserts on.

`cancel-in-progress` is an expression (`${{ github.ref != 'refs/heads/main' }}`), not `true`: superseded PR runs are still cancelled, but main runs must not be — cancelling one between the release job's tag push and its publish would strand a tag with no release. Same concurrency group without cancellation makes main runs *queue*, so a later run always observes the tag an earlier one pushed. Guarded by `TestReleaseWorkflow.test_main_runs_are_never_cancelled_mid_publish`.

Three standing CI policies. The workflow's own comments carry the version-specific detail — change them **there**, not here, so there is one place to update per bump:

- **No system packages.** espeak-ng is vendored, so `brew install espeak-ng` is a no-op (see Dependencies). `TestEspeakVendoring` fails the build if a `brew install` reappears in `ci.yml`, if the wheel stops shipping the library/data dir, or if `misaki` stops binding phonemizer to it.
- **Pins do not float.** `astral-sh/setup-uv` has published no major tags since v7, so it carries a full version (`@v10.0.0`; `@v10` does not resolve). `actions/checkout` is pinned per job by token scope: `@v7` in `lint-typecheck-test` (`contents: read`), SHA-pinned in the `release` job (`contents: write`), since a floating major tag is repointable upstream and would then execute against a repo-write token. `TestReleaseWorkflow.test_release_job_checkout_is_sha_pinned` enforces the write-token half. None of these refresh themselves, which is what `.github/dependabot.yml` is for. `uv` itself is pinned via setup-uv's `version:` input and should track the local `.venv/pyvenv.cfg`.
- **Cache is pruned explicitly.** setup-uv v9 flipped `prune-cache` to `false`; left at that default this repo saves its whole ~1.9 GB uv cache per lockfile hash against GitHub's 10 GB per-repo quota, so `ci.yml` sets it back to `true` (~180 MB). The cost is the other direction: `uv sync` re-fetches wheels from PyPI instead of reading them back out of the cache.

**Releases** (`ci.yml`'s `release` job): fully automatic on a version bump. Flow is just **bump `version` in `pyproject.toml` → `uv lock` → commit → push to `main`**. No manual tagging.

The job runs on `ubuntu-latest` (no build/test — that is the other job's), `needs: lint-typecheck-test` so a red tree cannot ship, and is gated `if: github.event_name == 'push' && github.ref == 'refs/heads/main'` so PRs — fork PRs especially — never reach the `contents: write` token. That token is granted per job, so `lint-typecheck-test` (which executes third-party code via `uv sync`) stays `contents: read`. Three steps: annotated `git tag -a vX.Y.Z` + push → `gh release create --draft --generate-notes --verify-tag` → `gh release edit --draft=false`. Drafting first means a failure between steps leaves an invisible draft, not a half-formed public release; GitHub auto-marks the newest published release "Latest", so no `--latest`.

The bump is detected by asking whether the tag and release for `pyproject.toml`'s version **already exist** — never by diffing against the parent commit, which breaks on squash merges, multi-commit pushes, and re-runs. So every push to `main` enters the job and the common case is a three-step skip. The states are `absent` (tag + draft + publish), `draft` (publish only — a re-run finishes a job that died mid-flight), `published` (skip). Version must be exactly `X.Y.Z`; the job errors otherwise, mirroring `TestReleaseWorkflow.test_pyproject_version_is_grep_extractable`. For prereleases, add PEP 440-aware normalisation and pass `--prerelease`.

There is deliberately **no** tag-triggered `release.yml` — it published from a hand-pushed tag without consulting CI, and a second publish path risks double-cutting one version. `TestReleaseWorkflow.test_tag_triggered_release_workflow_stays_retired` fails the build if it reappears. To re-cut a release, delete the GitHub release (and its tag) and re-run the CI job.

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- isort with combine-as-imports (configured in `pyproject.toml`)
- `pyproject.toml` sets no `[tool.ruff.lint] select`, so the project runs on **ruff's default rule set** — which ruff 0.16 widened from 59 rules to 413 (this is what first enabled the `I` isort rules the repo had already configured but never enforced). A ruff upgrade can therefore surface new findings without any code change; check `ruff check .` after every bump rather than assuming a clean tree stays clean.
- Tests that assert *about* a config file (`ci.yml`, `.streamlit/config.toml`) must match against a **comment-stripped** view of it, never the raw text — `TestReleaseWorkflow._release_job_directives` is the helper for the release job. This repo documents each policy in prose directly above the line implementing it, so a raw substring check is satisfied by the explanation whether or not the policy survives: `assert "contents: write" in job` passed against the comment `# ... this job runs with \`contents: write\``, and so stayed green when the `permissions` block was deleted *or* hoisted workflow-wide. Anchor flags too — `assert "--draft" in job` is a substring of the publish step's `--draft=false`. Verify such a test by deleting the line it guards and confirming it fails; "it passes" proves nothing here.
- The two `except Exception` handlers in `streamlit_app.py` carry `# noqa: BLE001` and are deliberate: broad catches are correct here because the app must degrade to a friendly `st.error` / in-card `st.exception` instead of leaking a traceback into the UI. Don't narrow them to satisfy the linter.
- When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff to ensure best practices are followed.

## Dependencies

**System:** none. espeak-ng is *not* a system dependency here — `espeakng-loader` ships a prebuilt `libespeak-ng.dylib` plus `espeak-ng-data` inside the wheel, and `misaki/espeak.py` calls `EspeakWrapper.set_library()` / `set_data_path()` on those at import. `phonemizer`'s `EspeakWrapper.library` resolves lazily and checks its `_ESPEAK_LIBRARY` class attribute *first*, ahead of `$PHONEMIZER_ESPEAK_LIBRARY` and `ctypes.util.find_library('espeak-ng')`, so the system lookup is unreachable once misaki has been imported. This covers the synthesis path too: mlx-audio's `tts/models/kokoro/pipeline.py` imports `misaki.espeak` for `EspeakG2P`/`EspeakFallback` rather than touching phonemizer directly, and nothing in mlx-audio shells out to an `espeak` binary (its only `shutil.which` calls are for `ffmpeg`). Confirmed by running the real, unmocked G2P for English and Spanish on a machine with no `espeak`/`espeak-ng` on `PATH`. A `brew install espeak-ng` therefore changes nothing — don't add it to setup docs or CI.

> **Keep the venv path short.** espeak-ng stores its data directory in a fixed 160-byte `N_PATH_HOME` buffer. If the resolved `espeakng_loader/espeak-ng-data` path exceeds it, the path is truncated, espeak falls back to the build-machine path baked into the wheel (`/Users/runner/work/espeakng-loader/...`), fails to find `phontab`, and calls `exit(1)` directly — the process dies with **exit code 1, no Python traceback, no pytest summary, and buffered output lost**. It reads like a mysterious interpreter-specific crash. To check a given checkout, measure the resolved data directory — `<venv>/lib/pythonX.Y/site-packages/espeakng_loader/espeak-ng-data` — and keep it comfortably under ~159 characters; a `.venv` in the repo root is normally fine, while a venv under a deeply nested CI cache or temp directory may not be. (The 160-byte figure is inferred from the observed truncation boundary, not read from espeak-ng's source, so treat it as approximate.) Since `misaki/espeak.py` calls `EspeakWrapper.set_data_path()` at import time, a later override cannot fix it — espeak is already initialized.

**Runtime:** `en-core-web-sm` (pinned URL; update wheel URL if spaCy is upgraded), `espeakng-loader`, `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `num2words`, `numpy`, `phonemizer-fork`, `soundfile`, `spacy`, `streamlit`. The English G2P stack (`spacy`, `num2words`, `phonemizer-fork`, `espeakng-loader`) is pulled in directly rather than via `misaki[en]` to skip its heavy ML extras (`torch`, `spacy-curated-transformers`, and a direct `transformers` pull — `torch`/`spacy-curated-transformers` are then absent from `uv.lock`, though `transformers` is still installed transitively via `mlx-audio`/`mlx-lm`). Japanese requires a one-time UniDic dictionary download (`uv run python -m unidic download`, ~1 GB).

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`, `testpaths`), ty (`python-version = "3.12"`).

`.streamlit/config.toml` — `[server] fileWatcherType = "none"` plus the "Kokoro indigo" `[theme]`: indigo `primaryColor` (`#4F46E5`, AA-contrast with white button text), `8px` `baseRadius`, `linkUnderline = false`, Inter body font (weights to 800) with `headingFontWeights = [800, 600, …]` — an extrabold h1 above Streamlit's 700 default, h2–h6 at the 600 default, headings reuse the body Inter so no separate `headingFont` — and JetBrains Mono `codeFont`. `[theme.light]`/`[theme.dark]` blocks define both modes (so the toolbar mode toggle appears): each sets a full GitHub-Primer base palette (`backgroundColor`/`secondaryBackgroundColor`/`textColor`/`borderColor`) plus `red`/`orange`/`green` tuned to the utterance-length caption bands — keep the two blocks in sync. The app calls `st.set_page_config(page_title="Kokoro Studio", page_icon="🎙️", layout="wide")` as its first Streamlit command.

## Architecture

### Files

- `streamlit_app.py` — main app: language selector, text input + per-language sample buttons + Tokenize button + utterance-length caption + pronunciation note (left column), gender segmented control + per-card voice grid with per-card Play button + speed dropdown + inline audio playback + download button (right column)
- `voice_grades.py` — quality-grade table (`VOICE_GRADES`), rank table (`_GRADE_RANK`), and `_grade_rank` helper extracted from the Kokoro model card; consumed by the voice picker for sorting and labeling
- `samples/` — bundled public-domain sample text per language (9 directories × 3 files: `random.txt` quote pool plus two literary excerpts); referenced by `SAMPLE_BUTTONS` and read by `_load_sample`
- `.python-version` — pins the interpreter to `3.12` so local dev matches CI, `requires-python`, and ty (see Installation)
- `.streamlit/config.toml` — server config (`fileWatcherType = "none"`) plus the "Kokoro indigo" `[theme]` with `[theme.light]`/`[theme.dark]` blocks (so the toolbar mode toggle appears); the only `.streamlit/` file checked in (a `.gitignore` exception)
- `.claude/settings.json` — shared Claude Code guardrails, the only `.claude/` file checked in (a `.gitignore` exception; per-user `settings.local.json` stays ignored). Two parts: (1) `permissions.deny` blocks Edit/Write/MultiEdit on `uv.lock` (change deps via `uv add`/`uv remove`/`uv lock`) and `permissions.ask` prompts before editing `.python-version` — **ask**, not deny, because Installation documents changing the pin as a supported operation, it just needs the ~1 GB UniDic re-download acknowledged first. These are native permission rules rather than a `PreToolUse` hook so they cover every file-editing tool, survive case-insensitive path variants, and get Claude Code's built-in deny-circumvention detection (which watches for `sed -i`/`cat >`/`uv python pin` routing around a rule via Bash — a hook-based guard gets none of that). (2) A `PostToolUse` hook running `ruff check --fix` **then** `ruff format` (that order per the astral ruff skill — lint fixes reorder imports, formatting then cleans up) on `.py` files, scoped to paths under `$CLAUDE_PROJECT_DIR` so out-of-tree scratch files are never rewritten. Both ruff steps always run and the hook exits `2` if either fails, so syntax errors and unfixable lint surface to the agent instead of being swallowed
- `.github/workflows/ci.yml` — both workflows the repo has, as two jobs (see Commands › CI and › Releases). `lint-typecheck-test`: merge gate on `macos-latest`, `ruff check` / `ruff format --check` / `ty check` / `pytest` over the unit suite only, `contents: read`. `release`: `ubuntu-latest`, `needs` the gate, main-push only, `contents: write`, tag → draft → publish when `pyproject.toml`'s version has no tag yet
- `.github/dependabot.yml` — weekly `github-actions` update PRs (`commit-message.prefix: "ci"`). This is the refresh mechanism for pins that cannot float: setup-uv publishes no major tags past v7, and the release job's `actions/checkout` is SHA-pinned. Python dependencies are deliberately **not** covered — those go through `uv add`/`uv lock`
- `LICENSE` — MIT (declared via `license = "MIT"` + `license-files = ["LICENSE"]` in `pyproject.toml`); runtime deps add GPLv3/LGPL terms — see README acknowledgements
- `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, and `huggingface_hub` for import; the `streamlit` mock provides identity-pass-through shims for `cache_resource`, `cache_data`, and `fragment` so decorated functions keep running their real bodies under test
- `tests/test_streamlit_app.py` — unit tests, including `seq`-ordered cache eviction/recency, the eviction protect set's displayed-key registration, `.streamlit/config.toml` theme validation (incl. the extrabold-h1 heading weight), project-description consistency across `pyproject.toml`/README/CLAUDE.md, MIT-license consistency across `LICENSE`/`pyproject.toml`/README (`TestLicensing`), the auto-release job's invariants (`TestReleaseWorkflow` — pyproject-version extractability and X.Y.Z shape, the `needs`/main-only gate, the SHA-pinned write-token checkout, draft-before-publish ordering, main runs never cancelled mid-publish, and that a tag-triggered `release.yml` stays retired), and the espeak-ng vendoring canaries (`TestEspeakVendoring` — bundled library + data dir, the misaki→phonemizer binding, the ~159-char `N_PATH_HOME` path budget, and a no-`brew install` guard on `ci.yml`)
- `tests_integration/conftest.py` — clears `streamlit`, `streamlit_app`, `misaki`, `mlx_audio`, and `huggingface_hub` from `sys.modules` so AppTest gets the real modules (`streamlit_app` needs its own prefix entry — the `streamlit` prefix doesn't match it — so the app is re-imported fresh under the real modules); incompatible with `tests/conftest.py`'s mocks in one process, so `testpaths = ["tests"]` keeps the integration suite opt-in via an explicit `uv run pytest tests_integration/`
- `tests_integration/test_app_integration.py` — AppTest integration tests: initial render, sample buttons, Tokenize/Play enablement, gender filter, language switching, per-card speed controls. Feeds `AppTest.from_file` an absolute `APP_PATH` (`Path(__file__).resolve().parent.parent / "streamlit_app.py"`) — streamlit 1.61 changed relative-path resolution from the working directory to the directory of the calling file, so the bare `"streamlit_app.py"` this used to pass now raises `FileNotFoundError`

### Key Functions

- `ensure_repo_downloaded` — calls `huggingface_hub.snapshot_download` once per process; tries `local_files_only=True` first and only shows the download spinner when files are missing
- `get_voices` — walks the local snapshot's `voices/` directory and returns voice IDs for the given language, sorted by quality grade (best first, ties alphabetical)
- `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model`; called lazily on the first per-card Play click. Must also set `model.repo_id = REPO_ID` before returning: mlx-audio's loader builds `Model(config)` without forwarding a repo id, so `Model.repo_id` stays `None` and its `_get_pipeline` falls back to the class constant `REPO_ID = "prince-canuma/Kokoro-82M"` when it creates (and caches) each per-language `KokoroPipeline`. Unpinned, every voice's first Play re-downloads a tensor already present in our snapshot and raises a raw `RuntimeError` when offline. Guarded by `TestLoadPipeline.test_pins_voice_repo_to_our_snapshot`
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
- `_stale_cached_key` — returns the session-state key of the most recently generated cached audio for a `(voice, text, lang)` regardless of speed (highest `seq`), or `None`; called directly by `render_voice_card` for the `:material/volume_up:` badge, the stale-preview player, and to register the displayed key in the protect map
- `_find_stale_cached_audio` — thin convenience wrapper over `_stale_cached_key` returning that key's `VoiceResult` (or `None`); retained for its unit tests (production reads the value inline)
- `_evict_old_audio` — caps the session-state audio cache at `AUDIO_CACHE_LIMIT` (20) by deleting the lowest-`seq` (oldest-generated) entries; ordering comes from `seq`, not iteration order. Takes an optional `protect` set of keys never to evict — the main body resets `st.session_state["_displayed_card_keys"]` (a `voice → displayed key` dict) to `{}` each full rerun, and each card's fragment writes the key whose audio it is actually showing (the current-speed take if present, else the stale-preview key — a possibly different speed); the Play handler passes those values (plus the just-written key) so a fragment's eviction can't orphan a sibling card's visible player, including a stale-preview player at another speed
- `_audio_to_wav_bytes` — encodes a float32 audio array to WAV bytes via `soundfile` for the per-card download button
- `render_voice_card` — `@st.fragment`-wrapped; renders one bordered card per voice so a Play/speed interaction reruns only that card, not the whole script. Title via `_format_voice` (prefixed with a `:material/volume_up:` icon when cached audio exists), a 50/50 inner row with a speed selectbox (left) and `Play` button (`icon=":material/play_arrow:"`, right). It registers the key whose audio it is displaying (current-speed take if present, else the stale-preview key) into `st.session_state["_displayed_card_keys"]` on every rerun, then shows an `st.audio` player and Download button (`icon=":material/download:"`) when audio for `_cache_key(...)` is in session state, or a stale-preview player + "Click Play to refresh (speed changed)" caption when only another speed is cached. Play click runs `generate_one`, stamps `seq`, stores the result, and calls `_evict_old_audio`.
- `_render_persistent_phonemes` — re-renders the `Phoneme Tokens` expander (open) when `last_phonemes` matches the current `(text, lang_code)`, so tokenized phonemes persist across reruns
- `render_phonemes` — renders the `Phoneme Tokens` expander with `st.code`; `expanded` flag toggles open state
- `_estimate_phonemes` — cheap char-count × per-language multiplier (English 0.85, Romance/Portuguese 0.90, Hindi/Italian 1.00, Japanese 1.40, Mandarin 2.00); used by the length caption when no tokenization has run yet
- `_phoneme_band` — returns `(color, label)` for a phoneme count: `<20` red "very short", `20–99` orange "short", `100–399` green "ideal", `400–509` orange "long", `≥510` red "will be chunked"
- `_render_length_caption` — colored `st.caption` under the textarea showing exact phoneme count when `last_phonemes` matches current `(text, lang_code)`, else `~estimate`
- `_load_sample` — `@st.cache_data`, reads `samples/{lang_code}/{filename}` resolved relative to `streamlit_app.py`
- `_pick_sample` — returns the full sample text or a random non-empty line from a one-per-line pool, depending on `is_random`; the random branch avoids repeating the previous pick by tracking it in `st.session_state["_last_random_{lang}_{filename}"]` (re-rolls only when the pool has more than one line)
- `_set_text_from_sample` — `on_click` callback that writes the chosen sample into `st.session_state["text_input"]`; must run between reruns to avoid Streamlit's "widget key already instantiated" error
- `_render_sample_buttons` — renders one row of buttons per language from `SAMPLE_BUTTONS[lang_code]`; each button wired via `on_click=_set_text_from_sample`

### Model

[Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) (`load_model` from `mlx_audio.tts.utils`), 82M params in bf16 precision. Sample rate: 24000 Hz. MLX backend for Apple Silicon.

### Supported Languages

a=American English, b=British English, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Brazilian Portuguese, z=Mandarin Chinese — 9 languages

### Voice Discovery

On first launch, `ensure_repo_downloaded` calls `huggingface_hub.snapshot_download` to fetch the model and all voice files in one event (~355 MB), with a spinner shown only when the local HuggingFace cache is incomplete (detected via `snapshot_download(..., local_files_only=True)` raising `LocalEntryNotFoundError`). `get_voices` then walks the local snapshot's `voices/` directory and sorts results by quality grade (best first, ties broken alphabetically) using the `VOICE_GRADES` table in `voice_grades.py` (sourced from [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)). Voice files follow the naming convention `{lang}{gender}_{name}` (e.g. `af_heart` — American English, female, "heart") with `.safetensors` extension. Ungraded voices (Spanish, Brazilian Portuguese, or any future addition not in `VOICE_GRADES`) sort to the end. After the initial download the app is fully offline — but only because `load_pipeline` pins `model.repo_id` to `REPO_ID`, so mlx-audio resolves voice tensors at Play time from this same snapshot instead of its hard-coded `prince-canuma/Kokoro-82M` fallback (see Key Functions › `load_pipeline`). Voices added upstream require clearing the HuggingFace cache to pick up.

### Performance

- MLX backend runs natively on Apple Silicon (no PyTorch or MPS fallback needed)
- `@st.cache_resource` caches the model globally, tokenizers per language, and the snapshot path returned by `ensure_repo_downloaded`
- `@st.cache_data` caches voice lists per language code (local filesystem walk, no TTL needed)
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
- Gender filter: a single `st.segmented_control("Gender", ["All", "Female", "Male"], default="All", required=True, key="gender")` with collapsed label (`required=True` keeps one segment always selected — no deselected/empty state). `"All"` → `None` (no filter, show all); `"Female"`/`"Male"` filter to that gender. Translation handled by `_gender_code_from_selection`.
- Voice cards rendered via `render_voice_card`. Top 6 voices (by grade, via `_split_voices_for_display`) visible directly; the rest sit behind an `st.expander("Show all voices", icon=":material/library_music:")`
- When no voices match the gender filter, `st.caption("No voices match this filter.")` renders in place

**Voice card (per voice, via `render_voice_card`, `@st.fragment`):**
- `st.container(border=True)` frame (created inside the function, so each card is its own fragment instance — never pass a pre-built container in)
- Bold title via `_format_voice` (e.g. `**Heart (female) — A**`), prefixed with a `:material/volume_up:` icon when `_stale_cached_key` finds any cached audio for this voice/text/lang
- 50/50 inner row via `st.columns([1, 1])`: speed selectbox on the left, Play button on the right
- Speed selectbox: `SPEED_OPTIONS` (`[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]`), default `1.0`, formatted as `"{x}x"` (e.g. `1.0x`), keyed `f"speed_{voice}"`, label hidden
- Play button: `Play` with `icon=":material/play_arrow:"`, `type="primary"`, `width="stretch"`, keyed `f"play_{voice}"`, `disabled=not text.strip()`. Click handler loads the model (cached), runs `generate_one`, stamps `seq` via `_next_audio_seq`, stores the result in `st.session_state[_cache_key(voice, text, speed, lang_code)]`, then calls `_evict_old_audio`
- When the current speed's cache key is present: an inline `st.audio` player plus a Download button (`f"download_{voice}"`, WAV via `_audio_to_wav_bytes`, file named `f"{voice}_{speed}x.wav"`). Multiple voices' audios coexist for A/B comparison on the same text. When only another speed is cached: a "Click Play to refresh (speed changed)" caption above a stale-preview `st.audio` player (no download)

**Audio cache lifecycle:** (mechanics live in Key Functions `_cache_key`/`_next_audio_seq`/`_evict_old_audio` and Performance — only the resulting UX is summarized here)
- Changing voice/text/speed/lang yields a new `_cache_key`, so the current-speed player disappears until Play is pressed again; a stale-preview player for a previously generated speed may still show via `_stale_cached_key`
- Cached audios persist across reruns triggered by other interactions, bounded at `AUDIO_CACHE_LIMIT` (20)

**Behavior:**
- On initial render, `ensure_repo_downloaded` may show an `st.spinner` (`~355 MB`) for the one-time model + voices download when the local HuggingFace cache is incomplete; otherwise no spinner appears
- If the first-launch download fails (e.g. offline with no cache), the script shows `st.error(...)` and halts via `st.stop()` instead of leaking a Python traceback
- Chunk-by-chunk generation progress appears via `st.status` inside the active card during the Play-triggered run
- A per-card `ValueError` whose message contains `"No audio generated"` (the only such error raised) surfaces as a clean `st.error(str(e))`; every other exception — including any other `ValueError` — falls through to `st.exception()`. Either way other cards remain functional

## Resources

- [MLX Model](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- [Original Model](https://github.com/hexgrad/kokoro)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)
