# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

Streamlit application for generating multilingual speech using [Hexgrad Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) on Apple Silicon with MLX.

> Keep the tagline above byte-identical (markdown links stripped) to `pyproject.toml`'s `description` and the README tagline. `TestProjectMetadata.test_description_in_sync_with_docs` finds it by substring and compares the **whole line**, so nothing may be appended to it.

## Commands

| | |
|---|---|
| Run the app | `uv run streamlit run streamlit_app.py` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Typecheck | `uv run ty check` |
| Unit tests | `uv run pytest` |
| Integration tests | `uv run pytest tests_integration/` |

## Installation

No system dependencies — espeak-ng is vendored (see Dependencies), so `uv sync` is sufficient.

```bash
uv sync --group dev
uv run python -m unidic download   # one-time, ~1 GB, only needed for Japanese
uv run streamlit run streamlit_app.py
```

`.python-version` pins the interpreter to **3.12**, matching CI's `setup-uv` input, the `requires-python = ">=3.12"` floor, and `[tool.ty.environment]` — all four asserted by `TestPythonVersionConsistency`. Two things not to undo:

- **Don't delete the pin.** `uv sync` would fall back to interpreter discovery and take the first candidate satisfying `requires-python`, silently drifting local dev off the version CI gates on.
- **Don't raise ty's target above the floor** even when running a newer interpreter locally — type-checking against the declared floor is what catches accidental use of newer-only APIs.

Changing the pin makes the next `uv sync` rebuild the venv, which discards the ~1 GB UniDic dictionary (it installs into the venv's `unidic` package directory). Re-run the download afterwards if you use Japanese.

## Dependencies

**System: none.** espeak-ng is *not* a system dependency here — `espeakng-loader` ships a prebuilt `libespeak-ng.dylib` plus `espeak-ng-data` inside the wheel, and `misaki/espeak.py` calls `EspeakWrapper.set_library()` / `set_data_path()` on those at import. phonemizer's `EspeakWrapper.library` resolves lazily and checks its `_ESPEAK_LIBRARY` class attribute *first*, ahead of `$PHONEMIZER_ESPEAK_LIBRARY` and `ctypes.util.find_library('espeak-ng')`, so the system lookup is unreachable once misaki has been imported. This covers synthesis too: mlx-audio's kokoro pipeline imports `misaki.espeak` rather than touching phonemizer directly, and nothing in mlx-audio shells out to an `espeak` binary. **`brew install espeak-ng` therefore changes nothing — don't add it to setup docs or CI.** `TestEspeakVendoring` fails the build if a `brew install` reappears in `ci.yml`, if the wheel stops shipping the library/data dir, or if misaki stops binding phonemizer to it.

> **Keep the venv path short.** espeak-ng stores its data directory in a fixed 160-byte `N_PATH_HOME` buffer. If the resolved `espeakng_loader/espeak-ng-data` path exceeds it, the path is truncated, espeak falls back to the build-machine path baked into the wheel (`/Users/runner/work/espeakng-loader/...`), fails to find `phontab`, and calls `exit(1)` directly — the process dies with **exit code 1, no Python traceback, no pytest summary, and buffered output lost**. It reads like a mysterious interpreter-specific crash. To check a given checkout, measure the resolved data directory — `<venv>/lib/pythonX.Y/site-packages/espeakng_loader/espeak-ng-data` — and keep it comfortably under ~159 characters; a `.venv` in the repo root is normally fine, while a venv under a deeply nested CI cache or temp directory may not be. (The 160-byte figure is inferred from the observed truncation boundary, not read from espeak-ng's source, so treat it as approximate.) Since `misaki/espeak.py` calls `EspeakWrapper.set_data_path()` at import time, a later override cannot fix it — espeak is already initialized.

**Runtime:** `en-core-web-sm` (pinned URL; update it if spaCy is upgraded), `espeakng-loader`, `misaki[ja]`, `misaki[zh]`, `mlx-audio`, `num2words`, `numpy`, `phonemizer-fork`, `soundfile`, `spacy`, `streamlit`. The English G2P stack (`spacy`, `num2words`, `phonemizer-fork`, `espeakng-loader`) is pulled in directly rather than via `misaki[en]` to skip `spacy-curated-transformers`, which drags in `torch`; both are absent from `uv.lock` as a result. (`transformers` is still present, pulled directly by `mlx-audio`/`mlx-lm`.) Japanese needs the one-time UniDic download.

**Dev:** `ruff`, `ty`, `pytest`

## Configuration

`pyproject.toml` — project metadata, dependencies, dependency groups, ruff isort (`combine-as-imports`), pytest (`pythonpath`, `testpaths`), ty (`python-version = "3.12"`).

`.streamlit/config.toml` — `[server] fileWatcherType = "none"` plus the "Kokoro indigo" `[theme]`: indigo `primaryColor` (`#4F46E5`, AA-contrast with white button text), `8px` `baseRadius`, `linkUnderline = false`, Inter body font (weights to 800) with `headingFontWeights = [800, 600, …]` — an extrabold h1 above Streamlit's 700 default, h2–h6 at the 600 default, headings reuse the body Inter so no separate `headingFont` — and JetBrains Mono `codeFont`. Both font families are `fonts.googleapis.com` stylesheet URLs fetched by the browser per page load; offline the page falls back to system fonts (the offline guarantee covers the model/Python side only). `[theme.light]`/`[theme.dark]` blocks define both modes so the toolbar mode toggle appears: each sets a full GitHub-Primer base palette (`backgroundColor`/`secondaryBackgroundColor`/`textColor`/`borderColor`) plus `redColor`/`orangeColor`/`greenColor` tuned to the utterance-length caption bands — keep the two blocks in sync. Streamlit has no bare `red`/`orange`/`green` options; unsuffixed keys are silently ignored.

## CI & Releases

`.github/workflows/ci.yml` holds both jobs. **Version-specific detail lives in the workflow's own comments — change it there, not here, so there is one place to update per bump.**

### `lint-typecheck-test` (merge gate)

Push to `main` + every PR, `macos-latest` only (mlx/mlx-metal are `darwin`-gated in `uv.lock`), `timeout-minutes: 20`, `contents: read`. Runs `uv sync --locked --group dev` (fails on lockfile drift) → `ruff check .` → `ruff format --check .` → `ty check` → `pytest`. Note CI gates on `ruff format --check .`, not the bare `ruff format .`. Steps omit `name:` so GitHub labels them with the command itself. `tests_integration/` is excluded (needs the real modules + the ~355 MB download). Keep `python-version: "3.12"` in the `setup-uv` block — `TestPythonVersionConsistency.test_ci_workflow_matches` asserts on it.

`cancel-in-progress` is the expression `${{ github.ref != 'refs/heads/main' }}`, never `true`: superseded PR runs are still cancelled, but cancelling a main run between the release job's tag push and its publish would strand a tag with no release. The same concurrency group without cancellation makes main runs *queue*, so a later run always observes the tag an earlier one pushed. (`TestReleaseWorkflow.test_main_runs_are_never_cancelled_mid_publish`)

Three standing policies:

- **No system packages.** See Dependencies. Guard: `TestEspeakVendoring`.
- **Pins do not float.** `astral-sh/setup-uv` carries a full version (it has published no major tags since v7). `actions/checkout` is pinned per job by token scope — a floating major tag is repointable upstream, so the `contents: write` release job gets a SHA. Guard: `TestReleaseWorkflow.test_release_job_checkout_is_sha_pinned`. None of these refresh themselves, which is what `.github/dependabot.yml` is for; `uv`'s own pin should track the local `.venv/pyvenv.cfg`.
- **Cache is pruned explicitly.** setup-uv v9 flipped `prune-cache` to `false`; at that default this repo would save its entire uv cache per lockfile hash against GitHub's per-repo quota, so `ci.yml` sets it back to `true`. The cost runs the other way: `uv sync` re-fetches wheels from PyPI instead of reading them back out of the cache.

### `release` job

Fully automatic on a version bump. The flow is just **bump `version` in `pyproject.toml` → `uv lock` → commit → push to `main`**. No manual tagging.

`ubuntu-latest` (no build/test — that is the other job's), `needs: lint-typecheck-test` so a red tree cannot ship, gated `if: github.event_name == 'push' && github.ref == 'refs/heads/main'` so PRs — fork PRs especially — never reach the `contents: write` token. That token is granted **per job**, so `lint-typecheck-test` (which executes third-party code via `uv sync`) stays `contents: read` (`TestReleaseWorkflow.test_release_job_mirrors_version_guard`). Three steps: annotated `git tag -a vX.Y.Z` + push → `gh release create --draft --notes-file … --verify-tag` → `gh release edit --draft=false`. Drafting first means a failure between steps leaves an invisible draft, not a half-formed public release; GitHub auto-marks the newest published release "Latest", so no `--latest`.

Three decisions that are easy to undo by accident:

- **Bump detection asks whether the tag and release already exist** — never diffs against the parent commit, which breaks on squash merges, multi-commit pushes, and re-runs. So every push to `main` enters the job and the common case is a three-step skip. States: `absent` (tag — only when `tag_exists == 'false'`, so a mid-flight re-run drafts and publishes without re-tagging — then draft, then publish), `draft` (publish only, finishing a job that died mid-flight), `published` (skip). Version must be exactly `X.Y.Z` or the job errors (`TestReleaseWorkflow.test_pyproject_version_is_grep_extractable`). For prereleases, add PEP 440-aware normalisation and pass `--prerelease`.
- **Notes come from `release_notes.py`, not `--generate-notes`.** That flag enumerates merged *pull requests* only, and this repo pushes straight to `main` and has never opened one, so it produced nothing but a compare link no matter what shipped. Hence `fetch-depth: 0` on the release checkout (the script reads `git log <previous>..<tag>` locally) and a `previous` output on the `plan` step resolved from `releases/latest`, drafts and prereleases excluded. This only works while commit subjects stay conventional, which `TestWorkflowBinding.test_repo_history_is_fully_categorised` enforces over the last 40 commits.
- **There is deliberately no tag-triggered `release.yml`.** It published from a hand-pushed tag without consulting CI, and a second publish path risks double-cutting one version. `TestReleaseWorkflow.test_tag_triggered_release_workflow_stays_retired` fails the build if it reappears. To re-cut a release, delete the GitHub release (and its tag) and re-run the CI job.

## Code Style

- snake_case for functions/variables, PascalCase for classes; type annotations on all parameters and returns; isort with `combine-as-imports`.
- `pyproject.toml` sets no `[tool.ruff.lint] select`, so the project runs on **ruff's default rule set** — which ruff 0.16 widened from 59 rules to 413 (this is what first enabled the `I` isort rules the repo had configured but never enforced). A ruff upgrade can therefore surface new findings with no code change; run `ruff check .` after every bump rather than assuming a clean tree stays clean.
- Every `except Exception` in `streamlit_app.py` carries `# noqa: BLE001` deliberately — the first-launch download guard, the per-card Play handler, and the Tokenize handler. Each must degrade in place (`st.error` / in-card `st.exception` / `_tokenize_error_message`) instead of leaking a traceback into the UI. Don't narrow them to satisfy the linter.
- **Asserting about a config file:** parse it where a parser exists (`tomllib` for `.streamlit/config.toml` and `pyproject.toml`). Where you must substring-match — `ci.yml` — match against a **comment-stripped** view, never the raw text; `TestReleaseWorkflow._release_job_directives` is the helper (copies live in `TestEspeakVendoring` and `tests/test_release_notes.py`). This repo documents each policy in prose directly above the line implementing it, so a raw check is satisfied by the explanation whether or not the policy survives: `assert "contents: write" in job` passed against the comment `# ... this job runs with \`contents: write\``, and so stayed green when the `permissions` block was deleted *or* hoisted workflow-wide. Anchor flags too — `assert "--draft" in job` is a substring of the publish step's `--draft=false`. Verify such a test by deleting the line it guards and confirming it fails; "it passes" proves nothing here.
- When working with Python, invoke the relevant `/astral:<skill>` for uv, ty, and ruff.

## Architecture

### Files

- `streamlit_app.py` — the whole app: config constants, G2P, synthesis, and rendering. See Key Functions and UI below.
- `release_notes.py` — builds every GitHub Release body from conventional-commit subjects: `parse_commit` classifies a subject, `render` groups them under `SECTIONS`. Two deliberate choices: it is **stdlib only**, so the release job needs no Python setup beyond the runner image's `python3`; and an unrecognised subject is kept **verbatim** under `Other Changes` rather than dropped or de-prefixed, because a silently omitted commit understates the release with no signal that it did. Breaking changes lead *and* repeat under their own type, so skimming headings cannot miss one.
- `voice_grades.py` — `VOICE_GRADES`, `_GRADE_RANK`, and `_grade_rank`, extracted from the Kokoro model card; consumed by the voice picker for sorting and labeling.
- `samples/` — public-domain sample text, 9 directories × 3 files: `random.txt` (one quote per line, a pool) plus two literary excerpts. Referenced by `SAMPLE_BUTTONS`, read by `_load_sample`.
- `README.md` — user-facing docs (install, usage, troubleshooting), the screenshots, and the third-party GPLv3/LGPL acknowledgements. It duplicates several facts held here, so update both; its tagline, MIT badge and license link are asserted by `TestProjectMetadata`/`TestLicensing`.
- `assets/` — `screenshot-light.png` / `screenshot-dark.png`, embedded side-by-side in README with layout-describing alt text. Changes to the UI section below make these stale; recapture in both themes.
- `LICENSE` — MIT (declared via `license = "MIT"` + `license-files = ["LICENSE"]` in `pyproject.toml`); runtime deps add GPLv3/LGPL terms — see README acknowledgements.
- `.python-version`, `.streamlit/config.toml`, `.github/workflows/ci.yml` — see Installation, Configuration, and CI & Releases.
- `.github/dependabot.yml` — weekly `github-actions` update PRs (`commit-message.prefix: "ci"`). This is the refresh mechanism for the pins that cannot float. Python dependencies are deliberately **not** covered — those go through `uv add`/`uv lock`.
- `.claude/settings.json` — shared guardrails, the only `.claude/` file checked in (a `.gitignore` exception; per-user `settings.local.json` stays ignored). `permissions.deny` blocks edits to `uv.lock` (change deps via `uv add`/`uv remove`/`uv lock`); `permissions.ask` prompts on `.python-version` — **ask**, not deny, because changing the pin is supported, it just needs the UniDic re-download acknowledged first. Keep both as **native permission rules, not a `PreToolUse` hook**: only native rules cover every file-editing tool, survive case-insensitive path variants, and get built-in deny-circumvention detection (`sed -i`/`cat >`/`uv python pin` routed via Bash). A `PostToolUse` hook then runs `ruff check --fix` **then** `ruff format` (that order: lint fixes reorder imports, formatting cleans up) on `.py` files under `$CLAUDE_PROJECT_DIR`, exiting `2` if either fails so unfixable lint surfaces to the agent instead of being swallowed.
- `tests/conftest.py` — mocks `streamlit`, `mlx_audio`, `misaki`, `huggingface_hub` for import; the `streamlit` mock gives `cache_resource`, `cache_data`, and `fragment` identity pass-through shims so decorated functions keep running their real bodies.
- `tests/test_streamlit_app.py` — the unit suite. Config-file invariants live here alongside the app tests: `TestPythonVersionConsistency`, `TestProjectMetadata`, `TestLicensing`, `TestReleaseWorkflow`, `TestEspeakVendoring`, and the theme validation.
- `tests/test_release_notes.py` — the only thing exercising `release_notes.py` (it runs solely inside the release job, where a bad body is visible only after publishing). `TestWorkflowBinding` ties `ci.yml` to the script.
- `tests_integration/conftest.py` — clears `streamlit`, `streamlit_app`, `misaki`, `mlx_audio`, `huggingface_hub` from `sys.modules` so AppTest gets the real modules (`streamlit_app` needs its own prefix entry — the `streamlit` prefix doesn't match it). Incompatible with `tests/conftest.py`'s mocks in one process, so `testpaths = ["tests"]` keeps this suite opt-in.
- `tests_integration/test_app_integration.py` — AppTest tests over initial render, sample buttons, Tokenize/Play enablement, gender filter, language switching, per-card speed. Feeds `AppTest.from_file` an **absolute** `APP_PATH` — streamlit 1.61 changed relative-path resolution from the working directory to the directory of the calling file, so the bare `"streamlit_app.py"` this used to pass now raises `FileNotFoundError`.

### Key Functions

Only the entries with a non-obvious contract are listed; the rest of `streamlit_app.py` (`get_voices`, `load_tokenizer`, `_create_g2p`, `tokenize_text`, `generate_speech`, `generate_one`, `_format_voice`, `_filter_voices_by_gender`, `_gender_code_from_selection`, `_audio_to_wav_bytes`, `render_phonemes`, `_render_persistent_phonemes`, `_render_length_caption`, `_load_sample`, `_render_sample_buttons`) does what its name says.

- `load_pipeline` — cached global model via `mlx_audio.tts.utils.load_model`, deferred to the first Play click. **Must set `model.repo_id = REPO_ID` before returning:** mlx-audio's loader builds `Model(config)` without forwarding a repo id, so `Model.repo_id` stays `None` and its `_get_pipeline` falls back to the class constant `REPO_ID = "prince-canuma/Kokoro-82M"` when it creates (and caches) each per-language `KokoroPipeline`. Unpinned, every voice's first Play re-downloads a tensor already present in our snapshot and raises a raw `RuntimeError` when offline. Guarded by `TestLoadPipeline.test_pins_voice_repo_to_our_snapshot`.
- `ensure_repo_downloaded` — `huggingface_hub.snapshot_download` once per process; tries `local_files_only=True` first so the ~355 MB spinner appears only when files are actually missing.
- `_tokenize_error_message` — friendly text for a Tokenize failure. `lang_code == "j"` plus "unidic" in the error returns the `uv run python -m unidic download` recovery command (the dictionary lives in the venv and vanishes on any rebuild); everything else gets `Could not tokenize this text: {error}`. The hint is Japanese-only on purpose — `TestTokenizeErrorMessage` fails if another language claims it.
- `_text_digest` — stable 16-hex-char `hashlib.sha1` digest, so cache keys are reproducible across processes (unlike the previously used `hash()`, which is per-process randomized).
- `_cache_key` — `f"audio:{voice}:{lang_code}:{speed}:{_text_digest(text)}"`. Cache invalidates implicitly when any of voice/text/speed/lang changes.
- `_next_audio_seq` — monotonic counter from `st.session_state["_audio_seq"]`, stamped onto each `VoiceResult` at write time so cache ordering never relies on `st.session_state` iteration order (a hash-ordered set in real Streamlit, not insertion order).
- `_stale_cached_key` — key of the most recently generated audio for a `(voice, text, lang)` regardless of speed (highest `seq`), or `None`. Drives the cached badge, the stale-preview player, and the protect-map registration. `_find_stale_cached_audio` is a thin wrapper returning that key's value, retained for its unit tests (production reads the value inline).
- `_evict_old_audio` — caps the session-state cache at `AUDIO_CACHE_LIMIT` (20), deleting lowest-`seq` first. Takes an optional `protect` set: the main body resets `st.session_state["_displayed_card_keys"]` (a `voice → displayed key` dict) each full rerun and every card's fragment writes the key it is actually showing, so a fragment's eviction can't orphan a sibling card's visible player — including a stale-preview player at another speed.
- `render_voice_card` — `@st.fragment`, so a Play or speed interaction reruns only that card. It opens its own `st.container(border=True)` **inside the function**; never pass a pre-built container in, or the cards share a fragment instance.
- `_split_voices_for_display` — returns `(visible, hidden)`, top `top_n` (default 6) visible. A `selected` voice landing in the tail is pinned into the visible section — a path production never takes (the app always passes `None`), kept for `TestSplitVoicesForDisplay`.
- `_estimate_phonemes` — stripped char count × `_PHONEME_MULTIPLIERS` (`a`/`b` 0.85, `e`/`f`/`p` 0.90, `h`/`i` 1.00, `j` 1.40, `z` 2.00; unknown code → 1.0). Used by the length caption before any tokenization has run.
- `_phoneme_band` — `(color, label)` for a phoneme count, per VOICES.md guidance: `<20` red "very short", `20–99` orange "short", `100–399` green "ideal", `400–509` orange "long", `≥510` red "will be chunked".
- `_pick_sample` — full excerpt, or a random non-empty line from a `random.txt` pool. The random branch avoids repeating the previous pick via `st.session_state["_last_random_{lang}_{filename}"]`, re-rolling only when the pool has more than one line.
- `_set_text_from_sample` — `on_click` callback writing into `st.session_state["text_input"]`. It **must** run between reruns, or Streamlit raises "widget key already instantiated".

### Model, Languages & Voices

[Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) via `load_model` from `mlx_audio.tts.utils` — 82M params in bf16, 24000 Hz, MLX backend for Apple Silicon. Nine languages: `a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese, `p` Brazilian Portuguese, `z` Mandarin Chinese.

**Adding a language touches five places in lockstep:** `LANGUAGES`, `ESPEAK_LANGUAGES` (unless it has a dedicated misaki G2P like `j`/`z`), `_PHONEME_MULTIPLIERS`, `SAMPLE_BUTTONS` + three files in `samples/<code>/`, and `VOICE_GRADES` if graded. `test_language_count` hard-codes 9 and the `*_covers_all_languages` tests fail on any partial edit.

Voice files are named `{lang}{gender}_{name}` (e.g. `af_heart`) with a `.safetensors` extension. `get_voices` walks the snapshot's `voices/` directory and sorts by quality grade from `VOICE_GRADES` (sourced from [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)), ties alphabetical; ungraded voices (Spanish, Brazilian Portuguese, anything new) sort last. Voices added upstream need the HuggingFace cache cleared to appear. After the initial download the app is fully offline — but only because `load_pipeline` pins `model.repo_id`.

### Performance

- MLX runs natively on Apple Silicon — no PyTorch or MPS fallback.
- `@st.cache_resource` for the model, the per-language tokenizers, and the snapshot path; `@st.cache_data` for per-language voice lists (a local filesystem walk, no TTL needed).
- Model load is deferred to the first Play click, so initial render isn't blocked.
- `generate_speech` uses `np.asarray(..., dtype=np.float32)` to avoid copying chunks that are already float32.
- `render_voice_card` being a fragment means a Play click reruns one card, not the script. One accepted trade-off: a sibling card's cached badge refreshes on the next full rerun, not instantly.

### UI

Layout is `st.set_page_config(layout="wide", …)` (first command) → `st.title` → a full-width `Language` selectbox → a two-column split. Generated audio renders inline inside each card; there is no separate output row.

**Left column:** a tall `st.text_area` (no character cap) → a row of three sample buttons per language from `SAMPLE_BUTTONS` (random-quote plus two excerpts, each wired `on_click=_set_text_from_sample`) → a `Tokenize` button, disabled on empty text, storing `(text, lang_code, phonemes)` in `st.session_state["last_phonemes"]` → the utterance-length caption (exact count when `last_phonemes` matches the current `(text, lang_code)`, else `~estimate`) → the phoneme expander, which opens itself on a match so tokenized output survives reruns → the always-visible `PRONUNCIATION_TIPS` note.

**Right column:** an `st.segmented_control` gender filter with `required=True`, so one segment is always selected and there is no empty state; `"All"` maps to `None` via `_gender_code_from_selection`. Then the voice cards — top 6 by grade visible, the rest behind a "Show all voices" expander, or a caption when the filter matches nothing.

**Voice card:** title (with a cached-audio badge) over a 50/50 row of speed selectbox and Play button, both keyed per voice. Play generates at the selected speed, stamps `seq`, stores under `_cache_key`, and evicts. Below that: the player plus a WAV download when the current speed is cached, or a stale-preview player and a "speed changed" caption when only another speed is. Multiple voices' audio coexists for A/B comparison on the same text, bounded at `AUDIO_CACHE_LIMIT`.

**Failure behavior** — every path degrades in place rather than replacing the page:

- First-launch download failure (offline with no cache): `st.error` then `st.stop()`.
- Tokenize failure: `st.error(_tokenize_error_message(...))`, which upgrades the Japanese missing-UniDic case to a recovery command.
- Per-card generation: a `ValueError` containing `"No audio generated"` (the only one raised) becomes a clean `st.error`; anything else — including any other `ValueError` — falls through to `st.exception`. Other cards stay functional either way.
- Chunk-by-chunk progress appears via `st.status` inside the active card.

## Resources

- [Original model](https://github.com/hexgrad/kokoro) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)
