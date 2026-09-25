import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

# Options the app passed to each Streamlit decorator, recorded as it was
# applied: {decorator name: {decorated function name: kwargs}}. The shim below
# has to swallow those kwargs to hand the bare function back, which leaves a
# test no other way to see them — and `show_spinner` in particular is a
# deliberate per-function setting, not a default. See TestCacheSpinners.
DECORATOR_KWARGS: dict[str, dict[str, dict[str, Any]]] = {}


def _passthrough_decorator(name: str) -> Callable[..., Any]:
    """Return a shim for one Streamlit decorator that hands the function back.

    Streamlit's decorators are used in two shapes and the app uses both:
    `@st.cache_resource` applies the decorator straight to the function, while
    `@st.cache_resource(show_spinner=...)` calls it first and applies what it
    returns. So the shim must accept being handed either the function or the
    options — and it records the options on the way past.
    """
    recorded = DECORATOR_KWARGS.setdefault(name, {})

    def decorator(*args: Any, **kwargs: Any) -> Any:
        def apply(func: Any) -> Any:
            recorded[func.__name__] = kwargs
            return func

        return apply(args[0]) if args else apply

    return decorator


# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = _passthrough_decorator("cache_resource")
_st.cache_data = _passthrough_decorator("cache_data")
_st.fragment = _passthrough_decorator("fragment")
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "American English",
    "Speed": 1.0,
}.get(label, MagicMock())
_st.segmented_control.side_effect = lambda label, **_kw: (
    "All" if label == "Voice gender" else MagicMock()
)
_st.button.return_value = False
_st.text_area.return_value = ""
_st.columns.side_effect = lambda spec, **_kw: [
    MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
]
_st.session_state = {}
sys.modules["streamlit"] = _st

# Mock mlx_audio to prevent model downloads on import
_mlx_audio = MagicMock()
_mlx_audio_tts = MagicMock()
_mlx_audio_tts_utils = MagicMock()
sys.modules["mlx_audio"] = _mlx_audio
sys.modules["mlx_audio.tts"] = _mlx_audio_tts
sys.modules["mlx_audio.tts.utils"] = _mlx_audio_tts_utils

# Mock misaki to prevent espeak-ng dependency in tests
_misaki = MagicMock()
_misaki_en = MagicMock()
_misaki_ja = MagicMock()
_misaki_zh = MagicMock()
_misaki_espeak = MagicMock()
_misaki.en = _misaki_en
_misaki.ja = _misaki_ja
_misaki.zh = _misaki_zh
_misaki.espeak = _misaki_espeak
sys.modules["misaki"] = _misaki
sys.modules["misaki.en"] = _misaki_en
sys.modules["misaki.ja"] = _misaki_ja
sys.modules["misaki.zh"] = _misaki_zh
sys.modules["misaki.espeak"] = _misaki_espeak

# Mock huggingface_hub.snapshot_download by populating a tmpdir with empty
# voice files so get_voices() can do a real directory walk.
_voices_tmp = tempfile.mkdtemp(prefix="kokoro_test_voices_")
os.makedirs(os.path.join(_voices_tmp, "voices"), exist_ok=True)
for _fname in [
    "af_heart.safetensors",
    "af_bella.safetensors",
    "am_adam.safetensors",
    "bf_alice.safetensors",
    "bm_daniel.safetensors",
    "jf_alpha.safetensors",
    "zf_xiaobei.safetensors",
    "ef_dora.safetensors",
    "ff_siwis.safetensors",
    "hf_alpha.safetensors",
    "if_sara.safetensors",
    "pf_dora.safetensors",
]:
    open(os.path.join(_voices_tmp, "voices", _fname), "w").close()

_hf_hub = MagicMock()
_hf_hub.snapshot_download.return_value = _voices_tmp
sys.modules["huggingface_hub"] = _hf_hub


# huggingface_hub.errors.LocalEntryNotFoundError must be a real exception class
# (the app catches it); MagicMock attributes can't be used in `except` clauses.
class _MockLocalEntryNotFoundError(Exception):
    pass


_hf_hub_errors = MagicMock()
_hf_hub_errors.LocalEntryNotFoundError = _MockLocalEntryNotFoundError
sys.modules["huggingface_hub.errors"] = _hf_hub_errors
