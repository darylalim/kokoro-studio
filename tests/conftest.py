import os
import sys
import tempfile
from unittest.mock import MagicMock

# Mock streamlit to prevent UI initialization on import
_st = MagicMock()
_st.cache_resource = lambda f: f
_st.cache_data = lambda *args, **_kw: args[0] if args else (lambda f: f)
# @st.fragment must pass the decorated function through unchanged, supporting
# both bare `@st.fragment` and parametrized `@st.fragment(...)` forms.
_st.fragment = lambda func=None, **_kw: func if func is not None else (lambda g: g)
_st.selectbox.side_effect = lambda label, **_kw: {
    "Language": "American English",
    "Speed": 1.0,
}.get(label, MagicMock())
_st.segmented_control.side_effect = lambda label, **_kw: (
    "All" if label == "Gender" else MagicMock()
)
_st.button.return_value = False
_st.text_area.return_value = ""
_st.columns.side_effect = lambda spec: [
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
