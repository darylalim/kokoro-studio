import sys

# This suite runs as a SEPARATE pytest invocation from tests/, not alongside
# it. tests/conftest.py replaces streamlit/misaki/mlx_audio/huggingface_hub
# with MagicMock in sys.modules; AppTest needs the real modules. The two
# states are mutually exclusive within one Python process, so testpaths in
# pyproject.toml is set to ["tests"] only, and this directory must be run
# explicitly: `uv run pytest tests_integration/`.
#
# The cleanup below is a defense-in-depth measure in case the unit conftest
# has somehow already loaded (e.g. if a user passes both paths on the CLI).
_LEAKED_PREFIXES = (
    "streamlit",
    "streamlit_app",
    "misaki",
    "mlx_audio",
    "huggingface_hub",
)

for _name in list(sys.modules):
    if any(_name == p or _name.startswith(p + ".") for p in _LEAKED_PREFIXES):
        del sys.modules[_name]
