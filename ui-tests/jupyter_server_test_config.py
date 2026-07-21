"""Server configuration for integration tests.

!! Never use this configuration in production because it
opens the server to the world and provide access to JupyterLab
JavaScript objects through the global window variable.
"""
import os
import sys
import tempfile
from pathlib import Path

from jupyterlab.galata import configure_jupyter_server

configure_jupyter_server(c)

# Uncomment to set server log level to debug level
# c.ServerApp.log_level = "DEBUG"

# Isolate Jupyternaut's config store. Jupyternaut writes its config (custom
# models, API keys, ...) to `<jupyter_data_dir>/jupyter_ai/config.json`. Point
# JUPYTER_DATA_DIR at a fresh temp dir so tests start from an empty config and
# never touch the developer's real one. Set before any Jupyter import reads it.
os.environ.setdefault(
    "JUPYTER_DATA_DIR", tempfile.mkdtemp(prefix="jupyternaut-e2e-data-")
)

# Register the fake LiteLLM provider in this server process, so a test-only
# custom model (model ID `fake-litellm/echo`) streams a deterministic canned
# reply with no real model or credentials. See fake_litellm_provider.py.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from fake_litellm_provider import register as register_fake_provider  # noqa: E402

register_fake_provider()
