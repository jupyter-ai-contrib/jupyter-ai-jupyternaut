from __future__ import annotations
from asyncio import get_event_loop_policy
from jupyter_server.extension.application import ExtensionApp
from jupyter_server.serverapp import ServerApp

from traitlets import List, Unicode, Dict
from traitlets.config import Config
from typing import TYPE_CHECKING

from .config import ConfigManager, ConfigRestAPI
from .handlers import RouteHandler
from .jupyternaut import JupyternautPersona
from .models import ModelParametersRestAPI
from .secrets import EnvSecretsManager, SecretsRestAPI

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop


class JupyternautExtension(ExtensionApp):
    """
    The Jupyternaut server extension.

    This serves several REST APIs under the `/api/jupyternaut` route. Currently,
    for the sake of simplicity, they are hard-coded into the Jupyternaut server
    extension to allow users to configure the chat model & add API keys.

    In the future, these backend objects may be separated into other packages to
    allow other developers to re-use them in their personas.
    """

    name = "jupyter_ai_jupyternaut"
    handlers = [
        (r"api/jupyternaut/get-example/?", RouteHandler),
        (r"api/jupyternaut/config/?", ConfigRestAPI),
        (r"api/jupyternaut/model-parameters/?", ModelParametersRestAPI),
        (r"api/jupyternaut/secrets/?", SecretsRestAPI),
    ]

    allowed_providers = List(
        Unicode(),
        default_value=None,
        help="Identifiers of allowlisted providers. If `None`, all are allowed.",
        allow_none=True,
        config=True,
    )

    blocked_providers = List(
        Unicode(),
        default_value=None,
        help="Identifiers of blocklisted providers. If `None`, none are blocked.",
        allow_none=True,
        config=True,
    )

    allowed_models = List(
        Unicode(),
        default_value=None,
        help="""
        Language models to allow, as a list of global model IDs in the format
        `<provider>:<local-model-id>`. If `None`, all are allowed. Defaults to
        `None`.

        Note: Currently, if `allowed_providers` is also set, then this field is
        ignored. This is subject to change in a future non-major release. Using
        both traits is considered to be undefined behavior at this time.
        """,
        allow_none=True,
        config=True,
    )

    blocked_models = List(
        Unicode(),
        default_value=None,
        help="""
        Language models to block, as a list of global model IDs in the format
        `<provider>:<local-model-id>`. If `None`, none are blocked. Defaults to
        `None`.
        """,
        allow_none=True,
        config=True,
    )

    model_parameters = Dict(
        key_trait=Unicode(),
        value_trait=Dict(),
        default_value={},
        help="""Key-value pairs for model id and corresponding parameters that
        are passed to the provider class. The values are unpacked and passed to
        the provider class as-is.""",
        allow_none=True,
        config=True,
    )

    error_logs_dir = Unicode(
        default_value=None,
        help="""Path to a directory where the error logs should be
        written to. Defaults to `jupyter-ai-logs/` in the preferred dir
        (if defined) or in root dir otherwise.""",
        allow_none=True,
        config=True,
    )

    initial_chat_model = Unicode(
        default_value=None,
        allow_none=True,
        help="""
        Default language model to use, as string in the format
        <provider-id>:<model-id>, defaults to None.
        """,
        config=True,
    )

    initial_language_model = Unicode(
        default_value=None,
        allow_none=True,
        help="""
        Default language model to use, as string in the format
        <provider-id>:<model-id>, defaults to None.
        """,
        config=True,
    )

    default_embeddings_model = Unicode(
        default_value=None,
        allow_none=True,
        help="""
        Default embeddings model to use, as string in the format
        <provider-id>:<model-id>, defaults to None.
        """,
        config=True,
    )

    default_completions_model = Unicode(
        default_value=None,
        allow_none=True,
        help="""
        Default completions model to use, as string in the format
        <provider-id>:<model-id>, defaults to None.
        """,
        config=True,
    )

    default_api_keys = Dict(
        key_trait=Unicode(),
        value_trait=Unicode(),
        default_value=None,
        allow_none=True,
        help="""
        Default API keys for model providers, as a dictionary,
        in the format `<key-name>:<key-value>`. Defaults to None.
        """,
        config=True,
    )

    @property
    def event_loop(self) -> AbstractEventLoop:
        """
        Returns a reference to the asyncio event loop.
        """
        return get_event_loop_policy().get_event_loop()

    def initialize_settings(self):
        # Log traitlets configuration
        self.log.info(f"Configured provider allowlist: {self.allowed_providers}")
        self.log.info(f"Configured provider blocklist: {self.blocked_providers}")
        self.log.info(f"Configured model allowlist: {self.allowed_models}")
        self.log.info(f"Configured model blocklist: {self.blocked_models}")
        self.log.info(f"Configured model parameters: {self.model_parameters}")
        defaults = {
            "model_provider_id": self.initial_language_model,
            "embeddings_provider_id": self.default_embeddings_model,
            "completions_model_provider_id": self.default_completions_model,
            "api_keys": self.default_api_keys,
            "fields": self.model_parameters,
            "embeddings_fields": self.model_parameters,
            "completions_fields": self.model_parameters,
        }

        # Initialize ConfigManager
        config_manager = ConfigManager(
            config=self.config,
            log=self.log,
            allowed_providers=self.allowed_providers,
            blocked_providers=self.blocked_providers,
            allowed_models=self.allowed_models,
            blocked_models=self.blocked_models,
            defaults=defaults,
        )

        # Bind ConfigManager instance to global settings dictionary
        self.settings["jupyternaut.config_manager"] = config_manager

        # Bind ConfigManager instance to Jupyternaut as a class variable
        JupyternautPersona.config_manager = config_manager

        # Initialize SecretsManager and bind it to global settings dictionary
        self.settings["jupyternaut.secrets_manager"] = EnvSecretsManager(parent=self)


    def _link_jupyter_server_extension(self, server_app: ServerApp):
        """Setup custom config needed by this extension."""
        c = Config()
        c.ContentsManager.allow_hidden = True
        c.ContentsManager.hide_globs = [
            "__pycache__",  # Python bytecode cache directories
            "*.pyc",  # Compiled Python files
            "*.pyo",  # Optimized Python files
            ".DS_Store",  # macOS system files
            "*~",  # Editor backup files
            ".ipynb_checkpoints",  # Jupyter notebook checkpoint files
            ".git",  # Git version control directory
            ".venv",  # Python virtual environment directory
            "venv",  # Python virtual environment directory
            "node_modules",  # Node.js dependencies directory
            ".pytest_cache",  # PyTest cache directory
            ".mypy_cache",  # MyPy type checker cache directory
            "*.egg-info",  # Python package metadata directories
        ]
        server_app.update_config(c)
        super()._link_jupyter_server_extension(server_app)
