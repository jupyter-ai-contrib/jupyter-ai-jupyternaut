from __future__ import annotations
import asyncio
from pathlib import Path
import pytest
from traitlets.config import Config, LoggingConfigurable
import logging
from jupyter_server.services.contents.filemanager import AsyncFileContentsManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jupyter_server.serverapp import ServerApp


pytest_plugins = ("pytest_jupyter.jupyter_server", )


@pytest.fixture
def jp_server_config(jp_server_config, tmp_path):
    return Config({"ServerApp": {"jpserver_extensions": {"jupyter_ai_jupyternaut": True}}, "ContentsManager": {"root_dir": str(tmp_path)}})


class MockJupyternautExtension(LoggingConfigurable):
    """Mock AiExtension class for testing purposes."""
    
    serverapp: ServerApp

    def __init__(self, *args, serverapp: ServerApp, **kwargs):
        super().__init__(*args, **kwargs)
        self.serverapp = serverapp
        self._log = None
        
    @property
    def log(self) -> logging.Logger:
        return self.serverapp.log
        
    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        return self.serverapp.io_loop.asyncio_loop
    
    @property
    def contents_manager(self) -> AsyncFileContentsManager:
        return self.serverapp.contents_manager


@pytest.fixture
def mock_extension(jp_server_config, jp_configurable_serverapp) -> MockJupyternautExtension:
    """
    Returns a mocked `JupyternautExtension` object that can be passed as the
    `parent` argument to objects normally initialized by `JupyternautExtension`.
    This should be passed to the "manager singletons" like `ConfigManager` and
    `EnvSecretsManager`.

    See `MockJupyternautExtension` in `conftest.py` for a complete description of the
    attributes, properties, and methods available. If something is missing,
    please feel free to add to it in your PR.
    
    Returns:
        A `MockAiExtension` instance that can be passed as the `parent` argument
        to objects normally initialized by `AiExtension`.
    """
    serverapp = jp_configurable_serverapp()
    return MockJupyternautExtension(config=jp_server_config, serverapp=serverapp)
