from __future__ import annotations
from jupyter_server.base.handlers import APIHandler as BaseAPIHandler
from pydantic import ValidationError
from tornado import web
from tornado.web import HTTPError
from typing import TYPE_CHECKING

from .config_manager import KeyEmptyError, WriteConflictError
from .config_models import UpdateConfigRequest

if TYPE_CHECKING:
    from .config_manager import ConfigManager


class ConfigRestAPI(BaseAPIHandler):
    """
    Tornado handler that defines the Config REST API served on
    the `/api/jupyternaut/config` endpoint.
    """

    @property
    def config_manager(self) -> ConfigManager:
        return self.settings["jupyternaut.config_manager"]

    @web.authenticated
    def get(self):
        config = self.config_manager.get_config()
        if not config:
            raise HTTPError(500, "No config found.")

        self.finish(config.model_dump_json())

    def _refresh_persona_model_configurations(self) -> None:
        """
        Re-publish every live Jupyternaut persona's model configuration over
        awareness after a config change, so the model picker reflects newly
        added, edited, or reordered custom models without a page reload.

        Personas are looked up from the persona-manager registry in the shared
        web-app settings. The Jupyternaut persona is identified by duck-typing
        (its `sync_model_configuration` method) rather than an isinstance check,
        to avoid importing the persona class here (which would create an import
        cycle: the persona imports this config package).
        """
        persona_managers = (
            self.settings.get("jupyter-ai", {}).get("persona-managers") or {}
        )
        for manager in persona_managers.values():
            for persona in manager.personas.values():
                sync = getattr(persona, "sync_model_configuration", None)
                if callable(sync):
                    try:
                        sync()
                    except Exception:
                        self.log.exception(
                            "Failed to refresh model configuration for persona "
                            f"'{getattr(persona, 'id', '?')}'."
                        )

    @web.authenticated
    def post(self):
        try:
            config = UpdateConfigRequest(**self.get_json_body())
            self.config_manager.update_config(config)
            self._refresh_persona_model_configurations()
            self.set_status(204)
            self.finish()
        except (ValidationError, WriteConflictError, KeyEmptyError) as e:
            self.log.exception(e)
            raise HTTPError(500, str(e)) from e
        except ValueError as e:
            self.log.exception(e)
            raise HTTPError(500, str(e.cause) if hasattr(e, "cause") else str(e))
        except Exception as e:
            self.log.exception(e)
            raise HTTPError(
                500, "Unexpected error occurred while updating the config."
            ) from e