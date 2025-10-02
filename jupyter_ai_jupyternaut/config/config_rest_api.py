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

    @web.authenticated
    def post(self):
        try:
            config = UpdateConfigRequest(**self.get_json_body())
            self.config_manager.update_config(config)
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