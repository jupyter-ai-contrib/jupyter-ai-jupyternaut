from __future__ import annotations
import json

from jupyter_server.base.handlers import APIHandler as BaseAPIHandler
from tornado.web import authenticated, HTTPError

from ..jupyternaut.jupyternaut import _litellm_chat_models


class LiteLLMModelsRestAPI(BaseAPIHandler):
    """
    Tornado handler that serves the list of chat model IDs known to LiteLLM on
    the `/api/jupyternaut/litellm-models` endpoint. Used by the custom-model
    editor in the settings view to populate a searchable model dropdown.
    """

    @authenticated
    def get(self):
        try:
            models = _litellm_chat_models()
            self.set_header("Content-Type", "application/json")
            self.finish(json.dumps({"models": models}))
        except Exception as e:
            self.log.exception("Failed to list LiteLLM chat models.")
            raise HTTPError(500, f"Internal server error: {str(e)}")
