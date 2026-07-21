import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def generate_custom_model_id(model_id: str) -> str:
    """
    Generate a unique ID for a custom model in the format
    ``custom-{uuid}/{model-id}``, e.g. ``custom-<uuid>/openai/hermes``.

    The ``custom-`` prefix lets consumers distinguish a user-defined custom
    model from a plain LiteLLM model ID, and the trailing model ID keeps the
    generated ID human-readable in logs and metadata.
    """
    return f"custom-{uuid.uuid4()}/{model_id}"


class CustomModel(BaseModel):
    """
    A user-defined custom model, configured in the Jupyternaut settings view
    and surfaced at the top of the model picker.

    Custom models are stored in the Jupyternaut config (see
    ``JaiConfig.custom_models``) keyed by their generated ``id``. The model
    picker only carries this ``id`` in message metadata; the persona looks up
    the full model ID and parameters from config when processing a message, so
    the parameters never travel over the wire.
    """

    id: str = Field(default_factory=lambda: generate_custom_model_id("model"))
    """
    Unique ID of this custom model, in the format ``custom-{uuid}/{model-id}``.
    Generated on creation and stable for the model's lifetime.
    """

    name: str
    """Human-readable name shown in the model picker UI."""

    description: Optional[str] = None
    """Optional description shown alongside the name in the UI."""

    model_id: str
    """
    The LiteLLM model ID passed to the agent, e.g. ``openai/hermes``. See
    https://docs.litellm.ai/docs/providers.
    """

    params: dict[str, Any] = {}
    """
    Model parameters passed to the agent (e.g. ``temperature``, ``api_base``),
    unpacked as keyword arguments when constructing the chat model.
    """


class JaiConfig(BaseModel):
    """
    Pydantic model that serializes and validates the Jupyter AI config.
    """

    model_provider_id: Optional[str] = None
    """
    Model ID of the chat model.

    NOTE (v3.1): The chat model is now chosen per-user, per-chat via the model
    picker instead of this global setting, so it is no longer surfaced in the
    settings UI. It is retained in the schema for back-compat (old config files
    must still load) and as the source for the built-in "default" model option:
    when set (e.g. via the ``initial_language_model`` traitlet), the default
    model option resolves to this value.
    """

    custom_models: list[CustomModel] = []
    """
    User-defined custom models, in display order. These appear at the top of
    the model picker, above the single built-in default option. Reordering in
    the settings UI reorders this list.
    """

    embeddings_provider_id: Optional[str] = None
    """
    Model ID of the embedding model.
    """

    completions_model_provider_id: Optional[str] = None
    """
    Model ID of the completions model.
    """

    api_keys: dict[str, str] = {}
    """
    Dictionary of API keys. The name of each key should correspond to the
    environment variable expected by the underlying client library, e.g.
    "OPENAI_API_KEY".
    """

    send_with_shift_enter: bool = False
    """
    Whether the "Enter" key should create a new line instead of sending the
    message.
    """

    fields: dict[str, dict[str, Any]] = {}
    """
    Dictionary that defines custom fields for each chat model.
    Key: chat model ID.
    Value: Dictionary of keyword arguments.
    """

    embeddings_fields: dict[str, dict[str, Any]] = {}
    completions_fields: dict[str, dict[str, Any]] = {}


class DescribeConfigResponse(BaseModel):
    model_provider_id: Optional[str] = None
    embeddings_provider_id: Optional[str] = None
    send_with_shift_enter: bool
    fields: dict[str, dict[str, Any]]

    api_keys: list[str]
    """
    List of the names of the API keys. This deliberately does not include the
    value of each API key in the interest of security.
    """

    last_read: int
    """
    Timestamp indicating when the configuration file was last read. Should be
    passed to the subsequent UpdateConfig request if an update is made.
    """

    completions_model_provider_id: Optional[str] = None
    completions_fields: dict[str, dict[str, Any]]
    embeddings_fields: dict[str, dict[str, Any]]

    custom_models: list[CustomModel] = []
    """User-defined custom models, in display order."""


class UpdateConfigRequest(BaseModel):
    model_provider_id: Optional[str] = None
    embeddings_provider_id: Optional[str] = None
    completions_model_provider_id: Optional[str] = None
    send_with_shift_enter: Optional[bool] = None
    api_keys: Optional[dict[str, str]] = None
    # if passed, this will raise an Error if the config was written to after the
    # time specified by `last_read` to prevent write-write conflicts.
    last_read: Optional[int] = None
    fields: Optional[dict[str, dict[str, Any]]] = None
    completions_fields: Optional[dict[str, dict[str, Any]]] = None
    embeddings_fields: Optional[dict[str, dict[str, Any]]] = None
    custom_models: Optional[list[CustomModel]] = None
    """
    If passed, replaces the full list of custom models (in display order). The
    settings UI sends the entire list on every save, so reordering, edits,
    additions, and deletions all flow through this single field.
    """

    @field_validator("send_with_shift_enter", "api_keys", "fields", mode="before")
    @classmethod
    def ensure_not_none_if_passed(cls, field_val: Any) -> Any:
        """
        Field validator ensuring that certain fields are never `None` if set.
        """
        assert field_val is not None, "size may not be None"
        return field_val
