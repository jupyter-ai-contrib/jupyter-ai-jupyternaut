"""
Tests for the custom-models config schema/manager and the Jupyternaut persona's
model resolution against the persona-manager awareness/model API (issue #61).
"""

import json
import logging

import pytest

from jupyter_ai_jupyternaut.config.config_manager import ConfigManager
from jupyter_ai_jupyternaut.config.config_models import (
    CustomModel,
    JaiConfig,
    UpdateConfigRequest,
    generate_custom_model_id,
)
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import (
    DEFAULT_MODEL_ID,
    JupyternautPersona,
    _extract_message_text,
    _litellm_chat_models,
)


@pytest.fixture
def config_path(tmp_path):
    return str(tmp_path / "config.json")


def _write_config(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


def _manager(path: str) -> ConfigManager:
    return ConfigManager(log=logging.getLogger("test"), defaults={}, config_path=path)


# ---------------------------------------------------------------------------
# Config schema & manager
# ---------------------------------------------------------------------------


def test_custom_model_id_format():
    cid = generate_custom_model_id("openai/hermes")
    assert cid.startswith("custom-")
    assert cid.endswith("/openai/hermes")


def test_old_config_loads_without_custom_models():
    """An existing config file with no `custom_models` key must still load."""
    old = {
        "model_provider_id": "anthropic/claude-3-5-haiku-latest",
        "completions_model_provider_id": "openai/gpt-4",
        "embeddings_provider_id": "openai/text-embedding-3-small",
        "api_keys": {"OPENAI_API_KEY": "sk-x"},
        "fields": {"anthropic/claude-3-5-haiku-latest": {"temperature": 0.5}},
        "send_with_shift_enter": True,
    }
    config = JaiConfig(**old)
    assert config.custom_models == []
    assert config.model_provider_id == "anthropic/claude-3-5-haiku-latest"


def test_update_and_lookup_custom_models(config_path):
    _write_config(config_path, {"model_provider_id": "anthropic/claude-3-5-haiku-latest"})
    cm = _manager(config_path)

    m1 = CustomModel(name="A", model_id="openai/a", params={"temperature": 0.7})
    m2 = CustomModel(name="B", model_id="openai/b")
    cm.update_config(UpdateConfigRequest(custom_models=[m1, m2]))

    assert [m.name for m in cm.custom_models] == ["A", "B"]
    assert cm.get_custom_model(m1.id).params == {"temperature": 0.7}
    assert cm.get_custom_model("does-not-exist") is None
    # Old, still-supported field is preserved.
    assert cm.chat_model == "anthropic/claude-3-5-haiku-latest"


def test_custom_model_ids_are_stable_across_unrelated_saves(config_path):
    """
    Saving an unrelated field must not regenerate custom-model IDs (regression:
    `exclude_unset` drops the default-factory `id`, which would break lookups).
    """
    _write_config(config_path, {})
    cm = _manager(config_path)
    m1 = CustomModel(name="A", model_id="openai/a")
    cm.update_config(UpdateConfigRequest(custom_models=[m1]))
    original_ids = [m.id for m in cm.custom_models]

    cm.update_config(UpdateConfigRequest(api_keys={"OPENAI_API_KEY": "sk-y"}))
    assert [m.id for m in cm.custom_models] == original_ids


def test_custom_models_replaced_not_appended(config_path):
    """Reorder/delete works: the full list is replaced, not merged/appended."""
    _write_config(config_path, {})
    cm = _manager(config_path)
    m1 = CustomModel(name="A", model_id="openai/a")
    m2 = CustomModel(name="B", model_id="openai/b")
    cm.update_config(UpdateConfigRequest(custom_models=[m1, m2]))

    # Reorder + delete m1 by sending the full desired list.
    cm.update_config(UpdateConfigRequest(custom_models=[m2]))
    assert [m.name for m in cm.custom_models] == ["B"]


def test_get_config_includes_custom_models(config_path):
    _write_config(config_path, {})
    cm = _manager(config_path)
    m1 = CustomModel(name="A", model_id="openai/a")
    cm.update_config(UpdateConfigRequest(custom_models=[m1]))
    assert [m.name for m in cm.get_config().custom_models] == ["A"]


# ---------------------------------------------------------------------------
# Persona model configuration & resolution
# ---------------------------------------------------------------------------


def _persona(cm: ConfigManager, selected):
    """A JupyternautPersona wired to `cm` with `get_model()` returning `selected`."""
    persona = JupyternautPersona.__new__(JupyternautPersona)
    persona.config_manager = cm
    persona.get_model = lambda: selected
    return persona


def test_model_configuration_lists_custom_then_catalog(config_path):
    _write_config(config_path, {})
    cm = _manager(config_path)
    m1 = CustomModel(name="Hermes", model_id="openai/hermes")
    cm.update_config(UpdateConfigRequest(custom_models=[m1]))

    mc = _persona(cm, None)._build_model_configuration()
    ids = [o.id for o in mc.options]
    # Custom model first, then the LiteLLM catalog. There is no explicit
    # "default" option — the picker renders its own built-in "Default" row.
    assert ids[0] == m1.id
    assert DEFAULT_MODEL_ID not in ids
    assert mc.current is None
    assert mc.settings == []


def test_resolve_custom_model(config_path):
    _write_config(config_path, {})
    cm = _manager(config_path)
    m1 = CustomModel(
        name="Hermes",
        model_id="openai/hermes",
        params={"api_base": "http://localhost:8080", "temperature": 0.9},
    )
    cm.update_config(UpdateConfigRequest(custom_models=[m1]))

    model_id, params = _persona(cm, m1.id)._resolve_model()
    assert model_id == "openai/hermes"
    assert params == {"api_base": "http://localhost:8080", "temperature": 0.9}


@pytest.mark.parametrize("selected", [None, DEFAULT_MODEL_ID, "custom-stale/openai/gone"])
def test_resolve_falls_back_to_configured_default(config_path, selected):
    _write_config(
        config_path,
        {
            "model_provider_id": "anthropic/claude-3-5-haiku-latest",
            "fields": {"anthropic/claude-3-5-haiku-latest": {"temperature": 0.3}},
        },
    )
    cm = _manager(config_path)
    model_id, params = _persona(cm, selected)._resolve_model()
    assert model_id == "anthropic/claude-3-5-haiku-latest"
    assert params == {"temperature": 0.3}


def test_resolve_returns_none_when_nothing_configured(config_path):
    _write_config(config_path, {})
    cm = _manager(config_path)
    assert _persona(cm, None)._resolve_model() == (None, {})


def test_resolve_direct_litellm_model(config_path):
    """A LiteLLM model ID picked directly from the list is used as-is."""
    _write_config(config_path, {})
    cm = _manager(config_path)
    model_id, params = _persona(cm, "openai/gpt-4o")._resolve_model()
    assert model_id == "openai/gpt-4o"
    assert params == {}


def test_resolve_direct_litellm_model_with_saved_params(config_path):
    """Params saved for a directly-picked model (config `fields`) are applied."""
    _write_config(
        config_path,
        {"fields": {"openai/gpt-4o": {"temperature": 0.2}}},
    )
    cm = _manager(config_path)
    model_id, params = _persona(cm, "openai/gpt-4o")._resolve_model()
    assert model_id == "openai/gpt-4o"
    assert params == {"temperature": 0.2}


# ---------------------------------------------------------------------------
# LiteLLM model catalog in the picker
# ---------------------------------------------------------------------------


def test_litellm_chat_models_is_large_and_excludes_placeholder():
    models = _litellm_chat_models()
    # LiteLLM knows well over a thousand chat models.
    assert len(models) > 1000
    assert "sample_spec" not in models
    assert models == sorted(models)


def test_model_configuration_includes_full_litellm_catalog(config_path):
    _write_config(config_path, {})
    cm = _manager(config_path)
    m1 = CustomModel(name="Hermes", model_id="openai/hermes")
    cm.update_config(UpdateConfigRequest(custom_models=[m1]))

    mc = _persona(cm, None)._build_model_configuration()
    ids = [o.id for o in mc.options]
    # Custom model first, then the LiteLLM catalog (no explicit default option).
    assert ids[0] == m1.id
    assert ids[1] != DEFAULT_MODEL_ID
    assert len(ids) > 1000
    # A representative LiteLLM model is present as a directly-selectable option.
    llm = _litellm_chat_models()
    assert llm[0] in ids


# ---------------------------------------------------------------------------
# Streaming-event text extraction
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def test_extract_message_text_from_string():
    assert _extract_message_text(_FakeMessage("hello world")) == "hello world"


def test_extract_message_text_from_content_blocks():
    blocks = [
        {"type": "text", "text": "hello "},
        {"type": "tool_call", "id": "1"},
        {"type": "text", "text": "world"},
    ]
    assert _extract_message_text(_FakeMessage(blocks)) == "hello world"


def test_extract_message_text_handles_missing_content():
    assert _extract_message_text(object()) == ""
    assert _extract_message_text(_FakeMessage(None)) == ""
