"""
A fake LiteLLM provider for E2E tests.

Jupyternaut runs on LiteLLM/LangChain, so its E2E tests need a model backend
that is deterministic and requires no real model or credentials. Instead of
mocking at a subprocess boundary (as the ACP client does with fake agents),
this registers a custom LiteLLM provider — see
https://docs.litellm.ai/docs/provider_registration — that streams a canned
completion. A test-only custom model is then created against it through the
settings UI (model ID ``fake-litellm/echo``).

This module is imported by ``jupyter_server_test_config.py`` so the provider is
registered in the same process the persona runs in.
"""

from __future__ import annotations

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk

# The provider name used in the LiteLLM model ID (``<provider>/<model>``). A
# test creates a custom model with model ID ``fake-litellm/echo`` against it.
FAKE_PROVIDER = "fake-litellm"

# The canned reply the fake provider streams for any prompt. Tests assert this
# text renders in the chat, proving the message reached the model through the
# selected custom model.
CANNED_REPLY = "Hello from the fake LiteLLM provider!"


class FakeLiteLLMProvider(CustomLLM):
    """A LiteLLM custom provider that streams a fixed canned reply."""

    async def astreaming(self, model, messages, **kwargs):
        words = CANNED_REPLY.split()
        for i, word in enumerate(words):
            yield GenericStreamingChunk(
                text=word + (" " if i < len(words) - 1 else ""),
                is_finished=False,
                finish_reason="",
                index=0,
                tool_use=None,
                usage=None,
            )
        # LiteLLM streams terminate with a final, empty, finished chunk that
        # carries usage.
        yield GenericStreamingChunk(
            text="",
            is_finished=True,
            finish_reason="stop",
            index=0,
            tool_use=None,
            usage={
                "prompt_tokens": 1,
                "completion_tokens": len(words),
                "total_tokens": 1 + len(words),
            },
        )


def register() -> None:
    """
    Register the fake provider on LiteLLM's ``custom_provider_map``. Idempotent:
    calling it more than once leaves a single registration for ``FAKE_PROVIDER``.
    """
    others = [
        entry
        for entry in (litellm.custom_provider_map or [])
        if entry.get("provider") != FAKE_PROVIDER
    ]
    litellm.custom_provider_map = others + [
        {"provider": FAKE_PROVIDER, "custom_handler": FakeLiteLLMProvider()}
    ]
