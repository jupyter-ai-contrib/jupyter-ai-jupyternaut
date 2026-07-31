"""Chat model wrapper that adds Jupyternaut-specific behavior to
``langchain_litellm.ChatLiteLLM``.

Jupyternaut used to vendor its own copy of ``ChatLiteLLM``. It now delegates to
the upstream ``langchain-litellm`` package
(https://github.com/langchain-ai/langchain-litellm) and only overrides the
async completion path so that ephemeral prompt caching is enabled by default.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_litellm import ChatLiteLLM as _UpstreamChatLiteLLM


class ChatLiteLLM(_UpstreamChatLiteLLM):
    """``langchain_litellm.ChatLiteLLM`` with Jupyternaut's default prompt caching.

    Enables ephemeral prompt caching of the last system message by default
    when calling ``litellm.acompletion()``, except on Amazon Bedrock's
    ``InvokeModel`` API (i.e. ``bedrock/…`` models that are not routed through
    ``bedrock/converse/…``) where an upstream LiteLLM bug currently breaks it.

    See:
      - https://docs.litellm.ai/docs/tutorials/prompt_caching
      - https://github.com/BerriAI/litellm/issues/17479
    """

    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        if not (
            self.model.startswith("bedrock/")
            and not self.model.startswith("bedrock/converse/")
        ):
            kwargs.setdefault(
                "cache_control_injection_points",
                [{"location": "message", "role": "system"}],
            )
        return await super().acompletion_with_retry(run_manager=run_manager, **kwargs)
