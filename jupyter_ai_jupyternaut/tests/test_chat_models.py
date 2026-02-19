import pytest
from litellm import Usage
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
)

from jupyter_ai_jupyternaut.jupyternaut.chat_models import _create_usage_metadata


class TestCreateUsageMetadata:
    """Tests for _create_usage_metadata covering null-check edge cases."""

    def test_none_details(self):
        """Models like gpt-5.2-codex return None for token detail objects."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        result = _create_usage_metadata(usage)

        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30
        assert result["input_token_details"]["audio"] == 0
        assert result["input_token_details"]["cache_creation"] == 0
        assert result["input_token_details"]["cache_read"] == 0
        assert result["output_token_details"]["audio"] == 0
        assert result["output_token_details"]["reasoning"] == 0

    def test_full_details(self):
        """Models that provide complete token detail breakdowns."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                audio_tokens=5,
                cached_tokens=10,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                audio_tokens=3,
                reasoning_tokens=12,
            ),
        )
        result = _create_usage_metadata(usage)

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["input_token_details"]["audio"] == 5
        assert result["input_token_details"]["cache_read"] == 10
        assert result["output_token_details"]["audio"] == 3
        assert result["output_token_details"]["reasoning"] == 12

    def test_details_with_none_subfields(self):
        """Detail objects exist but individual fields are None."""
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                audio_tokens=None,
                cached_tokens=None,
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                audio_tokens=None,
                reasoning_tokens=None,
            ),
        )
        result = _create_usage_metadata(usage)

        assert result["input_token_details"]["audio"] == 0
        assert result["input_token_details"]["cache_read"] == 0
        assert result["output_token_details"]["audio"] == 0
        assert result["output_token_details"]["reasoning"] == 0

    def test_none_prompt_and_completion_tokens(self):
        """Both top-level token counts are None."""
        usage = Usage(
            prompt_tokens=None,
            completion_tokens=None,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        result = _create_usage_metadata(usage)

        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0
