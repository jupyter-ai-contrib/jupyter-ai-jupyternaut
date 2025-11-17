import json
import pytest

from jupyter_ai_jupyternaut.models import parameter_schemas


async def test_get_example(jp_fetch):
    # When
    response = await jp_fetch("api/jupyternaut/get-example")

    # Then
    assert response.code == 200
    payload = json.loads(response.body)
    assert payload == {"data": "This is /api/jupyternaut/get-example endpoint!"}


@pytest.mark.parametrize(
    "model",
    [
        None,
        "openai/gpt-oss-120b",
        "hosted_vllm/doesntmatter",
        "anthropic/claude-3-5-haiku-latest",
    ],
)
async def test_get_parameters(jp_fetch, model):
    params = {}
    if model:
        params["model"] = model
    response = await jp_fetch("api/jupyternaut/model-parameters", params=params)
    assert response.code == 200
    payload = json.loads(response.body)
    expected_params = [
        "api_base",
        "max_tokens",
        "stop",
        "temperature",
        "top_p",
    ]
    if model:
        expected_params.extend(["max_completion_tokens"])
        if not model.startswith("anthropic/"):
            expected_params.extend(["frequency_penalty"])

    for param in expected_params:
        assert param in payload["parameter_names"]
        assert param in payload["parameters"]
        assert "description" in payload["parameters"][param]
