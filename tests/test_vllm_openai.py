from src.models import load_model
from src.models.vllm_openai import VLLMOpenAIModel


def test_load_model_routes_vllm_prefix_to_vllm_backend():
    model = load_model("vllm/Qwen/Qwen3-VL-32B-Thinking")

    assert isinstance(model, VLLMOpenAIModel)
    assert model.model_id == "Qwen/Qwen3-VL-32B-Thinking"


def test_vllm_build_content_omits_image_detail():
    model = VLLMOpenAIModel(model_id="Qwen/Qwen3-VL-32B-Thinking")

    content = model._build_content("What is happening?", ["abc123"])

    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert "detail" not in content[0]["image_url"]


def test_vllm_client_defaults_to_localhost_and_empty_key(monkeypatch):
    captured = {}

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setattr("src.models.vllm_openai.OpenAI", DummyOpenAI)

    model = VLLMOpenAIModel(model_id="Qwen/Qwen3-VL-32B-Thinking")
    model._get_client()

    assert captured == {
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1",
    }
