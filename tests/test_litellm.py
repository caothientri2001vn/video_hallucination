import numpy as np

from src.models import load_model
from src.models.litellm import LiteLLMModel
from src.models.openai_gpt import _resize_longest_side_bgr


def test_load_model_routes_litellm_prefix_to_litellm_backend():
    model = load_model("litellm/gemini-3-flash")

    assert isinstance(model, LiteLLMModel)
    assert model.model_id == "gemini-3-flash"
    assert model.n_frames == 128
    assert model.max_frame_longest_side == 480
    assert model.cache_namespace == "litellm_gemini-3-flash__vanilla"


def test_litellm_client_loads_dotenv_and_reads_proxy_env(monkeypatch):
    captured = {}
    load_dotenv_calls = []

    class DummyOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")
    monkeypatch.setenv("LITELLM_PROXY_URL", "https://litellm.example/v1")
    monkeypatch.setattr("src.models.litellm.load_dotenv", lambda: load_dotenv_calls.append(True))
    monkeypatch.setattr("src.models.litellm.OpenAI", DummyOpenAI)

    model = LiteLLMModel(model_id="gemini-3-flash")
    client = model._get_client()

    assert isinstance(client, DummyOpenAI)
    assert load_dotenv_calls == [True]
    assert captured == {
        "api_key": "proxy-key",
        "base_url": "https://litellm.example/v1",
    }


def test_litellm_frame_resize_uses_longest_side_for_horizontal_and_vertical():
    horizontal = np.zeros((720, 1280, 3), dtype=np.uint8)
    vertical = np.zeros((1280, 720, 3), dtype=np.uint8)

    assert _resize_longest_side_bgr(horizontal, 480).shape[:2] == (270, 480)
    assert _resize_longest_side_bgr(vertical, 480).shape[:2] == (480, 270)
