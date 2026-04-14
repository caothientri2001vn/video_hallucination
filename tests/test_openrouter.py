from src.models import load_model
from src.models.openai_gpt import OpenAIModel
from src.models.openrouter_gemini import OpenRouterGeminiModel


def test_openrouter_non_gemini_models_cap_frame_longest_side():
    model = load_model("openrouter/qwen/qwen3-vl-32b-instruct")

    assert isinstance(model, OpenAIModel)
    assert model.model_id == "qwen/qwen3-vl-32b-instruct"
    assert model.max_frame_longest_side == 480


def test_openrouter_gemini_models_cap_frame_longest_side():
    model = load_model("openrouter/google/gemini-3.1-flash-lite-preview")

    assert isinstance(model, OpenRouterGeminiModel)
    assert model.model_id == "google/gemini-3.1-flash-lite-preview"
    assert model.max_frame_longest_side == 480
