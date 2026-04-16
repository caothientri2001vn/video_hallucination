from types import SimpleNamespace

from src.models.openai_gpt import (
    _initial_token_limit_param,
    _is_input_too_long_error,
    _is_unsupported_token_limit_param_error,
    _extract_usage_stats,
)


def test_is_input_too_long_error_matches_alibaba_provider_message():
    exc = Exception(
        "Error code: 400 - {'error': {'message': 'Provider returned error', "
        "'metadata': {'raw': "
        '\'{"error":{"message":"<400> InternalError.Algo.InvalidParameter: '
        'Range of input length should be [1, 129024]",'
        '"code":"invalid_parameter_error"}}\'}}'
    )
    assert _is_input_too_long_error(exc) is True


def test_is_input_too_long_error_ignores_unrelated_bad_request():
    exc = Exception("400 invalid api key")
    assert _is_input_too_long_error(exc) is False


def test_extract_usage_stats_splits_openai_chat_completion_tokens():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
    )

    assert _extract_usage_stats(response) == {
        "requests": 1,
        "input_tokens": 11,
        "output_tokens": 7,
        "cached_input_tokens": 3,
        "reasoning_tokens": 2,
        "total_tokens": 18,
    }


def test_extract_usage_stats_accepts_mapping_payload_and_fallback_total():
    response = {
        "usage": {
            "input_tokens": 4,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 1},
            "output_tokens_details": {"reasoning_tokens": 2},
        }
    }

    assert _extract_usage_stats(response) == {
        "requests": 1,
        "input_tokens": 4,
        "output_tokens": 5,
        "cached_input_tokens": 1,
        "reasoning_tokens": 2,
        "total_tokens": 9,
    }


def test_direct_gpt5_uses_max_completion_tokens_limit_param():
    assert _initial_token_limit_param("gpt-5", None) == "max_completion_tokens"


def test_openai_compatible_backends_keep_max_tokens_limit_param():
    assert (
        _initial_token_limit_param("openai/gpt-5", "https://openrouter.ai/api/v1")
        == "max_tokens"
    )


def test_unsupported_max_tokens_error_is_detected():
    exc = Exception(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )

    assert _is_unsupported_token_limit_param_error(exc, "max_tokens") is True
    assert _is_unsupported_token_limit_param_error(exc, "max_completion_tokens") is False
