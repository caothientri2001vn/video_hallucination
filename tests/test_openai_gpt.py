from src.models.openai_gpt import _is_input_too_long_error


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
