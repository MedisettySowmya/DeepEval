import httpx

from config import model_config


def validate_response(generate_fn, prompt):

    if not model_config.RESPONSE_VALIDATION_ENABLED:
        response = generate_fn(prompt)
        return _result(True, None, None, response)

    try:
        response = generate_fn(prompt)

    except httpx.TimeoutException as error:
        if model_config.VALIDATE_TIMEOUT:
            return _result(False, "timeout", str(error), None)
        raise

    except Exception as error:
        if model_config.VALIDATE_GENERATION_ERROR:
            return _result(False, "generation_error", str(error), None)
        raise

    if model_config.VALIDATE_EMPTY_RESPONSE and _is_empty(response):
        return _result(
            False,
            "empty_response",
            "Generator returned an empty or null response.",
            response
        )

    return _result(True, None, None, response)


def _is_empty(response):
    return response is None or (
        isinstance(response, str) and response.strip() == ""
    )


def _result(valid, failure_type, reason, response):
    return {
        "valid": valid,
        "failure_type": failure_type,
        "reason": reason,
        "response": response
    }
