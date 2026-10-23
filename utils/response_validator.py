import httpx

from config import settings

def validate_response(ask_the_model, prompt):
    """Calls the model once and reports whether the answer is usable.

    `ask_the_model` is a function, not text. We hand the function in rather
    than calling Ollama directly here, so that a test can pass in a fake
    stand-in that returns a canned answer with no model running. The real one
    is models/model_client.generate_response.
    """

    if not settings.RESPONSE_VALIDATION_ENABLED:
        answer = ask_the_model(prompt)
        return {
            "valid": True,
            "failure_type": None,
            "reason": None,
            "response": answer,
        }

    try:
        answer = ask_the_model(prompt)

    except httpx.TimeoutException as error:
        if settings.VALIDATE_TIMEOUT:
            return {
                "valid": False,
                "failure_type": "timeout",
                "reason": str(error),
                "response": None,
            }
        raise

    except Exception as error:
        if settings.VALIDATE_GENERATION_ERROR:
            return {
                "valid": False,
                "failure_type": "generation_error",
                "reason": str(error),
                "response": None,
            }
        raise

    if settings.VALIDATE_EMPTY_RESPONSE and is_blank(answer):
        return {
            "valid": False,
            "failure_type": "empty_response",
            "reason": "Generator returned an empty or null response.",
            "response": answer,
        }

    return {
        "valid": True,
        "failure_type": None,
        "reason": None,
        "response": answer,
    }

def is_blank(answer):
    """True when the model effectively said nothing.

    Two separate cases, spelled out one at a time:
      - no answer object at all
      - an answer made only of spaces, tabs or newlines
    """

    if answer is None:
        return True

    if isinstance(answer, str) and answer.strip() == "":
        return True

    return False
