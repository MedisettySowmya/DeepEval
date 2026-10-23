VERIFIED = "VERIFIED"
EVALUATED = "EVALUATED"
UNVERIFIABLE = "UNVERIFIABLE"

def classify(test_case):

    has_ground_truth = (
        bool(test_case.get("expected_answer"))
        and not test_case.get("expects_abstention")
    )
    has_context = bool(test_case.get("context"))

    if has_ground_truth:
        return VERIFIED

    if has_context:
        return EVALUATED

    return UNVERIFIABLE

_HEDGING_PHRASES = [
    "i don't know", "i do not know",
    "i'm not sure", "i am not sure",
    "cannot confirm", "can't confirm",
    "cannot verify", "can't verify", "not able to verify",
    "no verified information", "no reliable information",
    "i do not have information", "i don't have information",
    "unclear", "uncertain", "not certain",
    "cannot provide a definitive", "can't provide a definitive",
    "there is no established",
]

def _response_appears_hedged(response_text):
    if not response_text:
        return False
    lowered = response_text.lower()
    return any(phrase in lowered for phrase in _HEDGING_PHRASES)

def _get_unverifiable_behavior_signal(actual_output):

    if not actual_output:
        return {
            "signal": "not_determined",
            "reason": "No response text to inspect.",
        }

    if _response_appears_hedged(actual_output):
        return {
            "signal": "appropriately_hedged",
            "reason": (
                "Response contains language indicating uncertainty or "
                "inability to verify the claim."
            )
        }

    return {
        "signal": "unsupported_claim_likely",
        "reason": (
            "No hedging language detected; the response asserts information "
            "confidently despite no context or expected answer being "
            "available to verify it. No metric corroborates this -- there is "
            "no DeepEval metric that checks factual support without a "
            "reference -- so this is a heuristic, not a measurement."
        )
    }

def classify_test_case(test_case, actual_output=None, evaluation_result=None):

    classification = classify(test_case)

    result = {"classification": classification, "behavior_signal": None}

    if classification == UNVERIFIABLE:
        result["behavior_signal"] = _get_unverifiable_behavior_signal(actual_output)

    return result
