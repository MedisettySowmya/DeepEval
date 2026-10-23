from models.model_client import generate_response
from utils.response_validator import validate_response
from prompts.prompt_builder import build_prompt
from evaluators.metric_manager import evaluate_test_case_detailed
from evaluators.result_classifier import classify_test_case
from evaluators.root_cause_analyzer import analyze_test_case
from evaluators.tier_classifier import annotate_evaluation_result
from human_review.review_gate import get_review_reasons

def _invalid_input_result(test_case, classify_fn, analyze_fn, review_fn):
    """Stage 0's first gate, before the model call.

    Input is the one mandatory column. A blank cell previously reached
    build_prompt and raised inside _hash_text -- pandas gives float('nan') for
    a blank, NaN is truthy, so the `or ""` guard was skipped and .encode() was
    called on a float. Failing fast here costs no model call and reports a
    reason instead of an uncaught exception.
    """

    validation_result = {
        "valid": False,
        "failure_type": "invalid_input",
        "reason": test_case.get("input_invalid_reason")
                  or "Input cell is empty; nothing to ask the model.",
        "response": None,
    }

    classification_result = classify_fn(test_case)

    return {
        "id": test_case.get("id"),
        "category": test_case.get("category"),
        "input": test_case.get("input"),
        "profile": test_case.get("profile"),
        "prompt": None,
        "validation": validation_result,
        "evaluation": None,
        "stages": None,
        "classification": classification_result,
        "root_cause_analysis": analyze_fn(
            test_case, validation_result, None, classification_result
        ),
        "human_review": review_fn(test_case, None),
    }

def run_test_case(
    test_case,
    generate_fn=None,
    evaluate_fn=None,
    classify_fn=None,
    analyze_fn=None,
    tier_fn=None,
    review_fn=None,
    instruction_variant=None,
    send_context=None,
):
    """Run one row: validate input, generate, validate answer, score, report.

    Optional *_fn arguments let offline tests supply fake implementations.
    """

    # 1. Use the normal implementations unless a caller supplies replacements.
    generate_fn = generate_fn or generate_response
    evaluate_fn = evaluate_fn or evaluate_test_case_detailed
    classify_fn = classify_fn or classify_test_case
    analyze_fn = analyze_fn or analyze_test_case
    tier_fn = tier_fn or annotate_evaluation_result
    review_fn = review_fn or get_review_reasons

    # 2. Stop before calling the model if the spreadsheet input is invalid.
    input_is_valid = test_case.get("input_valid", bool(test_case.get("input")))
    if not input_is_valid:
        return _invalid_input_result(test_case, classify_fn, analyze_fn, review_fn)

    # 3. Build the prompt, generate an answer and check it is usable.
    prompt = build_prompt(
        test_case,
        instruction_variant=instruction_variant,
        send_context=send_context,
    )

    validation_result = validate_response(generate_fn, prompt["text"])

    if not validation_result["valid"]:
        classification_result = classify_fn(test_case)
        return {
            "id": test_case.get("id"),
            "category": test_case.get("category"),
            "input": test_case.get("input"),
            "profile": test_case.get("profile"),
            "prompt": prompt,
            "validation": validation_result,
            "evaluation": None,
            "stages": None,
            "classification": classification_result,
            "root_cause_analysis": analyze_fn(
                test_case, validation_result, None, classification_result
            ),
            "human_review": review_fn(test_case, None),
        }

    # 4. Score the answer. Some callers return metrics without stage details.
    evaluated = evaluate_fn(
        test_case=test_case,
        prompt=prompt,
        actual_output=validation_result["response"],
    )

    if isinstance(evaluated, dict) and "metrics" in evaluated:
        evaluation_result = evaluated["metrics"]
        stage_outcomes = evaluated.get("stages")
    else:
        evaluation_result = evaluated
        stage_outcomes = None

    # 5. Add trust levels, classification, failure explanations and review flags.
    evaluation_result = tier_fn(
        evaluation_result, category=test_case.get("category")
    )

    classification_result = classify_fn(
        test_case,
        actual_output=validation_result["response"],
        evaluation_result=evaluation_result
    )

    root_cause_result = analyze_fn(
        test_case, validation_result, evaluation_result, classification_result
    )

    return {
        "id": test_case.get("id"),
        "category": test_case.get("category"),
        "input": test_case.get("input"),
        "profile": test_case.get("profile"),
        "prompt": prompt,
        "validation": validation_result,
        "evaluation": evaluation_result,
        "stages": stage_outcomes,
        "classification": classification_result,
        "root_cause_analysis": root_cause_result,
        "human_review": review_fn(test_case, evaluation_result)
    }
