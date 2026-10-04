import pytest

from utils.dataset_loader import load_all_datasets
from config.generator_resolver import generate_response
from evaluators.hallucination_evaluator import HallucinationEvaluator
from utils.severity_utils import (get_severity, get_severity_score)
from utils.report_generator import results_data


dataset = load_all_datasets(
    "datasets/hallucination_dataset_with_types.xlsx"
)

evaluator = HallucinationEvaluator()


@pytest.mark.parametrize("data", dataset)
def test_hallucination(data: any):

    category = data["category"]

    prompt = data["input"]

    context = data["context"]

    actual_output = generate_response(prompt)

    result = evaluator.evaluate(
        input_text=prompt,
        actual_output_text=actual_output,
        context_text=context
    )
    print("Score:",result["score"])

    hallucination_percentage = result["score"] * 100

    severity = get_severity(hallucination_percentage)
    severity_score = get_severity_score(severity)
  
    results_data.append({
        "Category": category,
        "Question": prompt,
        "Context": context,
        "LLM Response": actual_output,
        "Hallucination %": round(hallucination_percentage, 2),
        "Reason": result["reason"],
        "Severity": severity,
        "Severity Score": severity_score
    })
   
    print("\n===================================")
    print("Category:", category)
    print("Prompt:", prompt)
    print("Context:", context)
    print("Actual Output:", actual_output)
    print("Hallucination %:", hallucination_percentage)
    print("Reason:", result["reason"])
    print("Severity:", severity)
    print("Severity Score:", severity_score)
    print("===================================\n")

    (
    f"Hallucination score {result['score']} exceeded threshold. "
    f"Reason: {result['reason']}"
    )