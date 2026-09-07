import json
import pytest

from utils.dataset_loader_copy import load_qa_boat_datasets
from config.qa_boat_config import generate_response_qa_boat
from evaluators.hallucination_evaluator import HallucinationEvaluator
from utils.severity_utils import get_severity, get_severity_score
from utils.report_generator import results_data


dataset = load_qa_boat_datasets(
    "datasets/Test_dataset.xlsx"
)

evaluator = HallucinationEvaluator()


@pytest.mark.parametrize("data", dataset)
def test_hallucination(data: dict):

    category = data["category"]
    requirement = data["requirement"]
    acceptance_criteria = data["acceptance_criteria"]

    prompt = f"""
Generate test cases for the following requirement.

Requirement:
{requirement}

Acceptance Criteria:
{acceptance_criteria}

Return only valid JSON with testCases.
"""

    actual_output = generate_response_qa_boat(prompt)

    generated_data = json.loads(actual_output)
    

    generated_test_cases = generated_data["testCases"]

    evaluation_context = [
        f"Requirement:\n{requirement}\n\n"
        f"Acceptance Criteria:\n{acceptance_criteria}"
    ]

    for test_case in generated_test_cases:

        test_case_output = json.dumps(test_case, indent=2)

        result = evaluator.evaluate(
            input_text=requirement,
            actual_output_text=test_case_output,
            context_text=evaluation_context
        )

        hallucination_percentage = result["score"] * 100

        severity = get_severity(hallucination_percentage)

        severity_score = get_severity_score(severity)

        results_data.append({
            "Category": category,
            "Test Case ID": test_case.get(
                "testCaseId",
                test_case.get("id", "Not available")
            ),
            "Test Case Title": test_case.get(
                "title",
                test_case.get("description", "Not available")
            ),
            "Requirement": requirement,
            "Acceptance Criteria": acceptance_criteria,
            "Generated Test Case": test_case_output,
            "Hallucination %": round(hallucination_percentage, 2),
            "Reason": result["reason"],
            "Severity": severity,
            "Severity Score": severity_score
        })

    assert True