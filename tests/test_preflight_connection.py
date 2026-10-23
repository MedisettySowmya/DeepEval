from types import SimpleNamespace

import pytest

from dataset_checks import preflight
from evaluators.metric_manager import interpret_metric_result
from tests import test_evaluation_pipeline as pipeline
from dataset_checks.dataset_loader import build_test_case
from reporting.report_generator import generate_report
import pandas as pd

def row(reference="Paris"):
    return build_test_case("offline", "test", 0, "Capital?", "Paris", reference)

def test_all_three_dataset_metrics_execute_without_answer(monkeypatch):
    seen = []
    class Module:
        def is_applicable(self, case):
            return True, None

        def build(self, judge):
            def measure(case):
                assert case.actual_output is None
                seen.append(case.retrieval_context)
            return SimpleNamespace(measure=measure, success=True, score=1, reason="ok")

    monkeypatch.setattr(preflight, "ALL_METRIC_MODULES", {
        key: Module() for key in preflight.DATASET_CHECK_METRICS})
    case = row()
    report = preflight.run_preflight([case], judge_model=object())
    assert len(seen) == 3
    assert case["context_role"] == "supporting"
    assert len(report["rows"][0]["checks"]) == 3

def test_no_context_or_input_does_not_call_judge(monkeypatch):
    monkeypatch.setattr(preflight, "create_judge", lambda: pytest.fail("unexpected model call"))
    for case in (build_test_case("a", "test", 0, "Question"),
                 build_test_case("b", "test", 0, "", "Context")):
        assert all(value["status"] == "not_applicable"
                   for value in preflight.check_row(case).values())

@pytest.mark.parametrize("status", ["failed", "error"])
def test_pipeline_runs_preflight_first_and_keeps_findings_separate(monkeypatch, status):
    events = []
    results = []
    def check(cases):
        events.append("preflight")
        cases[0]["context_role"] = "adversarial"
        return {"rows": [{"checks": {"ContextualRecallMetric": {
            "status": status, "reason": "test finding"}}}]}
    def run(case):
        assert case["context_role"] == "adversarial"
        events.append("generation")
        return {"validation": {"valid": True}, "evaluation": {}}
    monkeypatch.setattr(pipeline, "run_preflight", check)
    monkeypatch.setattr(pipeline, "run_test_case", run)
    monkeypatch.setattr(pipeline, "results_data", results)
    if status == "error":
        with pytest.raises(AssertionError, match="Dataset check execution failed"):
            pipeline.test_pipeline(row())
    else:
        pipeline.test_pipeline(row())
    assert events == ["preflight", "generation"]
    assert results[0]["result"]["dataset_checks"]["checks"]
    assert results[0]["result"]["evaluation"] == {}

@pytest.mark.parametrize("score,expected", [(0.1, "passed"), (0.9, "failed")])
def test_adversarial_faithfulness_uses_contradiction_score(score, expected):
    metric = SimpleNamespace(score=score, success=score >= 0.5)
    status, _ = interpret_metric_result("FaithfulnessMetric", metric,
                                       {"context_role": "adversarial"})
    assert status == expected

def test_dataset_checks_are_reported_separately(tmp_path):
    case = row()
    check = {"id": case["id"], "input": case["input"], "context_role": "unknown",
             "checks": {"ContextualRecallMetric": {
                 "status": "error", "reason": "Judge <unavailable>", "score": None}}}
    result = {"validation": {"valid": True, "response": "Paris"},
              "evaluation": {}, "dataset_checks": check}
    path = generate_report([{"test_case": case, "result": result}], output_dir=tmp_path)
    frame = pd.read_excel(path, sheet_name="Dataset Checks")
    assert frame.iloc[0]["Metric"] == "ContextualRecallMetric"
    assert frame.iloc[0]["Status"] == "error"
    pages = list(tmp_path.glob("*.html"))
    assert pages
    assert "Judge &lt;unavailable&gt;" in pages[0].read_text(encoding="utf-8")
