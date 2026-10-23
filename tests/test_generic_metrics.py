from deepeval.test_case import LLMTestCaseParams

from config import settings
from evaluators import metric_manager, metric_registry
from evaluators.metrics.generic import GenericMetricModule


def test_only_registered_metrics_are_loaded():
    expected = set(metric_registry.REGISTRY) | set(metric_registry.DATASET_CHECK_METRICS)
    assert set(metric_manager.ALL_METRIC_MODULES) == expected


def test_generic_metrics_build_with_shared_settings(monkeypatch):
    for key, module in metric_manager._GENERIC_MODULES.items():
        captured = {}

        def constructor(model, threshold):
            captured.update(model=model, threshold=threshold)
            return captured

        monkeypatch.setattr(module, "metric_class", constructor)
        assert module.build("test-judge") == {
            "model": "test-judge",
            "threshold": settings.METRIC_THRESHOLDS.get(key, settings.DEFAULT_METRIC_THRESHOLD),
        }


def test_generic_metric_requires_its_declared_fields():
    class ReferenceMetric:
        _required_params = [LLMTestCaseParams.EXPECTED_OUTPUT]

    module = GenericMetricModule("ReferenceMetric", ReferenceMetric)
    assert module.is_applicable({}) == (
        False, "missing required test-case data: expected_output"
    )
    assert module.is_applicable({"expected_answer": "Canberra"}) == (True, None)
