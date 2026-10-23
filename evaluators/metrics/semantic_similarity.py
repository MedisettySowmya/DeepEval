from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCaseParams

from config import settings

KEY = "SemanticSimilarityMetric"
TIER = "statistical"
REQUIRED_PARAMS = [
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def is_applicable(test_case):
    if not test_case.get("expected_answer"):
        return False, "missing required test-case data: expected_output"
    return True, None

class SemanticSimilarityMetric(BaseMetric):
    """Cosine similarity between the actual answer and the Ground_Truth.
    No LLM call, no judge model -- unlike every other metric module in this
    project, build() below does not pass a model in at all."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        from sentence_transformers import util

        model = _get_model()
        embeddings = model.encode(
            [test_case.actual_output, test_case.expected_output]
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        self.score = round(similarity, 4)
        self.success = self.score >= self.threshold
        self.reason = (
            f"Cosine similarity between the answer and the Ground_Truth is "
            f"{self.score} (threshold {self.threshold}). No LLM judge was "
            f"used for this score -- only sentence embeddings."
        )
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Semantic Similarity"

def build(judge_model):
    threshold = settings.METRIC_THRESHOLDS.get(
        KEY, settings.DEFAULT_METRIC_THRESHOLD
    )
    return SemanticSimilarityMetric(threshold=threshold)
