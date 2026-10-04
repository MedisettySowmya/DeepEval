from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCaseParams
from sentence_transformers import SentenceTransformer, util

from config import model_config


# ---------------------------------------------------------------------------
# Semantic Similarity -- a correctness check with NO AI judge involved.
#
# Why this exists: ExactMatchMetric only works for a short answer like a
# single number or name ("Mars"). It fails on a full-sentence answer even
# when the answer is correct, because "Mars is the Red Planet" is not
# character-for-character identical to "The Red Planet is Mars." This
# metric compares MEANING instead of exact characters, using sentence
# embeddings -- so a correct answer written in different words still
# scores well.
#
# TIER = "statistical": this fills the one tier that
# metric_tiering/tier_classifier.py reserved but never had a metric for.
# Same bedrock-trust idea as ExactMatchMetric (no LLM, so nothing to
# calibrate against a human), but it understands paraphrasing, which
# ExactMatchMetric cannot.
# ---------------------------------------------------------------------------

KEY = "SemanticSimilarityMetric"
TIER = "statistical"
REQUIRED_PARAMS = [
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
]

# A small, well-known, free model that turns a sentence into a list of
# numbers ("embedding") such that sentences with similar meaning end up
# with similar numbers. Loaded once per process and reused for every row --
# loading it fresh per row would be slow. The first run on a machine
# downloads it (~90MB) to a local cache; every run after that is instant.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
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
        # No async path needed -- measure() does no network/judge call, so
        # there is nothing to gain from running it asynchronously. This
        # just lets DeepEval's own call sites that await a_measure() work.
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return bool(self.success)

    @property
    def __name__(self):
        return "Semantic Similarity"


def build(judge_model):
    # judge_model is accepted and ignored -- every metric module's build()
    # is called the same way by metric_manager, but this metric needs no
    # judge at all.
    threshold = model_config.METRIC_THRESHOLDS.get(
        KEY, model_config.DEFAULT_METRIC_THRESHOLD
    )
    return SemanticSimilarityMetric(threshold=threshold)
