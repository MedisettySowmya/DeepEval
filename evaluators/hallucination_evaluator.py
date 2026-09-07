from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from config.ollama_config import load_model


class HallucinationEvaluator:

    def __init__(self):

        self.metric = HallucinationMetric(
            threshold=0.1,
            model=load_model()
        )

    def evaluate(self, input_text, actual_output_text, context_text):

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output_text,
            context=context_text
        )

        self.metric.measure(test_case)

        return {
            "score": self.metric.score,
            "reason": self.metric.reason,
            "success": self.metric.success            
        }