from deepeval.models import AnthropicModel, OllamaModel

from config import model_config
from config.generator_resolver import get_generator_provider


JUDGE_PROVIDER_CLASSES = {
    "anthropic": AnthropicModel,
    "ollama": OllamaModel,
}


def get_judge_provider():
    return model_config.JUDGE_PROVIDER


def load_model():

    provider = get_judge_provider()

    model_class = JUDGE_PROVIDER_CLASSES.get(provider)

    if model_class is None:
        raise ValueError(
            f"Unknown JUDGE_PROVIDER '{provider}'. "
            f"Available providers: {list(JUDGE_PROVIDER_CLASSES.keys())}"
        )

    kwargs = {
        "model": model_config.JUDGE_MODEL,
        "temperature": model_config.JUDGE_TEMPERATURE,
    }

    # Previously omitted for every provider, which meant an Ollama judge
    # silently fell back to deepeval's default host instead of the configured
    # one -- generation would keep working while judging broke, with errors
    # the tiering downgrades to not_applicable rather than a loud failure.
    judge_endpoint = getattr(model_config, "JUDGE_ENDPOINT", None)
    if provider == "ollama" and judge_endpoint:
        kwargs["base_url"] = judge_endpoint

    return model_class(**kwargs)


def get_judge_independence():
    """True when the judge is not the model under test.

    Compares the model, not just the provider: two different models from the
    same provider (phi3 answering, llama3 judging) are independent judges, and
    reporting them as dependent would understate the trust available. The
    provider check stays because a provider change always implies a model
    change."""

    return (
        get_judge_provider() != get_generator_provider()
        or model_config.JUDGE_MODEL != model_config.GENERATOR_MODEL
    )
