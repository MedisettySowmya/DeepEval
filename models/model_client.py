"""Connect configured Ollama models to the evaluation pipeline."""

import ollama
from deepeval.models import OllamaModel

from config import settings


def create_generator_client() -> ollama.Client:
    """Build an Ollama client using the configured endpoint and timeout."""
    return ollama.Client(
        host=settings.GENERATOR_ENDPOINT,
        timeout=settings.GENERATOR_TIMEOUT,
    )


def generate_response(prompt, client=None):
    """Send a prompt to the generator model and return its answer."""
    # Create a client if the caller has not provided one.
    if client is None:
        client = create_generator_client()

    response = client.chat(
        model=settings.GENERATOR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": settings.GENERATOR_TEMPERATURE},
    )
    return response["message"]["content"]


def create_judge() -> OllamaModel:
    """Build the DeepEval judge; metrics make the evaluation requests later."""
    return OllamaModel(
        model=settings.JUDGE_MODEL,
        base_url=settings.JUDGE_ENDPOINT,
        temperature=settings.JUDGE_TEMPERATURE,
        timeout=settings.JUDGE_TIMEOUT,
    )


def uses_different_models() -> bool:
    """Compare configured identities; this does not establish judge independence."""
    return (
        settings.GENERATOR_PROVIDER != settings.JUDGE_PROVIDER
        or settings.GENERATOR_MODEL != settings.JUDGE_MODEL
    )
