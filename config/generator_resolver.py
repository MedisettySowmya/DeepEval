from config import model_config
from config.ollama_config import generate_response as ollama_generate_response


GENERATOR_PROVIDER_FUNCTIONS = {
    "ollama": ollama_generate_response,
}


def get_generator_provider():
    return model_config.GENERATOR_PROVIDER


def generate_response(prompt):

    provider = get_generator_provider()

    generator = GENERATOR_PROVIDER_FUNCTIONS.get(provider)

    if generator is None:
        raise ValueError(
            f"Unknown GENERATOR_PROVIDER '{provider}'. "
            f"Available providers: {list(GENERATOR_PROVIDER_FUNCTIONS.keys())}"
        )

    return generator(
        prompt,
        model=model_config.GENERATOR_MODEL,
        endpoint=model_config.GENERATOR_ENDPOINT,
        temperature=model_config.GENERATOR_TEMPERATURE,
        timeout=model_config.GENERATOR_TIMEOUT
    )
