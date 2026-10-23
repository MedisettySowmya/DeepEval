"""Check model-client code without calling a real Ollama server.

Mock creates a fake client. return_value is its pretend answer, while
side_effect can make it raise an error. monkeypatch temporarily replaces
settings or clients for one test and restores them afterward.
"""

from unittest.mock import Mock

import httpx
import pytest

from config import settings
from models import model_client


def test_generator_uses_configured_request(monkeypatch):
    # Setup: choose test settings and a fake answer.
    monkeypatch.setattr(settings, "GENERATOR_MODEL", "test-generator")
    monkeypatch.setattr(settings, "GENERATOR_ENDPOINT", "http://localhost:1234")
    monkeypatch.setattr(settings, "GENERATOR_TIMEOUT", 42)
    monkeypatch.setattr(settings, "GENERATOR_TEMPERATURE", 0.2)
    fake_client = Mock()
    fake_client.chat.return_value = {"message": {"content": "test answer"}}
    fake_client_factory = Mock(return_value=fake_client)
    monkeypatch.setattr(model_client.ollama, "Client", fake_client_factory)

    # Action: ask a question.
    answer = model_client.generate_response("test question")

    # Check: the answer and connection/request settings are correct.
    assert answer == "test answer"
    fake_client_factory.assert_called_once_with(host="http://localhost:1234", timeout=42)
    fake_client.chat.assert_called_once_with(
        model="test-generator",
        messages=[{"role": "user", "content": "test question"}],
        options={"temperature": 0.2},
    )


def test_generator_reuses_supplied_client(monkeypatch):
    # Setup: supply a client so no new client should be created.
    fake_client = Mock()
    fake_client.chat.return_value = {"message": {"content": "test answer"}}
    fake_client_factory = Mock()
    monkeypatch.setattr(model_client.ollama, "Client", fake_client_factory)

    # Action and check: send two questions through the same client.
    first_answer = model_client.generate_response("first question", client=fake_client)
    assert first_answer == "test answer"
    fake_client.chat.assert_called_with(
        model=settings.GENERATOR_MODEL,
        messages=[{"role": "user", "content": "first question"}],
        options={"temperature": settings.GENERATOR_TEMPERATURE},
    )

    second_answer = model_client.generate_response("second question", client=fake_client)
    assert second_answer == "test answer"
    fake_client.chat.assert_called_with(
        model=settings.GENERATOR_MODEL,
        messages=[{"role": "user", "content": "second question"}],
        options={"temperature": settings.GENERATOR_TEMPERATURE},
    )
    assert fake_client.chat.call_count == 2
    fake_client_factory.assert_not_called()


def test_generator_passes_timeout_to_caller(monkeypatch):
    # Setup: make the automatically created client raise a timeout.
    fake_client = Mock()
    timeout_error = httpx.ReadTimeout("request timed out")
    fake_client.chat.side_effect = timeout_error
    fake_client_factory = Mock(return_value=fake_client)
    monkeypatch.setattr(model_client.ollama, "Client", fake_client_factory)

    # Action and check: the caller receives the original error.
    with pytest.raises(httpx.ReadTimeout) as caught_error:
        model_client.generate_response("question")
    assert caught_error.value is timeout_error


def test_supplied_client_passes_timeout_to_caller():
    # Setup: make a supplied client raise a timeout.
    fake_client = Mock()
    timeout_error = httpx.ReadTimeout("request timed out")
    fake_client.chat.side_effect = timeout_error

    # Action and check: the caller receives the original error.
    with pytest.raises(httpx.ReadTimeout) as caught_error:
        model_client.generate_response("question", client=fake_client)
    assert caught_error.value is timeout_error


def test_judge_uses_separate_configuration(monkeypatch):
    # Setup: choose judge settings and replace the real judge with a fake.
    monkeypatch.setattr(settings, "JUDGE_MODEL", "test-judge")
    monkeypatch.setattr(settings, "JUDGE_ENDPOINT", "http://localhost:5678")
    monkeypatch.setattr(settings, "JUDGE_TEMPERATURE", 0.1)
    monkeypatch.setattr(settings, "JUDGE_TIMEOUT", 90)
    fake_judge = Mock()
    fake_judge_factory = Mock(return_value=fake_judge)
    monkeypatch.setattr(model_client, "OllamaModel", fake_judge_factory)

    # Action: create the judge.
    judge = model_client.create_judge()

    # Check: creating the judge uses its settings but does not ask a question.
    assert judge is fake_judge
    fake_judge_factory.assert_called_once_with(
        model="test-judge",
        base_url="http://localhost:5678",
        temperature=0.1,
        timeout=90,
    )
    fake_judge.generate.assert_not_called()


def test_same_model_names_return_false(monkeypatch):
    # Setup: both roles use the same provider and model name.
    monkeypatch.setattr(settings, "GENERATOR_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "JUDGE_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "GENERATOR_MODEL", "same-model")
    monkeypatch.setattr(settings, "JUDGE_MODEL", "same-model")

    # Action and check.
    result = model_client.uses_different_models()
    assert result is False


def test_different_model_names_return_true(monkeypatch):
    # Setup: the provider is the same, but the model names differ.
    monkeypatch.setattr(settings, "GENERATOR_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "JUDGE_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "GENERATOR_MODEL", "generator-model")
    monkeypatch.setattr(settings, "JUDGE_MODEL", "judge-model")

    # Action and check.
    result = model_client.uses_different_models()
    assert result is True
