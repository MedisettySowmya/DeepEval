import pytest

from config import settings
from prompts.prompt_builder import get_instruction


def test_environment_overrides_dotenv_and_default(monkeypatch):
    monkeypatch.setattr(settings, "_DOTENV", {"CONFIG_TEST_VALUE": "file"})
    monkeypatch.delenv("CONFIG_TEST_VALUE", raising=False)
    assert settings.setting("CONFIG_TEST_VALUE", "default") == "file"
    monkeypatch.setenv("CONFIG_TEST_VALUE", "environment")
    assert settings.setting("CONFIG_TEST_VALUE", "default") == "environment"
    monkeypatch.delenv("CONFIG_TEST_VALUE")
    monkeypatch.setattr(settings, "_DOTENV", {})
    assert settings.setting("CONFIG_TEST_VALUE", "default") == "default"


def test_boolean_and_fractional_settings(monkeypatch):
    monkeypatch.setenv("CONFIG_TEST_VALUE", "false")
    assert settings.setting("CONFIG_TEST_VALUE", True) is False
    monkeypatch.setenv("CONFIG_TEST_VALUE", "0.25")
    assert settings.setting("CONFIG_TEST_VALUE", 0.0) == 0.25


@pytest.mark.parametrize("value,default", [("", "model"), ("maybe", True), ("1.5", 300)])
def test_bad_environment_values_have_named_errors(monkeypatch, value, default):
    monkeypatch.setenv("CONFIG_TEST_VALUE", value)
    with pytest.raises(settings.ConfigurationError, match="CONFIG_TEST_VALUE"):
        settings.setting("CONFIG_TEST_VALUE", default)


@pytest.mark.parametrize("name,value", [
    ("INSTRUCTION_VARIANT", "resistence"),
    ("GENERATOR_PROVIDER", "unsupported"),
    ("JUDGE_PROVIDER", "anthropic"),
    ("JUDGE_MODEL", " "),
    ("JUDGE_TIMEOUT", 0),
    ("JUDGE_ENDPOINT", "localhost:11434"),
    ("GENERATOR_ENDPOINT", "localhost:11434"),
    ("GENERATOR_ENDPOINT", "http://localhost:99999"),
    ("GENERATOR_TIMEOUT", 0),
    ("JUDGE_TEMPERATURE", float("nan")),
    ("HUMAN_REVIEW_SAMPLE_RATE", 1.1),
])
def test_invalid_configuration_fails_early(monkeypatch, name, value):
    monkeypatch.setattr(settings, name, value)
    with pytest.raises(settings.ConfigurationError, match=name):
        settings.validate_config()


def test_invalid_metric_threshold(monkeypatch):
    monkeypatch.setattr(settings, "METRIC_THRESHOLDS", {"BLEUMetric": -0.1})
    with pytest.raises(settings.ConfigurationError, match="BLEUMetric"):
        settings.validate_config()


def test_unknown_runtime_instruction_is_not_silently_empty():
    with pytest.raises(settings.ConfigurationError, match="Unknown instruction variant"):
        get_instruction("resistence")
    assert get_instruction("none") == ("", "none")
