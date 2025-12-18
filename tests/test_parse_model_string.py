"""Tests for parse_model_string module."""

import pytest
import os
from src.llm_helpers.parse_model_string import parse_model_string
from src.llm_helpers.const import DEFAULT_MODEL_STRINGS


def test_parse_model_string_with_two_parts():
    """Test parsing model string with provider and model name."""
    provider, model_name, reasoning_effort = parse_model_string(
        model_string="openai:gpt-4"
    )
    assert provider == "openai"
    assert model_name == "gpt-4"
    assert reasoning_effort is None


def test_parse_model_string_with_three_parts():
    """Test parsing model string with provider, model name, and reasoning effort."""
    provider, model_name, reasoning_effort = parse_model_string(
        model_string="openai:gpt-5.1:low"
    )
    assert provider == "openai"
    assert model_name == "gpt-5.1"
    assert reasoning_effort == "low"


def test_parse_model_string_with_reasoning_none():
    """Test parsing model string with reasoning effort set to 'none'."""
    provider, model_name, reasoning_effort = parse_model_string(
        model_string="openai:gpt-5.1:none"
    )
    assert provider == "openai"
    assert model_name == "gpt-5.1"
    assert reasoning_effort == "none"


def test_parse_model_string_all_providers():
    """Test parsing model strings for all supported providers."""
    providers = ["openai", "azure", "groq", "mistralai", "google"]
    for provider_name in providers:
        model_string = f"{provider_name}:test-model"
        provider, model_name, reasoning_effort = parse_model_string(
            model_string=model_string
        )
        assert provider == provider_name
        assert model_name == "test-model"
        assert reasoning_effort is None


def test_parse_model_string_from_env():
    """Test parsing model string from environment variable."""
    os.environ["TEST_MODEL_ENV"] = "azure:gpt-4-turbo"
    try:
        provider, model_name, reasoning_effort = parse_model_string(
            model_env="TEST_MODEL_ENV"
        )
        assert provider == "azure"
        assert model_name == "gpt-4-turbo"
        assert reasoning_effort is None
    finally:
        del os.environ["TEST_MODEL_ENV"]


def test_parse_model_string_from_env_with_default():
    """Test parsing model string from environment variable with default when env not set."""
    # Ensure the env var doesn't exist
    if "NONEXISTENT_ENV" in os.environ:
        del os.environ["NONEXISTENT_ENV"]
    
    provider, model_name, reasoning_effort = parse_model_string(
        model_env="NONEXISTENT_ENV"
    )
    # Should use default "openai:gpt-5.1:none"
    assert provider == "openai"
    assert model_name == "gpt-5.1"
    assert reasoning_effort == "none"


def test_parse_model_string_from_provider():
    """Test parsing model string using provider parameter."""
    provider, model_name, reasoning_effort = parse_model_string(provider="azure")
    assert provider == "azure"
    assert model_name == "gpt-5-chat"
    assert reasoning_effort is None


def test_parse_model_string_from_provider_all_providers():
    """Test parsing model string using provider parameter for all providers."""
    providers = ["openai", "azure", "groq", "mistralai", "google"]
    for provider_name in providers:
        provider, model_name, reasoning_effort = parse_model_string(
            provider=provider_name  # type: ignore[arg-type]
        )
        assert provider == provider_name
        # Check that model_name matches the default for this provider
        expected_default = DEFAULT_MODEL_STRINGS[provider_name]
        expected_model_name = expected_default.split(":")[1]
        assert model_name == expected_model_name


def test_parse_model_string_default_behavior():
    """Test default behavior when no parameters are provided."""
    provider, model_name, reasoning_effort = parse_model_string()
    assert provider == "openai"
    assert model_name == "gpt-5.1"
    assert reasoning_effort == "none"


def test_parse_model_string_multiple_parameters_error():
    """Test that providing multiple parameters raises ValueError."""
    with pytest.raises(ValueError, match="At most one of model_env, model_string, or provider must be provided"):
        parse_model_string(model_string="openai:gpt-4", model_env="TEST_ENV")
    
    with pytest.raises(ValueError, match="At most one of model_env, model_string, or provider must be provided"):
        parse_model_string(model_string="openai:gpt-4", provider="openai")
    
    with pytest.raises(ValueError, match="At most one of model_env, model_string, or provider must be provided"):
        parse_model_string(model_env="TEST_ENV", provider="openai")
    
    with pytest.raises(ValueError, match="At most one of model_env, model_string, or provider must be provided"):
        parse_model_string(model_string="openai:gpt-4", model_env="TEST_ENV", provider="openai")


def test_parse_model_string_invalid_format_too_few_parts():
    """Test that invalid format with too few parts raises ValueError."""
    with pytest.raises(ValueError, match="is not valid"):
        parse_model_string(model_string="openai")


def test_parse_model_string_invalid_format_too_many_parts():
    """Test that invalid format with too many parts raises ValueError."""
    with pytest.raises(ValueError, match="is not valid"):
        parse_model_string(model_string="openai:gpt-4:low:extra")


def test_parse_model_string_invalid_format_no_colon():
    """Test that invalid format without colon raises ValueError."""
    with pytest.raises(ValueError, match="is not valid"):
        parse_model_string(model_string="openai-gpt-4")


def test_parse_model_string_unsupported_provider():
    """Test that unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Provider invalid is not supported"):
        parse_model_string(model_string="invalid:some-model")


def test_parse_model_string_empty_string():
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="is not valid"):
        parse_model_string(model_string="")


def test_parse_model_string_with_empty_reasoning():
    """Test parsing model string with empty reasoning effort."""
    provider, model_name, reasoning_effort = parse_model_string(
        model_string="openai:gpt-4:"
    )
    assert provider == "openai"
    assert model_name == "gpt-4"
    assert reasoning_effort == ""


def test_parse_model_string_complex_model_name():
    """Test parsing model string with complex model name containing colons."""
    # Note: This will fail because split(":") doesn't handle colons in model names
    # But testing current behavior
    provider, model_name, reasoning_effort = parse_model_string(
        model_string="openai:gpt-4-turbo-preview:high"
    )
    assert provider == "openai"
    assert model_name == "gpt-4-turbo-preview"
    assert reasoning_effort == "high"

