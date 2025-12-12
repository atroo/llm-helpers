"""Tests for LLM client functions."""

import pytest
from langchain_core.messages import HumanMessage
from src.llm_helpers.get_llm import get_llm, MODEL_PROVIDERS, TEST_MODEL_STRINGS
from dotenv import load_dotenv

load_dotenv()


def has_reasoning(response) -> bool:
    """
    Check if the response contains reasoning.
    
    Args:
        response: The LLM response object
        
    Returns:
        True if reasoning is present, False otherwise
    """
    if not hasattr(response, "content"):
        return False
    
    # If content is a list, check for reasoning type
    if isinstance(response.content, list):
        return any(
            isinstance(item, dict) and item.get("type") == "reasoning"
            for item in response.content
        )
    
    return False


def has_no_reasoning(response) -> bool:
    """
    Check if the response does NOT contain reasoning.
    
    Args:
        response: The LLM response object
        
    Returns:
        True if no reasoning is present, False otherwise
    """
    return not has_reasoning(response)

@pytest.mark.asyncio
async def test_get_llm_openai():
    """Test get_llm with OpenAI provider."""
    llm, provider = get_llm(model_string=TEST_MODEL_STRINGS["openai"])
    
    assert provider == "openai"
    assert llm is not None
    
    # Invoke the LLM with a simple message
    message = HumanMessage(content="Say 'hello' in one word.")
    response = await llm.ainvoke([message])
    
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0
    assert has_no_reasoning(response)
    
    print(f"\nOpenAI Response: {response.content}")


@pytest.mark.asyncio
async def test_get_llm_azure():
    """Test get_llm with Azure provider."""
    llm, provider = get_llm(model_string=TEST_MODEL_STRINGS["azure"])
    
    assert provider == "azure"
    assert llm is not None
    
    # Invoke the LLM with a simple message
    message = HumanMessage(content="Say 'hello' in one word.")
    response = await llm.ainvoke([message])
    
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0
    assert has_no_reasoning(response)
    
    print(f"\nAzure Response: {response.content}")


@pytest.mark.asyncio
async def test_get_llm_groq():
    """Test get_llm with Groq provider."""
    llm, provider = get_llm(model_string=TEST_MODEL_STRINGS["groq"])
    
    assert provider == "groq"
    assert llm is not None
    
    # Invoke the LLM with a simple message
    message = HumanMessage(content="Say 'hello' in one word.")
    response = await llm.ainvoke([message])
    
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0
    assert has_no_reasoning(response)
    
    print(f"\nGroq Response: {response.content}")


@pytest.mark.asyncio
async def test_get_llm_with_reasoning():
    """Test get_llm with reasoning parameter."""
    llm, provider = get_llm(model_string="openai:gpt-5.1:low")
    
    assert provider == "openai"
    assert llm is not None
    
    # Invoke the LLM with a simple reasoning task
    message = HumanMessage(content="What is 2+2?")
    response = await llm.ainvoke([message])
    
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0
    assert has_reasoning(response)
    
    print(f"\nOpenAI with reasoning Response: {response.content}")


@pytest.mark.asyncio
async def test_get_llm_from_env():
    """Test get_llm reading from environment variable."""
    import os
    
    # Set environment variable
    os.environ["TEST_MODEL"] = TEST_MODEL_STRINGS["openai"]
    
    llm, provider = get_llm(model_env="TEST_MODEL")
    
    assert provider == "openai"
    assert llm is not None
    
    # Invoke the LLM
    message = HumanMessage(content="Say 'hello' in one word.")
    response = await llm.ainvoke([message])
    
    assert response is not None
    assert len(response.content) > 0
    assert has_no_reasoning(response)
    
    print(f"\nEnv-based Response: {response.content}")
    
    # Cleanup
    del os.environ["TEST_MODEL"]


def test_get_llm_unsupported_provider():
    """Test that unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Provider invalid is not supported"):
        get_llm(model_string="invalid:some-model")


def test_get_llm_invalid_format():
    """Test that invalid model string format raises ValueError."""
    with pytest.raises(ValueError, match="is not valid"):
        get_llm(model_string="invalid_format")


def test_model_providers_constant():
    """Test that MODEL_PROVIDERS contains expected providers."""
    assert "openai" in MODEL_PROVIDERS
    assert "azure" in MODEL_PROVIDERS
    assert "groq" in MODEL_PROVIDERS
    assert len(MODEL_PROVIDERS) == 3
