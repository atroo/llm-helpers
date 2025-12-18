"""Tests for LLM client functions."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.llm_helpers.get_llm import get_llm
from dotenv import load_dotenv
from src.llm_helpers.const import DEFAULT_MODEL_STRINGS

load_dotenv()



def has_reasoning(response: AIMessage) -> bool:
    """Check if the response contains reasoning."""
    # If content is a list, check for reasoning type
    if isinstance(response.content, list):
        return any(
            isinstance(item, dict) and item.get("type") == "reasoning"
            for item in response.content
        )
    
    return False


def has_no_reasoning(response: AIMessage) -> bool:
    return not has_reasoning(response)


async def run_basic_llm_test(provider_name: str, should_have_reasoning: bool = False):
    """
    Helper function to test basic LLM functionality for a given provider.
    """
    llm, provider = get_llm(model_string=DEFAULT_MODEL_STRINGS[provider_name])
    
    assert provider == provider_name
    assert llm is not None
    
    # Invoke the LLM with a simple message
    message = HumanMessage(content="Say 'hello' in one word.")
    response = await llm.ainvoke([message])

    assert isinstance(response, AIMessage)
    if should_have_reasoning:
        assert has_reasoning(response)
    else:
        assert has_no_reasoning(response)
    
    print(f"\n{provider_name.capitalize()} Response: {response.content}")


@pytest.mark.asyncio
async def test_get_llm_openai():
    """Test get_llm with OpenAI provider."""
    await run_basic_llm_test("openai")


@pytest.mark.asyncio
async def test_get_llm_azure():
    """Test get_llm with Azure provider."""
    await run_basic_llm_test("azure")


@pytest.mark.asyncio
async def test_get_llm_groq():
    """Test get_llm with Groq provider."""
    await run_basic_llm_test("groq")

@pytest.mark.asyncio
async def test_get_llm_mistralai():
    """Test get_llm with MistralAI provider."""
    await run_basic_llm_test("mistralai")

@pytest.mark.asyncio
async def test_get_llm_google():
    """Test get_llm with Google provider."""
    await run_basic_llm_test("google")


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
    assert isinstance(response, AIMessage)
    assert has_reasoning(response)
    
    print(f"\nOpenAI with reasoning Response: {response.content}")


@pytest.mark.asyncio
async def test_get_llm_from_env():
    """Test get_llm reading from environment variable."""
    import os
    
    # Set environment variable
    os.environ["TEST_MODEL"] = DEFAULT_MODEL_STRINGS["openai"]
    
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