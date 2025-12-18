"""Factory functions for creating LLM client instances."""

import os
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from .const import MODEL_PROVIDERS
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr
from .parse_model_string import parse_model_string

def get_llm(
    model_env: str | None = None,
    model_string: str | None = None,
    streaming: bool = True,
) -> tuple[BaseChatModel, MODEL_PROVIDERS]:
    """
    Get a configured LLM client based on model specification.

    Args:
        model_env: Environment variable name containing the model specification
        model_string: Direct model specification string
        streaming: Whether to enable streaming responses

    Returns:
        Tuple of (llm_client, provider_name)

    Raises:
        ValueError: If model specification is invalid or provider is unsupported

    Model specification format:
        - "provider:model_name" or "provider:model_name:reasoning"
        - Examples: "openai:gpt-5.1:none", "azure:gpt-5-chat"
    """

    provider, model_name, reasoning_effort = parse_model_string(model_env, model_string)

    params = {
        "output_version": "responses/v1",
        "streaming": streaming,
        "model": model_name,
    }
    if reasoning_effort is not None:
        params["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    match provider:
        case "azure":
            base_url = os.environ["AZURE_BASE_URL"]
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
            
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=SecretStr(api_key),
                **params,
            )
            return llm, provider   

        case "openai":
            llm = ChatOpenAI(**params)

            return llm, provider

        case "groq":
            llm = ChatGroq(**params)

            return llm, provider
        case "mistralai":
            llm = ChatMistralAI(**params)

            return llm, provider
        case "google":
            llm = ChatGoogleGenerativeAI(
                **params,
            )
            return llm, provider

        case _:
            raise ValueError(f"Unsupported provider: {provider}, supported: 'openai', 'azure'")
