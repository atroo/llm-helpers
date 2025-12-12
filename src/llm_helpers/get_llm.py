"""Factory functions for creating LLM client instances."""

import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


MODEL_PROVIDERS = ["openai", "azure", "groq"]
TEST_MODEL_STRINGS = {'openai':'openai:gpt-5.1:none', 'azure':'azure:gpt-5-chat', 'groq':'groq:openai/gpt-oss-120b'}


def get_llm(
    model_env: str | None = None,
    model_string: str | None = None,
    streaming: bool = True,
):
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
    if model_string is not None:
        model_name = model_string
    elif model_env is not None:
        model_name = os.getenv(model_env, "openai:gpt-5.1:none")
    else:
        model_name = "openai:gpt-5.1:none"

    if len(model_name.split(":")) == 2:
        provider, model_name = model_name.split(":")
        reasoning_effort = None
    elif len(model_name.split(":")) == 3:
        provider, model_name, reasoning_effort = model_name.split(":")
    else:
        raise ValueError(
            f"Model name {model_name} is not valid. It must be in the format 'provider:model_name[:reasoning]', e.g., 'openai:gpt-5.1:none' e.g. 'azure:gpt-5-chat'."
        )

    if provider not in MODEL_PROVIDERS:
        raise ValueError(
            f"Provider {provider} is not supported. Supported providers are: {MODEL_PROVIDERS}"
        )

    print(f"{model_env}: Using LLM model: {model_name} from provider: {provider}")

    params = {
        "output_version": "responses/v1",
        "streaming": streaming,
        "model": model_name,
        "use_responses_api": True,
    }
    if reasoning_effort is not None:
        params["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    match provider:
        case "azure":
            llm = ChatOpenAI(
                base_url=os.environ["AZURE_BASE_URL"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                **params,
            )
            return llm, provider

        case "openai":
            llm = ChatOpenAI(**params)

            return llm, provider

        case "groq":
            del params["use_responses_api"]
            llm = ChatGroq(**params)

            return llm, provider

        case _:
            raise ValueError(
                f"Unsupported provider: {provider}, supported: 'openai', 'azure'"
            )
