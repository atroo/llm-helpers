from typing import get_args, cast
import os
from .const import MODEL_PROVIDERS, DEFAULT_MODEL_STRINGS


def parse_model_string(
    model_env: str | None = None, 
    model_string: str | None = None,
    provider: MODEL_PROVIDERS | None = None,
) -> tuple[MODEL_PROVIDERS, str, str | None]:

    # check that only one of model_env or model_string or provider is provided
    if sum(1 for x in [model_env, model_string, provider] if x is not None) > 1:
        raise ValueError("At most one of model_env, model_string, or provider must be provided")


    if model_string is not None:
        model_string = model_string
    elif model_env is not None:
        model_string = os.getenv(model_env, "openai:gpt-5.1:none")
    elif provider is not None:
        model_string = DEFAULT_MODEL_STRINGS[provider]
    else:
        model_string = DEFAULT_MODEL_STRINGS["openai"]

    if len(model_string.split(":")) == 2:
        _provider, model_name = model_string.split(":")
        reasoning_effort = None
    elif len(model_string.split(":")) == 3:
        _provider, model_name, reasoning_effort = model_string.split(":")
    else:
        raise ValueError(
            f"Model string {model_string} is not valid. It must be in the format 'provider:model_name[:reasoning]', e.g., 'openai:gpt-5.1:none' e.g. 'azure:gpt-5-chat'."
        )

    if _provider not in get_args(MODEL_PROVIDERS):
        raise ValueError(
            f"Provider {_provider} is not supported. Supported providers are: {get_args(MODEL_PROVIDERS)}"
        )

    return cast(MODEL_PROVIDERS, _provider), model_name, reasoning_effort
