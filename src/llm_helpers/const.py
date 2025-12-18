from typing import Literal, TypedDict


MODEL_PROVIDERS = Literal["openai", "azure", "groq", "mistralai", "google"]

class DefaultModelStrings(TypedDict, total=True):
    openai: str
    azure: str
    groq: str
    mistralai: str
    google: str


DEFAULT_MODEL_STRINGS: DefaultModelStrings = {
    "openai": "openai:gpt-5.1:none",
    "azure": "azure:gpt-5-chat",
    "groq": "groq:openai/gpt-oss-120b",
    "mistralai": "mistralai:mistral-large-latest",
    "google": "google:gemini-3-flash-preview:minimal"
}