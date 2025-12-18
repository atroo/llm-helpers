"""Utilities for handling file uploads for different LLM providers."""

import base64
from fastapi import UploadFile


async def file_to_message(file: UploadFile, model_provider: str) -> dict:
    """
    Converts an uploaded file to a message dictionary suitable as input for different model providers.
    
    Args:
        file: The uploaded file from FastAPI
        model_provider: The LLM provider ('openai', 'azure', 'groq')
        
    Returns:
        A dictionary formatted for the specified model provider
        
    Raises:
        ValueError: If the model provider is not supported
        NotImplementedError: If the feature is not yet implemented for the provider
    """
    content = await file.read()
    content_b64 = base64.b64encode(content).decode("utf-8")

    match model_provider:
        case "openai":
            return {
                "type": "input_file",
                "filename": file.filename,
                "file_data": f"data:{file.content_type};base64,{content_b64}",
            }
        case "azure":
            return {
                "type": "file",
                "file": {
                    "type": "input_file",
                    "filename": file.filename,
                    "file_data": f"data:{file.content_type};base64,{content_b64}",
                },
            }
        case "groq":
            raise NotImplementedError("File upload not supported for Groq models yet.")
        case "mistralai":
            raise NotImplementedError("File upload not supported for MistralAI models yet.")
        
        # https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai#multimodal-usage
        case "google":
            return {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": f"{file.content_type}",
                    "data": content_b64,
                }
        case _:
            raise ValueError(f"Unsupported model provider: {model_provider}")
