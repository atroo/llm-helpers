"""Tests for file_utils module."""

import pytest
from pathlib import Path
from fastapi import UploadFile
from io import BytesIO
from src.llm_helpers.file_utils import file_to_message
from src.llm_helpers.get_llm import get_llm
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

def create_upload_file_from_pdf(pdf_path: Path) -> UploadFile:
    """Helper to create an UploadFile from a PDF path."""
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    file = UploadFile(
        filename=pdf_path.name,
        file=BytesIO(pdf_content),
        headers={"content-type": "application/pdf"},
    )
    return file


def get_test_pdf_path() -> Path:
    """Get the path to the test PDF file."""
    return Path(__file__).parent / "data" / "32047_53837_Ergaenzende_Informationen_zu_Ihrer_Abgabe.pdf"


async def _test_file_to_message_with_provider(model_string: str):
    """Helper function to test file_to_message with a given provider."""
    pdf_path = get_test_pdf_path()
    file = create_upload_file_from_pdf(pdf_path)
    
    # Get LLM client and invoke
    llm, provider = get_llm(model_string=model_string)
    message = await file_to_message(file, provider)
    
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this PDF document? Give a brief summary."},
            message
        ]
    )
    
    response = await llm.ainvoke([human_message])
    
    assert isinstance(response, AIMessage)
    print(f"\n{provider} Response: {response.content}")
    return response


@pytest.mark.asyncio
async def test_file_to_message_openai_with_real_pdf():
    await _test_file_to_message_with_provider("openai:gpt-5.1:none")


@pytest.mark.asyncio
async def test_file_to_message_azure_with_real_pdf():
    await _test_file_to_message_with_provider("azure:gpt-5-chat")


@pytest.mark.asyncio
async def test_file_to_message_groq_not_implemented():
    pdf_path = get_test_pdf_path()
    file = create_upload_file_from_pdf(pdf_path)
    
    with pytest.raises(NotImplementedError, match="File upload not supported for Groq"):
        await file_to_message(file, "groq")


@pytest.mark.asyncio
async def test_file_to_message_google_with_real_pdf():
    await _test_file_to_message_with_provider("google:gemini-3-flash-preview:minimal")


@pytest.mark.asyncio
async def test_file_to_message_unsupported_provider():
    pdf_path = get_test_pdf_path()
    file = create_upload_file_from_pdf(pdf_path)
    
    with pytest.raises(ValueError, match="Unsupported model provider: invalid"):
        await file_to_message(file, "invalid")


