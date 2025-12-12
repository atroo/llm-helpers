"""Tests for file_utils module."""

import pytest
from pathlib import Path
from fastapi import UploadFile
from io import BytesIO
from src.llm_helpers.file_utils import file_to_message
from src.llm_helpers.get_llm import get_llm
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def create_upload_file_from_pdf(pdf_path: Path) -> UploadFile:
    """Helper to create an UploadFile from a PDF path."""
    with open(pdf_path, "rb") as f:
        pdf_content = f.read()
    
    file = UploadFile(
        filename=pdf_path.name,
        file=BytesIO(pdf_content),
        headers={"content-type": "application/pdf"},
        # content_type = "application/pdf"
    )
    return file


@pytest.mark.asyncio
async def test_file_to_message_openai_with_real_pdf():
    """Test file_to_message with OpenAI using a real PDF file."""
    pdf_path = Path(__file__).parent / "data" / "32047_53837_Ergaenzende_Informationen_zu_Ihrer_Abgabe.pdf"
    
    file = create_upload_file_from_pdf(pdf_path)
    
    # Get LLM client and invoke
    llm, provider = get_llm(model_string="openai:gpt-5.1:none")
    message = await file_to_message(file, provider)
    
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this PDF document? Give a brief summary."},
            message
        ]
    )
    
    response = await llm.ainvoke([human_message])
    
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0
    
    print(f"\nOpenAI Response: {response.content}")


@pytest.mark.asyncio
async def test_file_to_message_azure_with_real_pdf():
    """Test file_to_message with Azure using a real PDF file."""
    pdf_path = Path(__file__).parent / "data" / "32047_53837_Ergaenzende_Informationen_zu_Ihrer_Abgabe.pdf"
    
    file = create_upload_file_from_pdf(pdf_path)
    
    # Get LLM client and invoke
    llm, provider = get_llm(model_string='azure:gpt-5-chat')
    message = await file_to_message(file, provider)
    
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "What is in this PDF document? Give a brief summary."},
            message
        ]
    )
    
    response = await llm.ainvoke([human_message])
    
    assert response is not None
    assert hasattr(response, "content")
    assert len(response.content) > 0
    
    print(f"\nAzure Response: {response.content}")


@pytest.mark.asyncio
async def test_file_to_message_groq_not_implemented():
    """Test that Groq provider raises NotImplementedError with real PDF."""
    pdf_path = Path(__file__).parent / "data" / "32047_53837_Ergaenzende_Informationen_zu_Ihrer_Abgabe.pdf"
    
    file = create_upload_file_from_pdf(pdf_path)
    
    with pytest.raises(NotImplementedError, match="File upload not supported for Groq"):
        await file_to_message(file, "groq")


@pytest.mark.asyncio
async def test_file_to_message_unsupported_provider():
    """Test that unsupported provider raises ValueError."""
    pdf_path = Path(__file__).parent / "data" / "32047_53837_Ergaenzende_Informationen_zu_Ihrer_Abgabe.pdf"
    
    file = create_upload_file_from_pdf(pdf_path)
    
    with pytest.raises(ValueError, match="Unsupported model provider: invalid"):
        await file_to_message(file, "invalid")


