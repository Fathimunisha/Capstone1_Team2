from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request model for asking questions to the RAG Agent.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question for Regulatory Compliance Bot",
        examples=["What is Basel III?"],
    )


class UploadRequest(BaseModel):
    """
    Metadata for uploaded document.
    (The actual PDF file will be received separately using UploadFile.)
    """

    document_name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Name of the uploaded PDF document",
        examples=["Capstone_Project_1_Regulatory_Compliance_System_FAQ.pdf"],
    )


class QueryRequest(BaseModel):

    question: str
