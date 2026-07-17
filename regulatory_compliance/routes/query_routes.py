from fastapi import APIRouter, status
from regulatory_compliance.models.request import AskRequest
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/ask", response_model=ApiResponse, status_code=status.HTTP_200_OK)
async def ask_question(request: AskRequest) -> ApiResponse:
    """
    Ask a question to the Regulatory Compliance RAG Bot.
    """

    response = await QueryService.ask_question(request)
    return response
