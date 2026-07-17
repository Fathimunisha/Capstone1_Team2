from regulatory_compliance.models.request import AskRequest
from regulatory_compliance.models.response import ApiResponse


class QueryService:
    """
    Handles user query operations.
    """

    @staticmethod
    async def ask_question(request: AskRequest) -> ApiResponse:
        """
        Process user question.
        """

        return ApiResponse(
            success=True,
            message="Question processed successfully.",
            data={
                "question": request.question,
                "answer": "This is a placeholder response. RAG implementation will be added in the next phase.",
            },
        )
