from regulatory_compliance.models.request import AskRequest
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.agents.rag_agents import RAGAgent


class QueryService:
    """
    Handles user query operations.
    """

    def __init__(self):
        self.agent = RAGAgent()

    @staticmethod
    async def ask_question(request: AskRequest) -> ApiResponse:
        """
        Process user question.

        This method is kept for backward compatibility with the
        older /ask endpoint.
        """

        return ApiResponse(
            success=True,
            message="Question processed successfully.",
            data={
                "question": request.question,
                "answer": "This is a placeholder response. RAG implementation will be added in the next phase.",
            },
        )

    def process_query(self, question: str):
        """
        Process a user query through the RAG Agent.

        The RAGAgent is responsible for:
        - Classifying the query as CHITCHAT, REGULATORY, or OUT_OF_SCOPE
        - Deciding whether document retrieval is required
        - Selecting the retrieval tool
        - Retrieving documents for regulatory questions
        - Generating the final answer
        """

        print("1. Query received:", question)

        result = self.agent.run(
            question,
            [],
        )

        print("2. Query processing completed")
        print("Query type:", result.get("query_type"))
        print("Tool used:", result.get("tool_used"))

        return result