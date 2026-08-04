# from regulatory_compliance.models.request import AskRequest
# from regulatory_compliance.models.response import ApiResponse
# from regulatory_compliance.agents.rag_agents import run_agent


# class QueryService:
#     """
#     Handles user query operations.
#     """

#     @staticmethod
#     async def ask_question(request: AskRequest) -> ApiResponse:
#         """
#         Process user question.
#         This method is kept for backward compatibility with the
#         older /ask endpoint.
#         """

#         return ApiResponse(
#             success=True,
#             message="Question processed successfully.",
#             data={
#                 "question": request.question,
#                 "answer": "This is a placeholder response. RAG implementation will be added in the next phase.",
#             },
#         )

#     def process_query(self, question: str, chat_history: None):
#         """
#         Process a user query through the RAG Agent.
#         The RAGAgent is responsible for:
#         - Selecting the retrieval tool
#         - Retrieving documents for regulatory questions
#         - Generating the final answer
#         """

#         print("1. Query received:", question, chat_history)
#         result = run_agent(
#             question,
#             chat_history,
#         )

#         print("2. Query processing completed")
#         print("Query type:", result.get("query_type"))
#         print("Tool used:", result.get("tool_used"))
#         return result


from regulatory_compliance.models.request import AskRequest
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.agents.rag_agents import run_agent


class QueryService:
    """
    Handles user query operations.
    """

    @staticmethod
    async def ask_question(request: AskRequest) -> ApiResponse:
        """
        Process user question.

        This method is kept for backward compatibility
        with the older /ask endpoint.
        """

        return ApiResponse(
            success=True,
            message="Question processed successfully.",
            data={
                "question": request.question,
                "answer": (
                    "This is a placeholder response. "
                    "RAG implementation will be added in the next phase."
                ),
            },
        )

    def process_query(
        self,
        question: str,
        chat_history: list | None = None,
    ):
        """
        Process a user query through the RAG Agent.

        The RAG Agent is responsible for:
        - Selecting the retrieval tool
        - Retrieving documents for regulatory questions
        - Generating the final answer
        """

        print(
            "1. Query received:",
            question,
            chat_history,
        )

        result = run_agent(
            question,
            chat_history,
        )

        print("2. Query processing completed")

        print(
            "Query type:",
            result.get("query_type"),
        )

        print(
            "Tool used:",
            result.get("tool_used"),
        )

        return result
