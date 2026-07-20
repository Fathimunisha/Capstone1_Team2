from regulatory_compliance.models.request import AskRequest
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.retrievers.hybrid_retrievers import HybridRetriever
from regulatory_compliance.agents.rag_agents import RAGAgent


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

    def __init__(self):

        self.retriever = HybridRetriever(top_k=5)

        self.agent = RAGAgent()

    def process_query(self, question: str):
        print("1. Query received:", question)
        return self.agent.run(question, [])

        documents = self.retriever.search(question)
        print("2. Retrieval completed. Documents:", len(documents))

        answer = self.agent.generate_answer(question, documents)
        print("3. LLM response generated")
        sources = []

        for doc in documents:

            document_id = doc.metadata.get("document_id")

            sources.append(
                {
                    "document_id": str(document_id) if document_id else None,
                    "chunk_index": int(doc.metadata.get("chunk_index", 0)),
                    "hybrid_score": float(doc.metadata.get("hybrid_score", 0.0)),
                }
            )
        print("4. Response prepared")
        response = {"answer": answer, "sources": sources}

        print("FINAL RESPONSE:")
        print(response)

        return response
