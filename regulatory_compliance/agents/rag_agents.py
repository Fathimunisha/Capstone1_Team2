# from typing import List
# from openai import OpenAI
# from langchain_core.documents import Document
# from regulatory_compliance.core.config import settings


# class RAGAgent:
#     """
#     Regulatory Compliance RAG Agent.

#     Responsibilities:
#     - Accept retrieved context
#     - Generate compliance-focused response
#     - Enforce citation requirement
#     """

#     def __init__(self):

#         self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

#     def generate_answer(self, query: str, documents: List[Document]) -> str:

#         context = self.build_context(documents)

#         system_prompt = """
#             You are a Regulatory Compliance Assistant.

#             Your role:
#             - Answer only using provided regulatory context.
#             - Do not hallucinate.
#             - If information is unavailable, clearly state:
#             "Information not available in provided documents."

#             Rules:
#             1. Answer only from provided retrieved documents.
#             2. Do not create information outside the context.
#             3. Provide concise compliance interpretation.
#             4. Do not generate a Sources section.
#             5. Citations will be handled separately by the application.
#             6. If information is unavailable, clearly state:
#             "Information is not available in the provided documents."

#             Temprature: 0

#             """

#         response = self.client.chat.completions.create(
#             model=settings.OPENAI_MODEL,
#             # temperature=0,
#             max_completion_tokens=2000,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {
#                     "role": "user",
#                     "content": f"""

#         User Query: {query}
#         Retrieved Regulatory Context:
#         {context}
#         """,
#                 },
#             ],
#         )

#         return response.choices[0].message.content

#     def build_context(self, documents: List[Document]) -> str:

#         context = ""

#         for index, doc in enumerate(documents, start=1):

#             context += f"""

#         --- Context {index} ---

#         Document ID:
#         {doc.metadata.get("document_id")}

#         Chunk:
#         {doc.metadata.get("chunk_index")}

#         Content:

#         {doc.page_content}


#         """

#         return context


import time
from typing import List, Dict

from langsmith import traceable

from regulatory_compliance.retrievers.hybrid_retrievers import HybridRetriever
from regulatory_compliance.services.llm_service import LLMService

CHITCHAT_PATTERNS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "thanks",
    "thank you",
    "how are you",
    "who are you",
]


class RAGAgent:

    def __init__(self):

        self.retriever = HybridRetriever(top_k=5)

        self.llm_service = LLMService()

    def is_chitchat(self, question: str):

        q = question.lower().strip()

        return any(
            q == pattern or q.startswith(pattern) for pattern in CHITCHAT_PATTERNS
        )

    def select_tool(self, question: str):

        q = question.lower()

        if any(keyword in q for keyword in ["exact", "clause", "section", "id"]):

            return "fts_search"

        if any(keyword in q for keyword in ["explain", "meaning", "concept"]):

            return "vector_search"

        return "hybrid_search"

    @traceable(name="rag_agent")
    def run(self, question: str, chat_history: List[Dict] = None):

        start_time = time.time()

        if chat_history is None:

            chat_history = []

        # -------------------------
        # Chitchat
        # -------------------------

        if self.is_chitchat(question):

            return {
                "answer": (
                    "Hello! I am your Regulatory Compliance AI assistant. "
                    "Please ask questions related to uploaded regulatory documents."
                ),
                "query_type": "chitchat",
                "tool_used": None,
                "sources": [],
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "confidence": 1.0,
            }

        # -------------------------
        # Tool Selection
        # -------------------------

        tool_name = self.select_tool(question)

        # -------------------------
        # Retrieval
        # -------------------------

        #
        # Currently HybridRetriever
        # is the main retriever
        #

        documents = self.retriever.search(question)

        if not documents:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "query_type": "rag",
                "tool_used": tool_name,
                "sources": [],
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "confidence": 0.2,
            }

        # -------------------------
        # Build Context
        # -------------------------

        context = "\n\n".join([(f"""
Source:
{doc.metadata.get('file_name')}

Page:
{doc.metadata.get('page_number')}

Section:
{doc.metadata.get('section_number')}

Content:
{doc.page_content}
""") for doc in documents])

        # -------------------------
        # LLM Answer
        # -------------------------

        answer = self.llm_service.generate_answer(question, context)

        # -------------------------
        # Citation Builder
        # -------------------------

        sources = []

        for doc in documents:

            metadata = doc.metadata

            sources.append(
                {
                    "document_id": metadata.get("document_id"),
                    "file_name": metadata.get("file_name"),
                    "page_number": metadata.get("page_number"),
                    "section_number": metadata.get("section_number"),
                    "regulation_type": metadata.get("regulation_type"),
                    "chunk_index": metadata.get("chunk_index"),
                    "retrieval_method": metadata.get("retrieval_method"),
                    "vector_score": metadata.get("vector_score"),
                    "fts_score": metadata.get("fts_score"),
                    "hybrid_score": metadata.get("hybrid_score"),
                    "snippet": doc.page_content[:300],
                }
            )

        return {
            "answer": answer,
            "query_type": "rag",
            "tool_used": tool_name,
            "sources": sources,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "confidence": 0.85,
        }
