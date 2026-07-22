import time
import json
from typing import List, Dict, Optional
from openai import OpenAI
from langsmith import traceable
from langchain_core.documents import Document
from regulatory_compliance.core.config import settings
from regulatory_compliance.retrievers.vector_retrievers import VectorRetriever
from regulatory_compliance.retrievers.fts_retrievers import FTSRetriever
from regulatory_compliance.retrievers.hybrid_retrievers import HybridRetriever


class RAGAgent:
    """
    Regulatory Compliance RAG Agent.

    Responsibilities:
    - Classify user intent
    - Select retrieval strategy
    - Retrieve relevant documents
    - Generate answer using system/user prompts
    - Preserve citation metadata
    """

    def __init__(self):

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

        self.vector_retriever = VectorRetriever(top_k=5)

        self.fts_retriever = FTSRetriever(top_k=5)

        self.hybrid_retriever = HybridRetriever(top_k=5)

    # ==========================================================
    # STEP 1: INTENT + TOOL SELECTION
    # ==========================================================

    def classify_query(self, question: str) -> Dict:

        system_prompt = """
You are an intent classifier for a Regulatory Compliance RAG Assistant.

The assistant is designed ONLY for questions related to:
- RBI regulations
- SEBI regulations
- Basel regulations
- Banking compliance
- Gold loan regulations
- Lending regulations
- KYC
- AML
- Financial regulations
- Regulatory guidelines
- Regulatory circulars
- Compliance policies
- Uploaded regulatory documents

Classify the user's question into exactly one category:

1. CHITCHAT
   Casual conversation such as:
   - Hi
   - Hello
   - How are you?
   - Thank you
   - Who are you?

2. REGULATORY
   Questions related to regulatory compliance or uploaded
   regulatory documents.

3. OUT_OF_SCOPE
   General questions unrelated to regulatory compliance.

If the category is REGULATORY, select exactly one retrieval tool:

- fts_search
  Use for exact text, clause, section, ID, or keyword lookup.

- vector_search
  Use for conceptual questions, explanation, meaning, or semantic understanding.

- hybrid_search
  Use when both semantic and keyword retrieval are useful.

Return ONLY valid JSON.

Required JSON format:

{
    "query_type": "CHITCHAT | REGULATORY | OUT_OF_SCOPE",
    "tool_name": "fts_search | vector_search | hybrid_search | none"
}

Do not include markdown.
Do not include explanations.
"""

        user_prompt = f"""
User Question:

{question}
"""

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_completion_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = response.choices[0].message.content.strip()

        try:

            result = json.loads(content)

        except json.JSONDecodeError:

            # Safe fallback

            result = {
                "query_type": "REGULATORY",
                "tool_name": "hybrid_search",
            }

        return result

    # ==========================================================
    # STEP 2: RETRIEVAL
    # ==========================================================

    def retrieve_documents(
        self,
        question: str,
        tool_name: str,
    ) -> List[Document]:

        if tool_name == "vector_search":

            return self.vector_retriever.search(question)

        elif tool_name == "fts_search":

            return self.fts_retriever.search(question)

        else:

            return self.hybrid_retriever.search(question)

    # ==========================================================
    # STEP 3: BUILD CONTEXT
    # ==========================================================

    def build_context(
        self,
        documents: List[Document],
    ) -> str:

        context = ""

        for index, doc in enumerate(
            documents,
            start=1,
        ):

            metadata = doc.metadata

            context += f"""

--- RETRIEVED DOCUMENT {index} ---

Document ID:
{metadata.get("document_id")}

File Name:
{metadata.get("file_name")}

Page Number:
{metadata.get("page_number")}

Section Number:
{metadata.get("section_number")}

Regulation Type:
{metadata.get("regulation_type")}

Chunk Index:
{metadata.get("chunk_index")}

Retrieval Method:
{metadata.get("retrieval_method")}

Vector Score:
{metadata.get("vector_score")}

FTS Score:
{metadata.get("fts_score")}

Hybrid Score:
{metadata.get("hybrid_score")}

Content:
{doc.page_content}

"""

        return context

    # ==========================================================
    # STEP 4: GENERATE ANSWER
    # ==========================================================

    def generate_answer(
        self,
        question: str,
        documents: List[Document],
    ) -> str:

        context = self.build_context(documents)

        system_prompt = """
You are a Regulatory Compliance Assistant.

Your role is to answer questions using ONLY the retrieved
regulatory documents provided in the user message.

Rules:

1. Use only the retrieved context.
2. Do not hallucinate or invent regulatory requirements.
3. Do not use external knowledge.
4. If the answer cannot be found in the retrieved documents,
   say:

   "Information is not available in the provided documents."

5. Provide a concise and professional compliance-focused answer.
6. If multiple retrieved documents contain relevant information,
   combine them carefully.
7. Do not create a Sources section.
8. Do not invent page numbers, sections, document names, or citations.
9. Citation metadata is handled separately by the application.
10. If the retrieved context contains conflicting information,
    clearly mention the conflict.
11. Do not answer unrelated general knowledge questions.

Temperature:
0
"""

        user_prompt = f"""
User Question:

{question}


Retrieved Regulatory Context:

{context}


Generate the final answer using only the retrieved regulatory context.
"""

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_completion_tokens=2000,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content

    # ==========================================================
    # STEP 5: MAIN RAG PIPELINE
    # ==========================================================

    @traceable(name="rag_agent")
    def run(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
    ):

        start_time = time.time()

        if chat_history is None:

            chat_history = []

        # ------------------------------------------
        # Query Classification
        # ------------------------------------------

        classification = self.classify_query(question)

        query_type = classification.get(
            "query_type",
            "REGULATORY",
        )

        tool_name = classification.get(
            "tool_name",
            "hybrid_search",
        )

        # ------------------------------------------
        # Chitchat
        # ------------------------------------------

        if query_type == "CHITCHAT":

            return {
                "answer": (
                    "Hello! I am your Regulatory Compliance AI assistant. "
                    "Please ask questions related to the uploaded "
                    "regulatory documents."
                ),
                "query_type": "chitchat",
                "tool_used": None,
                "sources": [],
                "latency_ms": round(
                    (time.time() - start_time) * 1000,
                    2,
                ),
                "confidence": 1.0,
            }

        # ------------------------------------------
        # Out of Scope
        # ------------------------------------------

        if query_type == "OUT_OF_SCOPE":

            return {
                "answer": (
                    "I am a Regulatory Compliance AI assistant focused "
                    "on RBI, SEBI, Basel, and internal regulatory documents. "
                    "I cannot answer unrelated general questions."
                ),
                "query_type": "out_of_scope",
                "tool_used": None,
                "sources": [],
                "latency_ms": round(
                    (time.time() - start_time) * 1000,
                    2,
                ),
                "confidence": 1.0,
            }

        # ------------------------------------------
        # Regulatory Retrieval
        # ------------------------------------------

        documents = self.retrieve_documents(
            question,
            tool_name,
        )

        # ------------------------------------------
        # No Documents
        # ------------------------------------------

        if not documents:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the uploaded documents."
                ),
                "query_type": "rag",
                "tool_used": tool_name,
                "sources": [],
                "latency_ms": round(
                    (time.time() - start_time) * 1000,
                    2,
                ),
                "confidence": 0.2,
            }

        # ------------------------------------------
        # Generate Answer
        # ------------------------------------------

        answer = self.generate_answer(
            question,
            documents,
        )

        # ------------------------------------------
        # Citation Metadata
        # ------------------------------------------

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

        # ------------------------------------------
        # Final Response
        # ------------------------------------------

        return {
            "answer": answer,
            "query_type": "rag",
            "tool_used": tool_name,
            "sources": sources,
            "latency_ms": round(
                (time.time() - start_time) * 1000,
                2,
            ),
            "confidence": 0.85,
        }
