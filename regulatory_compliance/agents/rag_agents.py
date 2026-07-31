import time
import json
from langsmith import traceable

from regulatory_compliance.services.llm_service import LLMService

from regulatory_compliance.tools.rag_tools import (
    vector_search_tool,
    fts_search_tool,
    hybrid_search_tool,
)

from langchain.agents import create_agent


class RAGAgent:

    def __init__(self):

        self.llm = LLMService().get_llm()

        self.agent = create_agent(
            model=self.llm,
            tools=[vector_search_tool, fts_search_tool, hybrid_search_tool],
            system_prompt="""

You are a Regulatory Compliance AI Assistant.


Your responsibility:

Answer questions related to:

- RBI regulations
- SEBI regulations
- Basel regulations
- Banking compliance
- Gold loan regulations
- AML
- KYC
- Uploaded regulatory documents
- Loans
- Gold Loans
- Fund regulations
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



Tool usage rules:

Tool Selection Rules:

You must select exactly ONE retrieval tool for regulatory questions.

Decision logic:

1. Use fts_search_tool ONLY when the user is asking for exact information.

Examples:
- "What is section 4.2?"
- "Find clause 8"
- "Show paragraph related to KYC"
- "Search keyword collateral"
- "Give exact wording"

Reason:
The user needs exact text matching.

-------------------------------------------------

2. Use vector_search_tool ONLY when the user wants conceptual understanding.

Examples:
- "Explain enhanced due diligence"
- "What does LTV mean?"
- "Why is KYC required?"
- "Explain gold loan default process"

Reason:
The user needs semantic interpretation.

-------------------------------------------------

3. Use hybrid_search_tool for all other regulatory questions.

Examples:
- "What are auction norms for gold loans?"
- "What are RBI requirements for KYC?"
- "Tell me lending guidelines"

Reason:
The user needs both keyword and semantic retrieval.

-------------------------------------------------

Do NOT always use hybrid_search_tool.
Choose based on the above rules.

4. For general questions:

Examples:
- Who is prime minister?
- Where is Mumbai?
- Places to visit

Do NOT call any tool.

Reply:

"I am a Regulatory Compliance assistant and can only answer questions related to regulatory documents."


# 5. For greetings and casual conversation:

#     Examples:
#     - hello
#     - hi
#     - good morning
#     - thank you
#     - how are you
    

#     DO NOT call any retrieval tool.
#     DO NOT search documents.

# 6. For normal chit chat discussions:

#     DO NOT call any retrieval tool.
#     DO NOT search documents.

# 7. For questions unrelated to regulatory compliance:

#     Answer directly using your own knowledge.

#     DO NOT call any retrieval tool.
#     DO NOT search documents.

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
    
8. Never create regulatory information.

9. If context does not contain answer:

Say:

"Information is not available in the provided documents."


Keep answers concise.

""",
        )

    def build_context(self, documents):
        """
        Build retrieved context for LangSmith evaluation.
        """

        context = ""

        for index, doc in enumerate(documents, start=1):

            metadata = doc.metadata

            context += f"""

    --- Document {index} ---

    Document ID:
    {metadata.get("document_id")}

    File Name:
    {metadata.get("file_name")}

    Page Number:
    {metadata.get("page_number")}

    Section:
    {metadata.get("section_number")}

    Regulation Type:
    {metadata.get("regulation_type")}

    Content:

    {doc.page_content}

    """

        return context

    @traceable(name="rag_agent")
    def run(self, question: str):

        start_time = time.time()

        response = self.agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question},
                ]
            }
        )

        tool_used = None
        sources = []
        context = ""

        for message in response["messages"]:
            if message.type == "tool":
                tool_response = json.loads(message.content)

                tool_used = tool_response.get("tool_used")
                sources = tool_response.get("sources", [])

        answer = response["messages"][-1].content

        return {
            "answer": answer,
            # For evaluation
            "inputs": {"question": question},
            "outputs": {"answer": answer},
            "context": context,
            "query_type": "rag",
            "tool_used": tool_used,
            "sources": sources,
            "latency_ms": round(
                (time.time() - start_time) * 1000,
                2,
            ),
            "confidence": 0.85,
        }
