from langchain_core.messages import HumanMessage, AIMessage
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
        
You have two modes:

MODE 1: Conversation

For:
- greetings
- user introduction
- remembering user's name
- thanks
- casual conversation

Do not call any tool.

Use previous conversation history.

---------------------------------

MODE 2: Regulatory Document Question

For questions related to:

- RBI
- SEBI
- Basel
- AML
- KYC
- Gold Loan
- Banking compliance
- Lending regulations
- Uploaded documents

You MUST call exactly ONE tool.

Tool selection:

fts_search_tool:
Use when query contains:
- exact section
- clause
- paragraph
- keyword lookup
- specific phrase


vector_search_tool:
Use when user asks:
- explain
- meaning
- interpretation
- concept

hybrid_search_tool:
Use for all other regulatory questions.

After retrieving documents:

- Answer only from retrieved documents.
- Never use previous answers as regulatory source.
- Never hallucinate.
- If information missing:

"Information is not available in the provided documents."

---------------------------------

IMPORTANT CURRENT QUERY RULE:

The latest user message always has highest priority.

Do not classify the current question based on previous conversation.

Use chat history only for:
- user identity
- greetings
- casual references like "what did I say?"

MODE 3: General knowledge

Do not call tools.

Reply:

"I am a Regulatory Compliance assistant and can help only with regulatory documents and compliance related questions."

---------------------------------

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
    def run(self, question: str, chat_history=None):

        start_time = time.time()

        messages = []

        if chat_history:

            for msg in chat_history:

                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))

                elif msg["role"] == "assistant":

                    messages.append(AIMessage(content=msg["content"]))

        # avoid duplicate current question
        if not messages or messages[-1].content != question:

            messages.append(HumanMessage(content=question))

        response = self.agent.invoke({"messages": messages})

        tool_used = None
        sources = []
        context = ""

        for message in response["messages"]:

            if message.type == "tool":

                try:

                    tool_response = json.loads(message.content)

                    tool_used = tool_response.get("tool_used")

                    sources = tool_response.get("sources", [])

                    context += tool_response.get("context", "")

                except Exception:

                    context += str(message.content)

        answer = response["messages"][-1].content

        return {
            "answer": answer,
            "inputs": {"question": question},
            "outputs": {"answer": answer},
            "context": context,
            "query_type": ("rag" if tool_used else "conversation"),
            "tool_used": tool_used,
            "sources": sources,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "confidence": 0.85,
        }
