from langchain_core.messages import HumanMessage, AIMessage
import time
import json
from langsmith import traceable
from regulatory_compliance.services.llm_service import LLMService
from langchain.agents import create_agent
from regulatory_compliance.tools.rag_tools import (
    vector_search_tool,
    fts_search_tool,
    hybrid_search_tool,
)

llm = LLMService().get_llm()


rag_agent = create_agent(
    model=llm,
    tools=[vector_search_tool, fts_search_tool, hybrid_search_tool],
    system_prompt="""

You are a Regulatory Compliance AI Assistant.

MODE 1: Conversation

For:
- greetings
- user introduction
- remembering user's name
- thanks
- casual conversation

Do not call tools.

Use previous conversation history.
--------------------------------
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

Tool Rules:
You MUST call exactly ONE tool.

1. fts_search_tool:
    Use for:
- exact section
- exact keyword
- clause
- paragraph
- specific phrase

2. vector_search_tool:
    Use for:
- explain
- meaning
- interpretation
- concept

3. hybrid_search_tool:
    Use for all other regulatory questions.

After retrieving documents:
- Answer only from retrieved documents.
- Never hallucinate.
- Do not use previous answers as source.

If information is missing:
Reply:
"Information is not available in the provided documents."
---------------------------------
IMPORTANT:
Latest user question has highest priority.
Do not classify current question using previous conversation.
Use history only for:
- identity
- greetings
- casual references
---------------------------------
MODE 3: General knowledge
Do not call tools.
Reply:
"I am a Regulatory Compliance assistant and can help only with regulatory documents and compliance related questions."

Keep answers concise.
""",
)


def build_context(documents):
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
def run_agent(question: str, chat_history=None):
    start_time = time.time()
    messages = []
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # prevent duplicate question
    if not messages or messages[-1].content != question:
        messages.append(HumanMessage(content=question))

    response = rag_agent.invoke({"messages": messages})

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
        "query_type": "rag" if tool_used else "conversation",
        "tool_used": tool_used,
        "sources": sources,
        "latency_ms": round((time.time() - start_time) * 1000, 2),
        "confidence": 0.85,
    }
