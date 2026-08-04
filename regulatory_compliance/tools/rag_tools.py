import json
from langchain_core.tools import tool
from regulatory_compliance.retrievers.vector_retrievers import VectorRetriever
from regulatory_compliance.retrievers.fts_retrievers import FTSRetriever
from regulatory_compliance.retrievers.hybrid_retrievers import HybridRetriever

vector_retriever = VectorRetriever(top_k=5)
fts_retriever = FTSRetriever(top_k=5)
hybrid_retriever = HybridRetriever(top_k=5)


def format_documents(documents):
    context = ""
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
        context += f"""
Document:
{metadata.get('file_name')}
Page:
{metadata.get('page_number')}
Content:
{doc.page_content}
"""
    return {"context": context, "sources": sources}


@tool
def vector_search_tool(query: str):
    """
     Use this tool for semantic understanding.
     Use when user asks:
     - explain
     - meaning
     - interpretation
     - why
     - how does regulation work
     - provide understanding of compliance concept
    This tool performs semantic similarity search.
    Do not use for exact document lookup.
    """
    docs = vector_retriever.search(query)
    result = format_documents(docs)
    return {
        "context": result["context"],
        "sources": result["sources"],
        "tool_used": "vector_search",
    }


@tool
def fts_search_tool(query: str):
    """
    Use this tool ONLY when the user query is a lookup request.
    Examples:
    - "KYC"
    - "BASEL III"
    - "section 4.2"
    - "RBI circular DBR..."
    - "Master Direction KYC"
    - "IRAC norms"
    The user expects exact matching content with 1 documet citations only.
    Do not use this tool for:
    - explain
    - describe
    - why
    - meaning
    """

    docs = fts_retriever.search(query)
    result = format_documents(docs)
    return {
        "context": result["context"],
        "sources": result["sources"],
        "tool_used": "fts_search",
    }


@tool
def hybrid_search_tool(query: str):
    """
    Use this tool for general regulatory questions.
    Combines:
    - keyword search
    - semantic vector search
    Use when user asks about:
    - general compliance questions
    - regulatory guidance
    - uploaded document questions
    This combines vector similarity and keyword search
    """

    docs = hybrid_retriever.search(query)
    result = format_documents(docs)
    return {
        "context": result["context"],
        "sources": result["sources"],
        "tool_used": "hybrid_search",
    }
