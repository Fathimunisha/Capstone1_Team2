from typing import List, Dict

from langchain_core.documents import Document

from regulatory_compliance.retrievers.vector_retrievers import VectorRetriever
from regulatory_compliance.retrievers.fts_retrievers import FTSRetriever


class HybridRetriever:
    """
    Hybrid Retrieval using:

    1. Vector similarity search
    2. PostgreSQL Full Text Search

    Ranking:
    Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, top_k: int = 8, rrf_constant: int = 60):

        self.top_k = top_k

        self.rrf_constant = rrf_constant

        self.vector_retriever = VectorRetriever(top_k=top_k)

        self.fts_retriever = FTSRetriever(top_k=top_k)

    def search(self, query: str) -> List[Document]:

        vector_docs = self.vector_retriever.search(query)

        fts_docs = self.fts_retriever.search(query)

        return self.rrf_merge(vector_docs, fts_docs)

    def rrf_merge(
        self, vector_results: List[Document], fts_results: List[Document]
    ) -> List[Document]:

        scores: Dict[str, float] = {}

        documents: Dict[str, Document] = {}

        # ----------------------------------
        # Vector Search Ranking
        # ----------------------------------

        for rank, doc in enumerate(vector_results, start=1):

            key = self.get_key(doc)

            scores[key] = scores.get(key, 0) + (1 / (self.rrf_constant + rank))

            documents[key] = doc

        # ----------------------------------
        # FTS Ranking
        # ----------------------------------

        for rank, doc in enumerate(fts_results, start=1):

            key = self.get_key(doc)

            scores[key] = scores.get(key, 0) + (1 / (self.rrf_constant + rank))

            if key in documents:

                # Merge metadata

                documents[key].metadata.update(doc.metadata)

            else:

                documents[key] = doc

        # ----------------------------------
        # Final Ranking
        # ----------------------------------

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_documents = []

        for key, score in ranked:

            doc = documents[key]

            doc.metadata.update(
                {
                    "hybrid_score": round(score, 6),
                    "retrieval_method": "hybrid_search",
                    "retrieval_details": {
                        "vector_score": doc.metadata.get("vector_score"),
                        "fts_score": doc.metadata.get("fts_score"),
                    },
                }
            )

            final_documents.append(doc)

        return final_documents[: self.top_k]

    def get_key(self, doc: Document):

        return (
            f"{doc.metadata.get('document_id')}_" f"{doc.metadata.get('chunk_index')}"
        )
