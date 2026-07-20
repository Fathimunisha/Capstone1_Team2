import logging
from typing import List

from langchain_core.documents import Document

from regulatory_compliance.core.db import Database
from regulatory_compliance.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Semantic search using pgvector cosine similarity.
    """

    def __init__(self, top_k: int = 8):

        self.top_k = top_k

        self.embedding_service = EmbeddingService()

    def search(self, query: str) -> List[Document]:

        try:

            # -----------------------------
            # Generate Query Embedding
            # -----------------------------

            query_embedding = self.embedding_service.generate_embedding(query)

            sql = """

            SELECT

                content,

                document_id,

                chunk_index,

                metadata,

                1 - (
                    embedding <=> %s::vector
                ) AS vector_score


            FROM document_chunks


            ORDER BY

                embedding <=> %s::vector


            LIMIT %s

            """

            documents = []

            with Database.get_connection() as conn:

                with conn.cursor() as cur:

                    cur.execute(sql, (query_embedding, query_embedding, self.top_k))

                    rows = cur.fetchall()

            for row in rows:

                content = row[0]

                document_id = row[1]

                chunk_index = row[2]

                stored_metadata = row[3] or {}

                vector_score = float(row[4])

                # Merge metadata safely

                metadata = {
                    **stored_metadata,
                    # DB fields
                    "document_id": str(document_id),
                    "chunk_index": chunk_index,
                    # Retrieval information
                    "vector_score": vector_score,
                    "retrieval_method": "vector_search",
                }

                documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception:

            logger.exception("Vector retrieval failed")

            raise
