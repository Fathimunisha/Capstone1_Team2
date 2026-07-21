import json
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

    def __init__(self, top_k: int = 5):

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
                if isinstance(stored_metadata, str):
                    stored_metadata = json.loads(stored_metadata)

                stored_metadata["file_name"] = (
                    stored_metadata.get("file_name")
                    or stored_metadata.get("source")
                    or "Unknown"
                )

                stored_metadata["page_number"] = (
                    stored_metadata.get("page_number")
                    or stored_metadata.get("page", 0) + 1
                )

                stored_metadata["section_number"] = (
                    stored_metadata.get("section_number") or "N/A"
                )

                vector_score = float(row[4])

                # Merge metadata safely

                metadata = {
                    **stored_metadata,
                    "document_id": str(document_id),
                    "chunk_index": chunk_index,
                    "vector_score": vector_score,
                    "retrieval_method": "vector_search",
                }

                documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception:

            logger.exception("Vector retrieval failed")

            raise
