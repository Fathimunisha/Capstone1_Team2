import json
from typing import List
from langchain_core.documents import Document
from regulatory_compliance.core.db import Database


class FTSRetriever:
    """
    PostgreSQL Full Text Search Retriever.

    Uses:
    - PostgreSQL tsvector
    - ts_rank_cd
    """

    def __init__(self, top_k: int = 5):

        self.top_k = top_k

    def search(self, query: str) -> List[Document]:

        sql = """

        SELECT

            content,
            document_id,
            chunk_index,
            metadata,
            ts_rank_cd(
                tsv,
                plainto_tsquery(
                    'english',
                    %s
                )

            ) AS fts_score

        FROM document_chunks
        WHERE
            tsv @@ plainto_tsquery(
                'english',
                %s
            )
        ORDER BY
            fts_score DESC
        LIMIT %s

        """

        documents = []

        with Database.get_connection() as conn:

            with conn.cursor() as cur:

                cur.execute(sql, (query, query, self.top_k))

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
                stored_metadata.get("page_number") or stored_metadata.get("page", 0) + 1
            )

            stored_metadata["section_number"] = (
                stored_metadata.get("section_number") or "N/A"
            )
            fts_score = float(row[4])
            metadata = {
                # Preserve ingestion metadata
                **stored_metadata,
                "document_id": str(document_id),
                "chunk_index": chunk_index,
                "fts_score": fts_score,
                "retrieval_method": "fts_search",
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents
