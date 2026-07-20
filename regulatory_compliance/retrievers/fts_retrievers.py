from typing import List

from langchain_core.documents import Document

from regulatory_compliance.core.db import Database


class FTSRetriever:
    """
    PostgreSQL Full Text Search Retriever.

    Uses:
    - PostgreSQL tsvector
    - ts_rank_cd
    - GIN index
    """

    def __init__(self, top_k: int = 8):

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
                    'simple',
                    %s
                )

            ) AS fts_score


        FROM document_chunks



        WHERE


            tsv @@ plainto_tsquery(

                'simple',

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

            fts_score = float(row[4])

            metadata = {
                # Preserve ingestion metadata
                **stored_metadata,
                # Database metadata
                "document_id": str(document_id),
                "chunk_index": chunk_index,
                # Retrieval metadata
                "fts_score": fts_score,
                "retrieval_method": "fts_search",
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents
