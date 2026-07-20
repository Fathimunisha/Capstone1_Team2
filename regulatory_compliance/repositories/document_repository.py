import json
import logging
import uuid
from typing import List
from langchain_core.documents import Document
from regulatory_compliance.core.db import Database

# logger
logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository class responsible for all database operations
    related to regulatory documents.
    """

    def insert_document(
        self,
        document_name: str,
        regulation_type: str,
        source_date,
        version: str,
        total_pages: int,
        total_chunks: int,
    ) -> str:
        document_id = str(uuid.uuid4())

        sql = """
        INSERT INTO documents
        (
            id,
            document_name,
            regulation_type,
            source_date,
            version,
            total_pages,
            total_chunks
        )

        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s
        )
        """

        try:

            with Database.get_connection() as conn:

                with conn.cursor() as cur:

                    cur.execute(
                        sql,
                        (
                            document_id,
                            document_name,
                            regulation_type,
                            source_date,
                            version,
                            total_pages,
                            total_chunks,
                        ),
                    )

                conn.commit()

            logger.info("Document inserted successfully.")

            return document_id

        except Exception as ex:

            logger.exception("Failed to insert document.")

            raise ex

    def insert_chunks_with_embeddings(
        self,
        document_id: str,
        chunks: List[Document],
        embeddings: list[list[float]],
    ):
        sql = """
        INSERT INTO document_chunks
        (
            id,
            document_id,
            chunk_index,
            page_number,
            section_number,                       
            content,
            embedding,
            tsv,
            metadata
        )

        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s,to_tsvector('english', %s),%s
        )
        """

        try:

            with Database.get_connection() as conn:

                with conn.cursor() as cur:

                    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

                        chunk_id = str(uuid.uuid4())

                        metadata = json.dumps(chunk.metadata)

                        page_number = chunk.metadata.get("page", 0)

                        section_number = chunk.metadata.get("section", None)

                        cur.execute(
                            sql,
                            (
                                chunk_id,
                                document_id,
                                index,
                                page_number,
                                section_number,
                                chunk.page_content,
                                embedding,
                                chunk.page_content,
                                metadata,
                            ),
                        )

                conn.commit()

            logger.info(
                "%s chunks stored successfully.",
                len(chunks),
            )

        except Exception as ex:

            logger.exception("Chunk storage failed.")

            raise
