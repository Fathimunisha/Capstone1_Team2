from pathlib import Path
from fastapi import UploadFile
from regulatory_compliance.core.config import settings
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.utils.exceptions import InvalidFileException
from regulatory_compliance.ingestion.ingestion import PDFIngestion
from regulatory_compliance.services.embedding_service import EmbeddingService
from regulatory_compliance.repositories.document_repository import DocumentRepository


class UploadService:
    """
    Handles PDF upload and ingestion.
    """

    @staticmethod
    async def upload_pdf(file: UploadFile) -> ApiResponse:
        if file.content_type != "application/pdf":
            raise InvalidFileException("Only PDF files are allowed.")
        # -------------------------
        # Save PDF
        # -------------------------
        upload_path = Path(settings.UPLOAD_FOLDER)
        upload_path.mkdir(parents=True, exist_ok=True)
        file_path = upload_path / file.filename
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(await file.read())
        print("PDF saved:", file_path)

        # -------------------------
        # Ingestion
        # -------------------------
        ingestion = PDFIngestion()
        chunks = ingestion.ingest(str(file_path))
        print("Chunks created:", len(chunks))
        if not chunks:
            raise Exception("No chunks generated from PDF")

        # -------------------------
        # Generate embeddings
        # -------------------------
        embedding_service = EmbeddingService()
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(texts)
        print("Embeddings generated:", len(embeddings))

        # -------------------------
        # Store in DB
        # -------------------------
        repository = DocumentRepository()
        document_id = repository.insert_document(
            document_name=file.filename,
            regulation_type="Internal",
            source_date=None,
            version="1.0",
            total_pages=len(chunks),
            total_chunks=len(chunks),
        )
        repository.insert_chunks_with_embeddings(document_id, chunks, embeddings)
        print("Inserted document:", document_id)
        return ApiResponse(
            success=True,
            message="PDF uploaded and indexed successfully.",
            data={
                "file_name": file.filename,
                "chunks": len(chunks),
                "document_id": document_id,
            },
        )
