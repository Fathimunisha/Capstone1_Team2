import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from regulatory_compliance.core.config import settings


class PDFIngestion:
    """
    Handles PDF loading, cleaning and chunking.
    Adds citation metadata.
    """

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def ingest(self, pdf_path: str) -> list[Document]:
        """
        Complete ingestion pipeline.
        """

        path = Path(pdf_path)

        if not path.exists():

            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # ----------------------------
        # Load PDF
        # ----------------------------

        loader = PyPDFLoader(str(path))

        documents = loader.load()

        # ----------------------------
        # Clean text
        # ----------------------------

        cleaned_documents = self.clean_documents(documents)

        # ----------------------------
        # Add metadata
        # ----------------------------

        enriched_documents = self.add_metadata(cleaned_documents, path.name)

        # ----------------------------
        # Chunking
        # ----------------------------

        chunks = self.split_documents(enriched_documents)

        return chunks

    def clean_documents(self, documents: list[Document]) -> list[Document]:

        cleaned = []

        for document in documents:

            text = document.page_content

            # Keep formatting because
            # regulatory documents need sections

            text = text.strip()

            document.page_content = text

            cleaned.append(document)

        return cleaned

    def add_metadata(self, documents: list[Document], file_name: str) -> list[Document]:
        """
        Add metadata required for citations.
        """

        for document in documents:

            page_number = document.metadata.get("page", 0)

            document.metadata.update(
                {
                    # file information
                    "file_name": file_name,
                    # PyPDF gives zero based page
                    # convert to human page
                    "page_number": page_number + 1,
                    # future enhancement
                    # section extraction
                    "section_number": self.extract_section(document.page_content),
                    # document category
                    "regulation_type": "RBI",
                }
            )

        return documents

    def extract_section(self, text: str):
        """
        Basic section detection.

        Example:
        3.1 Asset Classification
        Section 4:
        """

        patterns = [
            r"(\d+\.\d+\s+[A-Za-z ].+)",
            r"(Section\s+\d+[:\-]?\s*[A-Za-z ]+)",
        ]

        for pattern in patterns:

            match = re.search(pattern, text, re.IGNORECASE)

            if match:

                return match.group(1).strip()

        return None

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split while preserving metadata.
        """

        chunks = self.splitter.split_documents(documents)

        # Add chunk index

        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_index"] = index

        return chunks
