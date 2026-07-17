from pathlib import Path
from fastapi import UploadFile
from regulatory_compliance.core.config import settings
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.utils.exceptions import InvalidFileException


class UploadService:
    """
    Handles PDF upload operations.
    """

    @staticmethod
    async def upload_pdf(file: UploadFile) -> ApiResponse:
        """
        Validate file format and save or uploaded PDF file to the file path.
        """

        # Validate uploaded file is PDF file
        if file.content_type != "application/pdf":
            raise InvalidFileException("Only PDF files are allowed.")

        # Create upload directory if it doesn't exist
        upload_path = Path(settings.UPLOAD_FOLDER)
        upload_path.mkdir(parents=True, exist_ok=True)
        file_path = upload_path / file.filename

        # Save uploaded file
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(await file.read())

        return ApiResponse(
            success=True,
            message="PDF uploaded successfully.",
            data={"file_name": file.filename, "file_path": str(file_path)},
        )
