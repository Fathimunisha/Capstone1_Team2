from fastapi import APIRouter, File, UploadFile, status
from regulatory_compliance.models.response import ApiResponse
from regulatory_compliance.services.upload_service import UploadService

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("/pdf", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a regulatory compliance PDF.
    """

    response = await UploadService.upload_pdf(file)
    return response
