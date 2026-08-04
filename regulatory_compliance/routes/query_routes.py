from fastapi import APIRouter
from regulatory_compliance.models.request import QueryRequest
from regulatory_compliance.models.response import QueryResponse
from regulatory_compliance.services.query_service import QueryService

router = APIRouter(
    prefix="/api/v1",
    tags=["Query"],
)

service = QueryService()


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    result = service.process_query(
        request.question,
        request.chat_history,
    )
    return result
