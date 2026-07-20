from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel


class ApiResponse(BaseModel):

    message: str

    success: bool = True


class Source(BaseModel):

    document_id: Optional[str] = None

    chunk_index: Optional[int] = None

    hybrid_score: Optional[float] = None


class QueryResponse(BaseModel):

    answer: str

    sources: List[Source]
