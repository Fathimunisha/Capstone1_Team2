from typing import Any
from pydantic import BaseModel, Field


class ApiResponse(BaseModel):
    """
    Standard API Response Model
    """

    success: bool = Field(
        ..., description="Indicates whether the request was successful"
    )

    message: str = Field(..., description="Response message")

    data: Any | None = Field(default=None, description="Response payload")
