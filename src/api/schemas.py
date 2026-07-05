from pydantic import BaseModel
from typing import Any


class QueryRequest(BaseModel):
    filename: str
    model: str


class QueryResponse(BaseModel):
    input_filename: str
    output_filename: str
    detected: dict[str, Any] | list[dict[str, Any]]
