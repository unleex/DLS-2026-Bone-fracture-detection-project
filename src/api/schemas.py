from pydantic import BaseModel


class QueryRequest(BaseModel):
    filename: str
    model: str
    classes: list[int]


class QueryResponse(BaseModel):
    input_filename: str
    output_filename: str
    detected: list[list[str]]
