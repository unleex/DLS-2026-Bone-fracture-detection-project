from pydantic import BaseModel


class QueryRequest(BaseModel):
    filename: str
    model: str


class QueryResponse(BaseModel):
    input_filename: int
    output_filename: int
