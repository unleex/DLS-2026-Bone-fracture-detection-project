from pydantic import BaseModel


class QueryRequest(BaseModel):
    username: str
    filename: str


class QueryResponse(BaseModel):
    username: str
    input_filename: int
    output_filename: int
