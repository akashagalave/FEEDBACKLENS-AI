from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    top_k: Optional[int] = 10


class BatchRequest(BaseModel):
    company: str
    reviews: list[str]