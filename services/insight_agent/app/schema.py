from pydantic import BaseModel
from typing import Optional


class InsightRequest(BaseModel):
    query: str
    company: str
    focus: Optional[str] = None
    top_k: Optional[int] = 10


class InsightResponse(BaseModel):
    top_issues: list[str]
    patterns: list[str]
    sample_reviews: list[str]
    confidence_score: Optional[float] = None