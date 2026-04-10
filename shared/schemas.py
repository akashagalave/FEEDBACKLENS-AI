from pydantic import BaseModel
from typing import Optional


# ─── INPUT SCHEMAS ───────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    top_k: Optional[int] = 10


class BatchRequest(BaseModel):
    company: str
    reviews: list[str]


# ─── AGENT SCHEMAS ───────────────────────────────────────────

class UnderstandingOutput(BaseModel):
    company: str
    intent: str
    focus: Optional[str] = None


class RetrievedChunk(BaseModel):
    review: str
    company: str
    domain: str
    issue: str
    score: float


class InsightOutput(BaseModel):
    top_issues: list[str]
    patterns: list[str]
    sample_reviews: list[str]
    confidence_score: Optional[float] = None


class RecommendationOutput(BaseModel):
    recommendations: list[str]


# ─── FINAL OUTPUT SCHEMAS ────────────────────────────────────

class QueryResponse(BaseModel):
    company: str
    top_issues: list[str]
    patterns: list[str]
    recommendations: list[str]
    confidence_score: Optional[float] = None
    sample_reviews: Optional[list[str]] = None


class BatchResponse(BaseModel):
    company: str
    summary: dict
    patterns: list[str]
    recommendations: list[str]