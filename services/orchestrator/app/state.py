from typing import Optional, TypedDict


class AgentState(TypedDict):
    query: str
    company: Optional[str]
    top_k: int
    intent: Optional[str]
    focus: Optional[str]
    retrieved_chunks: Optional[list]
    top_issues: Optional[list]
    patterns: Optional[list]
    recommendations: Optional[list]
    confidence_score: Optional[float]
    sample_reviews: Optional[list]
    error: Optional[str]