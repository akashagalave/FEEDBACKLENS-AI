from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    company: str
    top_issues: list[str]
    patterns: list[str]


class RecommendationResponse(BaseModel):
    recommendations: list[str]