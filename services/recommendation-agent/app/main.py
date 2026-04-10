from fastapi import FastAPI, HTTPException
from .schema import RecommendationRequest, RecommendationResponse
from .agent import generate_recommendations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("recommendation-agent")

app = FastAPI(title="FeedbackLens Recommendation Agent", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "recommendation-agent"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        result = await generate_recommendations(
            company=request.company,
            top_issues=request.top_issues,
            patterns=request.patterns
        )
        return RecommendationResponse(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))