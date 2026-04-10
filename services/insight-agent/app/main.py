from fastapi import FastAPI, HTTPException
from .schema import InsightRequest, InsightResponse
from .agent import generate_insights
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("insight-agent")

app = FastAPI(title="FeedbackLens Insight Agent", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "insight-agent"}


@app.post("/insights", response_model=InsightResponse)
async def insights(request: InsightRequest):
    try:
        result = await generate_insights(
            query=request.query,
            company=request.company,
            focus=request.focus,
            top_k=request.top_k
        )
        return InsightResponse(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))