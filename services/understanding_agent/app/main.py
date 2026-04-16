from fastapi import FastAPI, HTTPException
from .schema import UnderstandRequest, UnderstandResponse
from .agent import understand_query
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("understanding-agent")

app = FastAPI(title="FeedbackLens Understanding Agent", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "understanding-agent"}


@app.post("/understand", response_model=UnderstandResponse)
async def understand(request: UnderstandRequest):
    try:
        result = await understand_query(
            query=request.query,
            company=request.company
        )
        return UnderstandResponse(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))