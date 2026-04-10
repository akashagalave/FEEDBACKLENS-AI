from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .graph import workflow
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("orchestrator")

app = FastAPI(title="FeedbackLens Orchestrator", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    top_k: Optional[int] = 10


@app.get("/health")
async def health():
    return {"status": "ok", "service": "orchestrator"}


@app.post("/run")
async def run(request: QueryRequest):
    try:
        initial_state = {
            "query": request.query,
            "company": request.company,
            "top_k": request.top_k,
            "intent": None,
            "focus": None,
            "retrieved_chunks": None,
            "top_issues": None,
            "patterns": None,
            "recommendations": None,
            "confidence_score": None,
            "sample_reviews": None,
            "error": None
        }

        result = await workflow.ainvoke(initial_state)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        return {
            "company": result.get("company"),
            "top_issues": result.get("top_issues", []),
            "patterns": result.get("patterns", []),
            "recommendations": result.get("recommendations", []),
            "confidence_score": result.get("confidence_score"),
            "sample_reviews": result.get("sample_reviews", [])
        }

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))