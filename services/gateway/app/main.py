import httpx
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from .schema import QueryRequest, BatchRequest
from .config import settings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("gateway")

app = FastAPI(title="FeedbackLens Gateway", version="1.0.0")

# ─── METRICS ─────────────────────────────────────────────────
REQUEST_COUNT = Counter("gateway_requests_total", "Total requests", ["endpoint"])
REQUEST_LATENCY = Histogram("gateway_request_latency_seconds", "Request latency", ["endpoint"])

# Mount prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── HEALTH ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}


# ─── QUERY ENDPOINT ──────────────────────────────────────────
@app.post("/analyze")
async def analyze(request: QueryRequest):
    REQUEST_COUNT.labels(endpoint="analyze").inc()

    with REQUEST_LATENCY.labels(endpoint="analyze").time():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.orchestrator_url}/run",
                    json=request.model_dump()
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Orchestrator error: {e}")
            raise HTTPException(status_code=502, detail="Orchestrator unreachable")


# ─── BATCH ENDPOINT ──────────────────────────────────────────
@app.post("/batch")
async def batch(request: BatchRequest):
    REQUEST_COUNT.labels(endpoint="batch").inc()

    with REQUEST_LATENCY.labels(endpoint="batch").time():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{settings.orchestrator_url}/batch",
                    json=request.model_dump()
                )
                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Orchestrator error: {e}")
            raise HTTPException(status_code=502, detail="Orchestrator unreachable")