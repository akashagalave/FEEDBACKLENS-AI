from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from .config import settings
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger
from shared.schemas import RetrievedChunk

logger = get_logger("insight-agent")

_embedding_model = None


# ---------------------------
# Embedding Model Loader
# ---------------------------
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded!")
    return _embedding_model


# ---------------------------
# Qdrant Client (Cloud + Local support)
# ---------------------------
def get_qdrant_client() -> AsyncQdrantClient:
    host = settings.qdrant_host
    if host.startswith("http://") or host.startswith("https://"):
        return AsyncQdrantClient(
            url=host,
            api_key=settings.qdrant_api_key
        )
    return AsyncQdrantClient(
        host=host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None
    )


# ---------------------------
# Hybrid Search (FINAL)
# ---------------------------
async def hybrid_search(
    query: str,
    company: str,
    focus: str = None,
    top_k: int = 10
) -> list[RetrievedChunk]:

    logger.info(f"Hybrid search | query={query} | company={company} | focus={focus}")

    qdrant = get_qdrant_client()
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    qdrant_filter = None
    if company and company != "unknown":
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="company",
                    match=MatchValue(value=company.lower())
                )
            ]
        )

    # ---------------------------
    # VECTOR SEARCH
    # ---------------------------
    vector_results = await qdrant.search(
        collection_name=settings.collection_name,
        query_vector=query_embedding,
        query_filter=qdrant_filter,
        limit=top_k * 3
    )

    # FALLBACK: if filter fails
    if not vector_results:
        logger.warning("No results with filter → retrying without filter")
        vector_results = await qdrant.search(
            collection_name=settings.collection_name,
            query_vector=query_embedding,
            limit=top_k * 3
        )

    await qdrant.close()

    if not vector_results:
        logger.error("No results even after fallback")
        return []

    logger.info(f"Vector results count: {len(vector_results)}")

    # ---------------------------
    # BM25 RANKING
    # ---------------------------
    corpus = [r.payload.get("review", "") for r in vector_results]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

    # ---------------------------
    # HYBRID SCORING
    # ---------------------------
    vector_weight = 0.7
    bm25_weight = 0.3

    combined = []
    for i, result in enumerate(vector_results):
        vector_score = result.score or 0.0
        bm25_score = float(bm25_scores[i]) / max_bm25
        final_score = (vector_weight * vector_score) + (bm25_weight * bm25_score)
        combined.append((result, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_results = combined[:top_k]

    # ---------------------------
    # FORMAT OUTPUT
    # ---------------------------
    chunks = []
    for result, score in top_results:
        payload = result.payload or {}
        chunks.append(
            RetrievedChunk(
                review=payload.get("review", ""),
                company=payload.get("company", ""),
                domain=payload.get("domain", ""),
                issue=payload.get("issue", ""),
                score=round(score, 4)
            )
        )

    logger.info(f"Hybrid search returned {len(chunks)} chunks for {company}")
    return chunks