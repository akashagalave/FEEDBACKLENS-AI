from qdrant_client import QdrantClient
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

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded!")
    return _embedding_model


async def hybrid_search(
    query: str,
    company: str,
    focus: str = None,
    top_k: int = 10
) -> list[RetrievedChunk]:

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    model = get_embedding_model()

    query_embedding = model.encode(query).tolist()

    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="company",
                match=MatchValue(value=company.lower())
            )
        ]
    )

    if focus:
        qdrant_filter.must.append(
            FieldCondition(
                key="issue",
                match=MatchValue(value=focus.lower())
            )
        )

    vector_results = qdrant.search(
        collection_name="feedbacklens",
        query_vector=query_embedding,
        query_filter=qdrant_filter,
        limit=top_k * 2
    )

    if not vector_results:
        logger.warning(f"No vector results for company={company}")
        return []

    corpus = [r.payload["review"] for r in vector_results]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    vector_weight = 0.6
    bm25_weight = 0.4

    combined = []
    for i, result in enumerate(vector_results):
        vector_score = result.score
        bm25_score = float(bm25_scores[i])
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = bm25_score / max_bm25
        final_score = (vector_weight * vector_score) + (bm25_weight * bm25_normalized)
        combined.append((result, final_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top_results = combined[:top_k]

    chunks = []
    for result, score in top_results:
        chunks.append(RetrievedChunk(
            review=result.payload["review"],
            company=result.payload["company"],
            domain=result.payload["domain"],
            issue=result.payload["issue"],
            score=round(score, 4)
        ))

    logger.info(f"Hybrid search returned {len(chunks)} chunks for {company}")
    return chunks