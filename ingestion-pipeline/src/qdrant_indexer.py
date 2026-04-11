import pandas as pd
import numpy as np
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType
)
from loguru import logger
from tqdm import tqdm


# ---------------------------
# Load params
# ---------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Qdrant Client
# ---------------------------
# def get_client(params: dict) -> QdrantClient:
#     return QdrantClient(
#         host=params["qdrant"]["host"],
#         port=params["qdrant"]["port"]
#     )
def get_client(params: dict) -> QdrantClient:
    return QdrantClient(
        host=params["qdrant"]["host"],
        port=params["qdrant"]["port"],
        api_key=params["qdrant"].get("api_key")
    )


# ---------------------------
# Create Collection
# ---------------------------
def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        logger.warning(f"Collection '{collection_name}' already exists. Deleting and recreating.")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    logger.info(f"Created collection: {collection_name}")

    # 🔥 Payload indexes for fast filtering
    client.create_payload_index(
        collection_name=collection_name,
        field_name="company",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="domain",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="issue",
        field_schema=PayloadSchemaType.KEYWORD
    )

    logger.info("Payload indexes created (company, domain, issue)")


# ---------------------------
# Normalize helper
# ---------------------------
def clean_text(val):
    if pd.isna(val):
        return ""
    return str(val).lower().strip()


# ---------------------------
# Index Data
# ---------------------------
def index_data(client: QdrantClient, collection_name: str, df: pd.DataFrame, embeddings: np.ndarray):

    # ✅ Safety check
    assert len(df) == len(embeddings), "Mismatch between data and embeddings"

    # 🔍 Debug sample
    logger.info("Sample data preview:")
    logger.info(df.head(3)[["company", "domain", "issue"]])

    points = []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Indexing")):

        point = PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                # ✅ NORMALIZED FIELDS (CRITICAL FIX)
                "company": clean_text(row.get("company")),
                "domain": clean_text(row.get("domain")),
                "issue": clean_text(row.get("issue")),

                # Optional fields
                "rating": float(row["rating"]) if "rating" in row and pd.notna(row["rating"]) else None,

                # Keep raw text readable
                "review": str(row.get("review", "")).strip(),
                "rag_text": str(row.get("rag_text", "")).strip()
            }
        )

        points.append(point)

    # 🚀 Batch upload
    batch_size = 100

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    logger.info(f"Indexed {len(points)} points into '{collection_name}'")


# ---------------------------
# Main
# ---------------------------
def main():
    params = load_params()
    collection_name = params["qdrant"]["collection_name"]

    logger.info("Loading cleaned data and embeddings")

    df = pd.read_csv("data/cleaned.csv")
    embeddings = np.load("data/embeddings.npy")

    vector_size = embeddings.shape[1]
    logger.info(f"Vector size: {vector_size}")

    client = get_client(params)

    create_collection(client, collection_name, vector_size)
    index_data(client, collection_name, df, embeddings)

    logger.info("✅ Indexing complete! Ready for hybrid search 🚀")


# ---------------------------
if __name__ == "__main__":
    main()