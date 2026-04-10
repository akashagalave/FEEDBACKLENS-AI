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


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_client(params: dict) -> QdrantClient:
    return QdrantClient(
        host=params["qdrant"]["host"],
        port=params["qdrant"]["port"]
    )


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


def index_data(client: QdrantClient, collection_name: str, df: pd.DataFrame, embeddings: np.ndarray):
    points = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Indexing")):
        point = PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "company": row["company"],
                "domain": row["domain"],
                "issue": row["issue"],
                "rating": float(row["rating"]) if "rating" in row else None,
                "review": row["review"],
                "rag_text": row["rag_text"]
            }
        )
        points.append(point)

    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    logger.info(f"Indexed {len(points)} points into '{collection_name}'")


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

    logger.info("Indexing complete!")


if __name__ == "__main__":
    main()