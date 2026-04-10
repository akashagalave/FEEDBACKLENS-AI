import pandas as pd
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from loguru import logger


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_embeddings(texts: list, model_name: str, batch_size: int) -> np.ndarray:
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(texts)} texts")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def main():
    params = load_params()

    model_name = params["embedding"]["model"]
    batch_size = params["embedding"]["batch_size"]

    logger.info("Loading cleaned data")
    df = pd.read_csv("data/cleaned.csv")

    embeddings = generate_embeddings(
        df["rag_text"].tolist(),
        model_name,
        batch_size
    )

    output_path = "data/embeddings.npy"
    np.save(output_path, embeddings)
    logger.info(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()