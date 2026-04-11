import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "feedbacklens"

client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_qdrant():
    if QDRANT_HOST.startswith("http"):
        return QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)


def retrieve_contexts(query: str, company: str, top_k: int = 5) -> list[str]:
    qdrant = get_qdrant()
    embedding = embedding_model.encode(query).tolist()

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        query_filter=Filter(
            must=[FieldCondition(key="company", match=MatchValue(value=company))]
        ),
        limit=top_k
    )
    return [r.payload["review"] for r in results]

def generate_answer(query: str, contexts: list[str]) -> str:
    context_str = "\n".join(f"- {c}" for c in contexts)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a product analyst. Answer based on the reviews provided."},
            {"role": "user", "content": f"Query: {query}\n\nReviews:\n{context_str}"}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


EVAL_QUESTIONS = [
    {"query": "What are the main delivery issues with Swiggy?", "company": "swiggy"},
    {"query": "What do customers complain about Swiggy delivery time?", "company": "swiggy"},
    {"query": "What are Uber pricing problems?", "company": "uber"},
    {"query": "What issues do Uber customers face?", "company": "uber"},
    {"query": "What are common Swiggy customer support issues?", "company": "swiggy"},
]


def run_evaluation():
    logger.info("Starting RAGAS evaluation...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in EVAL_QUESTIONS:
        query = item["query"]
        company = item["company"]

        logger.info(f"Evaluating: {query}")

        retrieved = retrieve_contexts(query, company, top_k=5)
        answer = generate_answer(query, retrieved)

        questions.append(query)
        answers.append(answer)
        contexts.append(retrieved)
        ground_truths.append(retrieved[:2])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [g[0] if g else "" for g in ground_truths]
    })

    logger.info("Running RAGAS metrics...")

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    )

    results_dict = {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
        "context_recall": round(float(results["context_recall"]), 4),
    }

    logger.info("RAGAS Results:")
    for k, v in results_dict.items():
        logger.info(f"  {k}: {v}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/ragas_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info("Results saved to reports/ragas_results.json")
    return results_dict


if __name__ == "__main__":
    run_evaluation()