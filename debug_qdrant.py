import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# -------------------------
# CONFIG (EDIT IF NEEDED)
# -------------------------
COLLECTION_NAME = "feedbacklens"
HOST = "localhost"
PORT = 6333

TEST_QUERY = "swiggy delivery issue"


# -------------------------
# INIT
# -------------------------
client = QdrantClient(host=HOST, port=PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")


print("\n==============================")
print("🔍 STEP 1: COLLECTION CHECK")
print("==============================")

collections = client.get_collections().collections
print("Available collections:", [c.name for c in collections])

if COLLECTION_NAME not in [c.name for c in collections]:
    print(f"❌ Collection '{COLLECTION_NAME}' NOT FOUND")
    exit()
else:
    print(f"✅ Using collection: {COLLECTION_NAME}")


print("\n==============================")
print("📦 STEP 2: SAMPLE DATA CHECK")
print("==============================")

records, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5,
    with_payload=True
)

if not records:
    print("❌ No data found in collection")
    exit()

for r in records:
    print(r.payload)


print("\n==============================")
print("🏢 STEP 3: COMPANY VALUES CHECK")
print("==============================")

companies = set()
records, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=100,
    with_payload=True
)

for r in records:
    companies.add(r.payload.get("company"))

print("Companies found:", companies)


print("\n==============================")
print("🧠 STEP 4: EMBEDDING TEST")
print("==============================")

query_vector = model.encode(TEST_QUERY).tolist()
print(f"Query vector size: {len(query_vector)}")


print("\n==============================")
print("🚀 STEP 5: SEARCH WITHOUT FILTER")
print("==============================")

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5
)

print(f"Results count (no filter): {len(results)}")

for r in results:
    print(r.payload.get("company"), "|", r.payload.get("issue"))


print("\n==============================")
print("🎯 STEP 6: SEARCH WITH FILTER (swiggy)")
print("==============================")

filter_query = Filter(
    must=[
        FieldCondition(
            key="company",
            match=MatchValue(value="swiggy")
        )
    ]
)

results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    query_filter=filter_query,
    limit=5
)

print(f"Results count (with filter): {len(results)}")

for r in results:
    print(r.payload.get("company"), "|", r.payload.get("issue"))


print("\n==============================")
print("💀 FINAL DIAGNOSIS")
print("==============================")

if len(results) == 0:
    print("❌ FILTER BROKEN → company mismatch OR wrong data")
else:
    print("✅ FILTER WORKING")

print("\n🔥 DEBUG COMPLETE")