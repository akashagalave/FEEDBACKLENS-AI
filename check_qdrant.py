from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host='qdrant', port=6333)

results = client.search(
    collection_name='feedbacklens',
    query_vector=[0.1]*384,
    query_filter=Filter(
        must=[FieldCondition(key='company', match=MatchValue(value='swiggy'))]
    ),
    limit=3
)
print('Results:', len(results))
for r in results:
    print(r.payload['company'], r.payload['issue'])