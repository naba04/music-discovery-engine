import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="song_segments")

with open("embeddings.json") as f:
    data = json.load(f)

# Grab the first segment embedding of the first track as a test query
test_track = data[0]
test_embedding = test_track["segment_embeddings"][0]

results = collection.query(
    query_embeddings=[test_embedding],
    n_results=5
)

print(f"Query track: {test_track['track_id']}")
print("Top 5 nearest segments found:")
for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
    print(f"  track={meta['track_id']}  segment={meta['segment_index']}  distance={distance:.4f}")
