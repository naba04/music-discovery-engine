import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="song_segments")

with open("embeddings.json") as f:
    data = json.load(f)

count = 0
for track in data:
    for i, emb in enumerate(track["segment_embeddings"]):
        collection.add(
            ids=[f"{track['track_id']}_seg{i}"],
            embeddings=[emb],
            metadatas=[{
                "track_id": track["track_id"],
                "segment_index": i,
                "filepath": track["filepath"]
            }]
        )
        count += 1

print(f"Loaded {count} segment embeddings into Chroma.")
print(f"Total items now in collection: {collection.count()}")
