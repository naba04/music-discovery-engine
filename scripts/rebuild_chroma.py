import json
import chromadb

with open("embeddings.json") as f:
    all_tracks = json.load(f)

client = chromadb.PersistentClient(path="./chroma_db")

# Drop and recreate so old partial data doesn't linger
try:
    client.delete_collection(name="song_segments")
except Exception:
    pass
collection = client.get_or_create_collection(name="song_segments")

ids, embeddings, metadatas = [], [], []
for track in all_tracks:
    tid = track["track_id"]
    for i, emb in enumerate(track["segment_embeddings"]):
        ids.append(f"{tid}_{i}")
        embeddings.append(emb)
        metadatas.append({"track_id": tid, "segment_index": i})

print(f"Adding {len(ids)} segments from {len(all_tracks)} tracks to Chroma...")

# Chroma has a max batch size, so add in chunks
CHUNK = 5000
for start in range(0, len(ids), CHUNK):
    end = start + CHUNK
    collection.add(
        ids=ids[start:end],
        embeddings=embeddings[start:end],
        metadatas=metadatas[start:end],
    )
    print(f"  added {min(end, len(ids))}/{len(ids)}")

print("Done rebuilding chroma_db.")
