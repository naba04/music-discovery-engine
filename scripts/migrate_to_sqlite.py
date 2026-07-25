import sqlite3
import json
import pandas as pd

DB_PATH = "music.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    title TEXT,
    artist TEXT,
    genre TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    query_track TEXT,
    result_track TEXT,
    net_votes INTEGER,
    PRIMARY KEY (query_track, result_track)
)
""")

print("Loading FMA metadata...")
metadata_df = pd.read_csv("data/fma_metadata/tracks.csv", index_col=0, header=[0, 1])

with open("embeddings.json") as f:
    all_tracks = json.load(f)

print(f"Migrating metadata for {len(all_tracks)} tracks...")
rows = []
for t in all_tracks:
    tid = t["track_id"]
    try:
        row = metadata_df.loc[int(tid)]
        title = row[("track", "title")]
        artist = row[("artist", "name")]
        genre = row[("track", "genre_top")]
    except (KeyError, ValueError):
        title, artist, genre = None, None, None
    rows.append((
        tid,
        title if isinstance(title, str) else "Unknown Title",
        artist if isinstance(artist, str) else "Unknown Artist",
        genre if isinstance(genre, str) else "Unknown Genre",
    ))

cur.executemany("INSERT OR REPLACE INTO tracks (track_id, title, artist, genre) VALUES (?, ?, ?, ?)", rows)

try:
    with open("feedback.json") as f:
        feedback_data = json.load(f)
    feedback_rows = []
    for key, net_votes in feedback_data.items():
        query_track, result_track = key.split("::")
        feedback_rows.append((query_track, result_track, net_votes))
    cur.executemany(
        "INSERT OR REPLACE INTO feedback (query_track, result_track, net_votes) VALUES (?, ?, ?)",
        feedback_rows
    )
    print(f"Migrated {len(feedback_rows)} feedback entries.")
except FileNotFoundError:
    print("No feedback.json found — skipping.")

conn.commit()
conn.close()
print(f"Done. Wrote {len(rows)} tracks to {DB_PATH}.")
