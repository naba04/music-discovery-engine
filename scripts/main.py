from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import chromadb
import numpy as np
import json
import sqlite3
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from scripts.llm_query import parse_query, explain_match
from scripts.track_lookup_helpers import find_track_id_by_title
from scripts.semantic_search import search_by_text
from scripts.feedback_store import record_vote, get_adjustment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
if os.path.isdir("data/fma_small"):
    app.mount("/audio", StaticFiles(directory="data/fma_small"), name="audio")

client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="song_segments")

with open("embeddings.json") as f:
    all_tracks = json.load(f)
track_lookup = {t["track_id"]: t for t in all_tracks}

with open("energy_curves.json") as f:
    curve_lookup = json.load(f)

db_conn = sqlite3.connect("music.db", check_same_thread=False)


def get_track_info(track_id: str):
    row = db_conn.execute(
        "SELECT title, artist, genre FROM tracks WHERE track_id = ?", (track_id,)
    ).fetchone()
    if row is None:
        return {"title": "Unknown Title", "artist": "Unknown Artist", "genre": "Unknown Genre"}
    return {"title": row[0], "artist": row[1], "genre": row[2]}


def distance_to_score(distance: float) -> int:
    score = max(0, 100 - (distance * 50))
    return round(score)


def scalar_euclidean(a, b):
    """Curves are stored as flat lists of plain numbers, so each point going into
    fastdtw is a lone float. euclidean() requires 1-D arrays, not bare scalars,
    so wrap each point in a list before comparing."""
    return euclidean([a], [b])


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Music Discovery API is running",
        "tracks_loaded": len(track_lookup),
        "trajectories_loaded": len(curve_lookup)
    }


@app.get("/similar/{track_id}")
def find_similar(track_id: str, top_k: int = 5, genre: str | None = None):
    track = track_lookup.get(track_id)
    if not track:
        return {"error": f"track_id '{track_id}' not found. Try one of: {list(track_lookup.keys())[:5]}"}

    embeddings = np.array(track["segment_embeddings"])
    query_vector = embeddings.mean(axis=0).tolist()

    results = collection.query(query_embeddings=[query_vector], n_results=200)

    scored = {}
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        tid = meta["track_id"]
        if tid == track_id:
            continue
        if tid not in scored or distance < scored[tid]:
            scored[tid] = distance

    adjusted = []
    for tid, dist in scored.items():
        base_score = distance_to_score(dist)
        match_score = max(0, min(100, base_score + get_adjustment(track_id, tid)))
        adjusted.append((tid, dist, match_score))
    adjusted.sort(key=lambda x: x[2], reverse=True)

    enriched_results = []
    for tid, dist, match_score in adjusted:
        info = get_track_info(tid)
        if genre and info["genre"].lower() != genre.lower():
            continue
        enriched_results.append({
            "track_id": tid,
            "distance": round(dist, 4),
            "match_score": match_score,
            "title": info["title"],
            "artist": info["artist"],
            "genre": info["genre"]
        })
        if len(enriched_results) >= top_k:
            break

    query_info = get_track_info(track_id)
    return {
        "query_track": track_id,
        "query_title": query_info["title"],
        "query_artist": query_info["artist"],
        "results": enriched_results
    }


@app.get("/similar_progression/{track_id}")
def find_similar_progression(track_id: str, top_k: int = 5, genre: str | None = None):
    query_curve = curve_lookup.get(track_id)
    if query_curve is None:
        return {"error": f"No energy curve for track_id '{track_id}'. Try one of: {list(curve_lookup.keys())[:5]}"}

    scored = []
    for tid, candidate_curve in curve_lookup.items():
        if tid == track_id:
            continue
        distance, _ = fastdtw(query_curve, candidate_curve, dist=scalar_euclidean)
        scored.append((tid, distance))

    enriched_results = []
    for tid, dist in sorted(scored, key=lambda x: x[1]):
        info = get_track_info(tid)
        if genre and info["genre"].lower() != genre.lower():
            continue
        enriched_results.append({
            "track_id": tid,
            "distance": round(dist, 4),
            "match_score": distance_to_score(dist),
            "title": info["title"],
            "artist": info["artist"],
            "genre": info["genre"]
        })
        if len(enriched_results) >= top_k:
            break

    query_info = get_track_info(track_id)
    return {
        "query_track": track_id,
        "query_title": query_info["title"],
        "query_artist": query_info["artist"],
        "results": enriched_results
    }


@app.get("/discover/{track_id}")
def discover(track_id: str, top_k: int = 5, genre: str | None = None, vibe_weight: float = 0.6):
    vibe_weight = max(0.0, min(vibe_weight, 1.0))

    track = track_lookup.get(track_id)
    query_curve = curve_lookup.get(track_id)
    if not track or query_curve is None:
        return {"error": f"track_id '{track_id}' not found or missing an energy curve."}

    embeddings = np.array(track["segment_embeddings"])
    query_vector = embeddings.mean(axis=0).tolist()
    vibe_results = collection.query(query_embeddings=[query_vector], n_results=200)

    vibe_scores = {}
    for meta, distance in zip(vibe_results["metadatas"][0], vibe_results["distances"][0]):
        tid = meta["track_id"]
        if tid == track_id:
            continue
        if tid not in vibe_scores or distance < vibe_scores[tid]:
            vibe_scores[tid] = distance

    max_vibe = max(vibe_scores.values()) if vibe_scores else 1
    candidates = list(vibe_scores.keys())

    progression_scores = {}
    for tid in candidates:
        candidate_curve = curve_lookup.get(tid)
        if candidate_curve is None:
            continue
        dist, _ = fastdtw(query_curve, candidate_curve, dist=scalar_euclidean)
        progression_scores[tid] = dist
    max_prog = max(progression_scores.values()) if progression_scores else 1

    blended = {}
    for tid in candidates:
        if tid not in progression_scores:
            continue
        norm_vibe = vibe_scores[tid] / max_vibe if max_vibe else 0
        norm_prog = progression_scores[tid] / max_prog if max_prog else 0
        base_blend = (vibe_weight * norm_vibe) + ((1 - vibe_weight) * norm_prog)
        adjustment = get_adjustment(track_id, tid) / 100
        blended[tid] = max(0, base_blend - adjustment)

    enriched_results = []
    for tid, dist in sorted(blended.items(), key=lambda x: x[1]):
        info = get_track_info(tid)
        if genre and info["genre"].lower() != genre.lower():
            continue
        enriched_results.append({
            "track_id": tid,
            "match_score": round(max(0, 100 - dist * 100)),
            "title": info["title"],
            "artist": info["artist"],
            "genre": info["genre"]
        })
        if len(enriched_results) >= top_k:
            break

    query_info = get_track_info(track_id)
    return {
        "query_track": track_id,
        "query_title": query_info["title"],
        "query_artist": query_info["artist"],
        "results": enriched_results
    }


@app.post("/feedback")
def submit_feedback(query_track: str, result_track: str, vote: int):
    if vote not in (1, -1):
        return {"error": "vote must be 1 (up) or -1 (down)"}
    new_total = record_vote(query_track, result_track, vote)
    return {"query_track": query_track, "result_track": result_track, "net_votes": new_total}


@app.get("/playlist/{track_id}")
def build_playlist(track_id: str, length: int = 5):
    length = max(2, min(length, 20))

    if track_id not in curve_lookup:
        return {"error": f"No energy curve for '{track_id}'"}

    playlist = [track_id]
    used = {track_id}

    for _ in range(length - 1):
        current_curve = curve_lookup[playlist[-1]]
        tail = current_curve[-len(current_curve) // 3:]

        best_tid, best_dist = None, None
        for tid, candidate_curve in curve_lookup.items():
            if tid in used:
                continue
            head = candidate_curve[:len(candidate_curve) // 3]
            dist, _ = fastdtw(tail, head, dist=scalar_euclidean)
            if best_dist is None or dist < best_dist:
                best_tid, best_dist = tid, dist

        if best_tid is None:
            break
        playlist.append(best_tid)
        used.add(best_tid)

    enriched = []
    for tid in playlist:
        info = get_track_info(tid)
        enriched.append({"track_id": tid, "title": info["title"], "artist": info["artist"], "genre": info["genre"]})

    return {"seed_track": track_id, "playlist": enriched}


@app.get("/query")
def natural_language_query(text: str, top_k: int = 5, genre: str | None = None):
    parsed = parse_query(text)
    reference_track_id = find_track_id_by_title(parsed.get("reference_track_title"), db_conn)

    if reference_track_id:
        if parsed.get("query_type") == "trajectory_similarity" and reference_track_id in curve_lookup:
            search_result = find_similar_progression(reference_track_id, top_k=top_k, genre=genre)
        else:
            search_result = find_similar(reference_track_id, top_k=top_k, genre=genre)

        if "error" in search_result:
            return {"parsed_query": parsed, "error": search_result["error"]}

        raw_results = search_result["results"]
    else:
        mood_text = parsed.get("mood_description") or text
        ranked = search_by_text(mood_text, collection, top_k=top_k * 4 if genre else top_k)
        raw_results = []
        for tid, dist in ranked:
            info = get_track_info(tid)
            if genre and info["genre"].lower() != genre.lower():
                continue
            raw_results.append({
                "track_id": tid,
                "distance": round(dist, 4),
                "match_score": distance_to_score(dist),
                "title": info["title"],
                "artist": info["artist"],
                "genre": info["genre"]
            })
            if len(raw_results) >= top_k:
                break

    explained_results = []
    for result in raw_results:
        explanation = explain_match(text, result)
        result_with_curve = {**result, "explanation": explanation, "energy_curve": curve_lookup.get(result["track_id"])}
        explained_results.append(result_with_curve)

    return {
        "user_query": text,
        "parsed_query": parsed,
        "used_reference_track": reference_track_id,
        "results": explained_results
    }
