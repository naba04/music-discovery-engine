import sqlite3

DB_PATH = "music.db"
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def record_vote(query_track: str, result_track: str, vote: int):
    """vote is +1 (thumbs up) or -1 (thumbs down)."""
    row = _conn.execute(
        "SELECT net_votes FROM feedback WHERE query_track = ? AND result_track = ?",
        (query_track, result_track)
    ).fetchone()
    new_total = (row[0] if row else 0) + vote
    _conn.execute(
        "INSERT OR REPLACE INTO feedback (query_track, result_track, net_votes) VALUES (?, ?, ?)",
        (query_track, result_track, new_total)
    )
    _conn.commit()
    return new_total

def get_adjustment(query_track: str, result_track: str) -> float:
    """Returns a small score adjustment based on accumulated votes for this pairing."""
    row = _conn.execute(
        "SELECT net_votes FROM feedback WHERE query_track = ? AND result_track = ?",
        (query_track, result_track)
    ).fetchone()
    net_votes = row[0] if row else 0
    return max(-15, min(15, net_votes * 3))
