def find_track_id_by_title(title_query: str, db_conn):
    """
    Loosely match a user-typed title against known track titles in the database.
    Returns a track_id if a confident match is found, else None.
    """
    if not title_query:
        return None
    pattern = f"%{title_query.strip().lower()}%"
    row = db_conn.execute(
        "SELECT track_id FROM tracks WHERE lower(title) LIKE ? LIMIT 1",
        (pattern,)
    ).fetchone()
    return row[0] if row else None
