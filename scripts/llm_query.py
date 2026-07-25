from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PARSE_MODEL = "gpt-4o-mini"
EXPLAIN_MODEL = "gpt-4o-mini"

PARSE_SYSTEM_PROMPT = """You convert a user's natural-language music request into JSON.
Return ONLY a JSON object matching this schema:
{
  "query_type": "point_similarity" | "trajectory_similarity",
  "reference_track_title": "the song title they mentioned by name, or null if none",
  "mood_description": "a short phrase capturing the mood/vibe they described, or null if they only referenced a track by name",
  "notes": "anything else relevant, like 'ends powerful' or 'builds slowly'"
}

Rules:
- Use "trajectory_similarity" if they describe change over time (build, drop, arc, gets louder/quieter, progression).
- Use "point_similarity" if they just want "songs like X" or describe a general mood with no shape/progression language.
- If they name a specific song or artist, put it in reference_track_title. Otherwise leave it null.
- If they describe a mood/vibe in their own words, put a short version of it in mood_description. Otherwise leave it null.
"""

def parse_query(user_text: str) -> dict:
    """Turn a natural-language request into structured JSON describing what search to run."""
    response = client.chat.completions.create(
        model=PARSE_MODEL,
        max_tokens=300,
        response_format={"type": "json_object"},  # forces valid JSON back, no markdown fences to strip
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


EXPLAIN_SYSTEM_PROMPT = """You explain, in one or two friendly sentences, why a recommended
song might match what the user asked for. Be specific but don't make up facts you weren't given —
only reference the title, artist, and similarity score you're given."""

def explain_match(user_query: str, result: dict) -> str:
    """Ask the LLM for a short, friendly explanation of why one result matched."""
    match_info = f"Title: {result.get('title')}, Artist: {result.get('artist')}, similarity distance: {result.get('distance')}"
    response = client.chat.completions.create(
        model=EXPLAIN_MODEL,
        max_tokens=150,
        messages=[
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": f"User asked: {user_query}\nMatched track info: {match_info}"}
        ]
    )
    return response.choices[0].message.content.strip()
