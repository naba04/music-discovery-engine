# Music Discovery Engine

An AI-powered music discovery API that finds similar songs by *sound*, not metadata — using audio embeddings (CLAP), vector similarity search (Chroma), energy-curve shape matching (DTW), and natural-language querying (LLM-powered).

## Features
- **Similarity search** — find songs that sound like a given track (`/similar`)
- **Progression matching** — find songs with a similar energy arc/shape over time, not just overall vibe (`/similar_progression`)
- **Hybrid ranking** — blend overall vibe + energy-curve shape into one score (`/discover`)
- **Natural language search** — "something moody and slow like Radiohead" (`/query`), powered by an LLM parsing the request
- **Genre filtering** on every search endpoint
- **Feedback loop** — thumbs up/down nudges future rankings (`/feedback`)
- **Auto-generated playlists** — chains tracks so energy flows smoothly from one to the next (`/playlist`)
- Built on the [FMA (Free Music Archive)](https://freemusicarchive.org/) dataset — real, Creative Commons-licensed tracks, not synthetic data

## Tech stack
FastAPI · Chroma (vector DB) · SQLite (metadata) · CLAP audio embeddings · fastdtw · OpenAI (query parsing/explanations) · Docker

## Running it locally

This project is designed to run locally via Docker — it isn't currently deployed to a public URL, since the CLAP audio model needs more RAM than most free hosting tiers provide.

**Prerequisites:** Docker Desktop, an OpenAI API key, the [`fma_small`](https://github.com/mdeff/fma) dataset downloaded into `data/fma_small/`.

```bash
git clone https://github.com/naba04/music-discovery-engine.git
cd music-discovery-engine
echo "OPENAI_API_KEY=your-key-here" > .env
docker build -t music-backend .
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/music.db:/app/music.db \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/energy_curves.json:/app/energy_curves.json \
  -v $(pwd)/data/fma_small:/app/data/fma_small:ro \
  music-backend
```

Or use the paired [frontend repo](https://github.com/naba04/music-discovery-frontend) alongside it via `docker compose up` for the full experience.

Visit `http://localhost:8000/` to confirm it's running, or `http://localhost:8000/similar/<track_id>` to test a search.
