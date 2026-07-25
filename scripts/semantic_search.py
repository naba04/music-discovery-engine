import laion_clap
import numpy as np

# Loaded once, at import time, so this cost is paid at server startup, not per request
_model = laion_clap.CLAP_Module(enable_fusion=False)
_model.load_ckpt()
_model.eval()

def embed_text_query(text: str):
    """Embed a text description into the same vector space as the audio embeddings."""
    return _model.get_text_embedding([text])[0].tolist()

def search_by_text(text: str, collection, track_id_to_exclude=None, top_k=5):
    """Search stored song segments directly against a text description, no reference track needed."""
    query_vector = embed_text_query(text)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=30
    )

    scored = {}
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        tid = meta["track_id"]
        if tid == track_id_to_exclude:
            continue
        if tid not in scored or distance < scored[tid]:
            scored[tid] = distance

    return sorted(scored.items(), key=lambda x: x[1])[:top_k]
