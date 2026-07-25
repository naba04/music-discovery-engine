import os
import json
import time
import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

DATA_DIR = "data/fma_small"
OUTPUT_FILE = "embeddings.json"
BATCH_SIZE = 25
SEGMENT_SECONDS = 10

print("Loading CLAP model...")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model.eval()

def find_mp3_files(root_dir):
    paths = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".mp3"):
                paths.append(os.path.join(subdir, f))
    return sorted(paths)

def load_existing():
    if not os.path.exists(OUTPUT_FILE):
        return []
    with open(OUTPUT_FILE) as f:
        return json.load(f)

def track_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def embed_track(path):
    waveform, sr = librosa.load(path, sr=48000, mono=True)
    segment_len = SEGMENT_SECONDS * sr
    segments = [waveform[i:i + segment_len] for i in range(0, len(waveform), segment_len)]
    segments = [s for s in segments if len(s) >= sr * 3]

    embeddings = []
    for seg in segments:
        inputs = processor(audio=seg, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_audio_features(**inputs)
        if hasattr(emb, "pooler_output"):
            emb = emb.pooler_output
        embeddings.append(emb.squeeze().numpy().tolist())
    return embeddings

def main():
    all_paths = find_mp3_files(DATA_DIR)
    print(f"Found {len(all_paths)} mp3 files total in {DATA_DIR}")

    processed = load_existing()
    done_ids = {t["track_id"] for t in processed}
    print(f"{len(done_ids)} already processed — resuming from there")

    remaining = [p for p in all_paths if track_id_from_path(p) not in done_ids]
    print(f"{len(remaining)} left to process")

    batch = []
    start_time = time.time()
    for i, path in enumerate(remaining):
        tid = track_id_from_path(path)
        try:
            embeddings = embed_track(path)
            if not embeddings:
                print(f"  skipping {tid}: no usable segments")
                continue
            batch.append({"track_id": tid, "segment_embeddings": embeddings})
        except Exception as e:
            print(f"  FAILED on {tid}: {e}")
            continue

        if len(batch) >= BATCH_SIZE:
            processed.extend(batch)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(processed, f)
            elapsed = time.time() - start_time
            print(f"  saved batch — {len(processed)} total done, {elapsed:.0f}s elapsed")
            batch = []

    if batch:
        processed.extend(batch)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(processed, f)

    print(f"Done. {len(processed)} tracks in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
