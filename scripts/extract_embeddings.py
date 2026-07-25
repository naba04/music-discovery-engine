import librosa
import numpy as np
import torch
import laion_clap
import os
import json

SEGMENT_SECONDS = 10
SAMPLE_RATE = 48000

print("Loading CLAP model...")
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()
model.eval()
print("Model ready.")

def segment_audio(filepath, segment_seconds=SEGMENT_SECONDS):
    """Load a file and cut it into equal-length chunks, padding the last one if needed."""
    waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    segment_len = segment_seconds * SAMPLE_RATE
    segments = []
    for start in range(0, len(waveform), segment_len):
        chunk = waveform[start:start + segment_len]
        if len(chunk) < segment_len:
            chunk = np.pad(chunk, (0, segment_len - len(chunk)))
        segments.append(chunk)
    return segments

def embed_segments(segments):
    """Convert each audio chunk into a 512-number embedding vector."""
    embeddings = []
    for seg in segments:
        seg_tensor = torch.from_numpy(seg).float().unsqueeze(0)
        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=seg_tensor, use_tensor=True)
        embeddings.append(emb.squeeze(0).numpy().tolist())
    return embeddings

def process_track(filepath, track_id):
    segments = segment_audio(filepath)
    embeddings = embed_segments(segments)
    return {
        "track_id": track_id,
        "filepath": filepath,
        "num_segments": len(segments),
        "segment_embeddings": embeddings
    }

def find_mp3_files(root_dir, limit=None):
    """Walk through FMA's subfolder structure and collect mp3 file paths."""
    files = []
    for subdir, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".mp3"):
                files.append(os.path.join(subdir, fname))
                if limit and len(files) >= limit:
                    return files
    return files

if __name__ == "__main__":
    # CHANGE THIS LINE to process 1000 tracks instead of 5
    audio_files = find_mp3_files("data/fma_small", limit=1000)  # Now processing 1000 tracks
    print(f"Found {len(audio_files)} files to process.")

    output = []
    for i, filepath in enumerate(audio_files):
        track_id = os.path.basename(filepath).replace(".mp3", "")
        print(f"[{i+1}/{len(audio_files)}] Processing {track_id}...")
        try:
            result = process_track(filepath, track_id)
            output.append(result)
            print(f"  -> {result['num_segments']} segments extracted.")
        except Exception as e:
            print(f"  -> SKIPPED due to error: {e}")

    with open("embeddings.json", "w") as f:
        json.dump(output, f)

    print(f"\nDone. Successfully processed {len(output)} / {len(audio_files)} tracks.")
    print("Saved to embeddings.json")
