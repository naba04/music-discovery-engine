import os
import json
import librosa
import numpy as np

DATA_DIR = "data/fma_small"
OUTPUT_FILE = "energy_curves.json"
HOP_LENGTH = 512
CURVE_POINTS = 50

def filepath_from_track_id(track_id: str) -> str:
    # e.g. "000002" -> data/fma_small/000/000002.mp3
    subfolder = track_id[:3]
    return os.path.join(DATA_DIR, subfolder, f"{track_id}.mp3")

def get_energy_curve(filepath, hop_length=HOP_LENGTH, target_points=CURVE_POINTS):
    waveform, sr = librosa.load(filepath, sr=None, mono=True)
    rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(5) / 5, mode="same")
    rms_min, rms_max = rms_smooth.min(), rms_smooth.max()
    rms_norm = (rms_smooth - rms_min) / (rms_max - rms_min + 1e-8)
    original_indices = np.linspace(0, len(rms_norm) - 1, num=target_points)
    curve = np.interp(original_indices, np.arange(len(rms_norm)), rms_norm)
    return curve.tolist()

def main():
    with open("embeddings.json") as f:
        all_tracks = json.load(f)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            curves = json.load(f)
    else:
        curves = {}

    print(f"{len(curves)} curves already computed — resuming from there")
    remaining = [t for t in all_tracks if t["track_id"] not in curves]
    print(f"{len(remaining)} left to process")

    for i, track in enumerate(remaining):
        tid = track["track_id"]
        try:
            curves[tid] = get_energy_curve(filepath_from_track_id(tid))
        except Exception as e:
            print(f"  skipped {tid}: {e}")
            continue

        if (i + 1) % 100 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(curves, f)
            print(f"  saved — {len(curves)} total done")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(curves, f)
    print(f"Done. {len(curves)} curves in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
