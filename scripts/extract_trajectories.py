import librosa
import numpy as np
import json

HOP_LENGTH = 512
CURVE_POINTS = 50  # downsample every curve to this many points, so DTW stays fast and comparable

def get_energy_curve(filepath, hop_length=HOP_LENGTH, target_points=CURVE_POINTS):
    """Load a track and compute a normalized, fixed-length loudness-over-time curve."""
    waveform, sr = librosa.load(filepath, sr=None, mono=True)
    rms = librosa.feature.rms(y=waveform, hop_length=hop_length)[0]

    # Smooth out frame-to-frame noise so the shape reflects the song's arc, not jitter
    rms_smooth = np.convolve(rms, np.ones(5) / 5, mode="same")

    # Normalize to 0-1 so a quiet acoustic song and a loud rock song are comparable on SHAPE
    rms_min, rms_max = rms_smooth.min(), rms_smooth.max()
    rms_norm = (rms_smooth - rms_min) / (rms_max - rms_min + 1e-8)

    # Downsample every song to the same number of points, regardless of its original length,
    # so DTW is comparing "shape across the whole song" fairly between a 20s clip and a 30s clip
    original_indices = np.linspace(0, len(rms_norm) - 1, num=target_points)
    curve = np.interp(original_indices, np.arange(len(rms_norm)), rms_norm)
    return curve.tolist()

if __name__ == "__main__":
    with open("embeddings.json") as f:
        all_tracks = json.load(f)

    output = {}
    for track in all_tracks:
        track_id = track["track_id"]
        print(f"Computing energy curve for {track_id}...")
        try:
            output[track_id] = get_energy_curve(track["filepath"])
        except Exception as e:
            print(f"Skipped {track_id}: {e}")

    with open("energy_curves.json", "w") as f:
        json.dump(output, f)
    print(f"Done. Computed curves for {len(output)} tracks.")
