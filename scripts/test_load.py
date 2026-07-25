import librosa

filepath = "data/fma_small/000/000002.mp3"
waveform, sr = librosa.load(filepath, sr=48000, mono=True)
print(f"Loaded audio. Sample rate: {sr}, Length: {len(waveform)} samples, Duration: {len(waveform)/sr:.2f} sec")
