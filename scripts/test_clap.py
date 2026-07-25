import laion_clap
import torch
import librosa

print("Loading CLAP model (first run downloads ~1-2GB of weights, be patient)...")
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()
model.eval()
print("Model loaded.")

waveform, sr = librosa.load("data/fma_small/000/000002.mp3", sr=48000, mono=True)
waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

with torch.no_grad():
    embedding = model.get_audio_embedding_from_data(x=waveform_tensor, use_tensor=True)

print(f"Embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[0][:5]}")
