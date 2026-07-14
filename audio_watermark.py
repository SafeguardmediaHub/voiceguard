from audioseal import AudioSeal
import torchaudio
import torch
import librosa

# Load AudioSeal model
generator = AudioSeal.load_generator("audioseal_wm_16bits")

# Load audio
audio, sr = librosa.load("Recording.wav", sr=None, mono=True)

# Convert numpy array to torch tensor
audio = torch.tensor(audio, dtype=torch.float32)

# Add channel dimension: [samples] -> [1, samples]
audio = audio.unsqueeze(0)

# Resample if needed
if sr != 16000:
    audio = torchaudio.functional.resample(audio, sr, 16000)
    sr = 16000

# Add batch dimension: [1, samples] -> [1, 1, samples]
audio_batch = audio.unsqueeze(0)

# Generate watermark
watermark = generator.get_watermark(
    audio_batch,
    sample_rate=sr
)

# Apply watermark
watermarked = audio_batch + watermark

# Save result
torchaudio.save(
    "watermarked_output.wav",
    watermarked[0].cpu(),
    sr
)

print("Saved watermarked audio.")