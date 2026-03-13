import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Path to audio
audio_path = "/home/meharuniza/Documents/speech-research/meharu-research/data/new_clip.wav"

# Load audio
y, sr = librosa.load(audio_path, sr=None)

print("Sampling Rate:", sr)

# Compute Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to decibels
mel_db = librosa.power_to_db(mel_spec, ref=np.max)

# Plot Mel Spectrogram
plt.figure(figsize=(12,4))
librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()