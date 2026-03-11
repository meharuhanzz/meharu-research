import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/home/meharuniza/Documents/speech-research/meharu-research/speech/stft-spectrogram/new_clip.wav"

# Load audio
y, sr = librosa.load(audio_path, sr=None)

print("Sampling Rate:", sr)

# Plot waveform
plt.figure(figsize=(12,4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.show()

# Compute STFT
D = librosa.stft(y)

# Convert to decibel scale
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot Spectrogram
plt.figure(figsize=(12,4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Spectrogram (STFT)")
plt.show()