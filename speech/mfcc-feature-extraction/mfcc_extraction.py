import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Load Audio
# -----------------------------
audio_path = "new_clip.wav"

y, sr = librosa.load(audio_path, sr=None)

print("Sampling Rate:", sr)
print("Audio Length:", len(y))


# -----------------------------
# Plot Waveform
# -----------------------------
plt.figure(figsize=(12,4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


# -----------------------------
# Compute Spectrogram (STFT)
# -----------------------------
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12,4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title("Spectrogram")
plt.show()


# -----------------------------
# Compute Mel Spectrogram
# -----------------------------
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

mel_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(12,4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title("Mel Spectrogram")
plt.show()


# -----------------------------
# Extract MFCC Features
# -----------------------------
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print("MFCC Shape:", mfcc.shape)


# -----------------------------
# Visualize MFCC
# -----------------------------
plt.figure(figsize=(12,4))
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
plt.title("MFCC Features")
plt.ylabel("MFCC Coefficients")
plt.show()


# -----------------------------
# Compute Mean MFCC (Feature Vector)
# -----------------------------
mfcc_mean = np.mean(mfcc.T, axis=0)

print("\nMFCC Feature Vector:")
print(mfcc_mean)