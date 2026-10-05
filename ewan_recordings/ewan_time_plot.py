import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Load the wave file
sample_rate, data = wavfile.read("ewan_recordings/ewan_01_5cm.wav")
data = data[:int(len(data)/4)]

# 2. Print sample rate
print("Sample rate =", sample_rate)

# 3. Calculate time axis in seconds
duration = len(data) / sample_rate
time = np.linspace(0, duration, num=len(data))

# 4. Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, data, color="blue", alpha=0.7)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig('ewan_recordings/time_spectrum_silence_ewan_01_5cm.svg', dpi=300, bbox_inches="tight")
