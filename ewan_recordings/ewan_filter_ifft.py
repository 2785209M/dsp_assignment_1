import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Load the wave file
sample_rate, data = wavfile.read("ewan_recordings/ewan_01_1m.wav")
#data = data[:int(len(data)/4)]
N = len(data)
duration = len(data) / sample_rate
time = np.linspace(0, duration, num=len(data))

# 1. Fft
Xf = (np.fft.fft(data))

# 2. Filter out 1-80 Hz
# N/sample_rate gives samples per Herz. Multiply by desired frequency to find which sample it corresponds to
k1 = int(N / sample_rate * 1)
k2 = int(N / sample_rate * 80)

# and in mirrored spectrum
k1_mirror = N - k1
k2_mirror = N - k2

# set those bands of spectrum to 0
# remember that below nyquist goes from k1:k2 whilst mirror goes from k2:k1
Xf[k1 : k2 + 1] = 0
Xf[k2_mirror : k1_mirror + 1] = 0

#create frequency axis
df = sample_rate / N
freqs = np.arange(0, N) * df

#test plot to see frequency spectrum
plt.figure()
plt.plot(freqs, np.abs(Xf))
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("")
plt.savefig('ewan_recordings/fft_spectrum_filtered_01_1m.svg', dpi=300, bbox_inches="tight")

#take ifft
xn = (np.fft.ifft(Xf))

# 4. Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, xn, color="blue", alpha=0.7)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig('ewan_recordings/time_spectrum_ifft_ewan_01_1m.svg', dpi=300, bbox_inches="tight")
