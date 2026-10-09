import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Load the wave file
sample_rate, data = wavfile.read("ewan_recordings/ewan_01_5cm.wav")
#data = data[:int(len(data)/4)]
N = len(data)
duration = len(data) / sample_rate
time = np.linspace(0, duration, num=len(data))

# 1. Fft
Xf = (np.fft.fft(data))

# 2. Filter out low frequencies from 0 to 500 to remove plosive bursts
# This is done with an additional cosine window because sharp edges in the frequency domain cause sinc ringing in time domain

f_cutoff = 500      # Everything below f_cut becomes 0
trans_width = 150    # Width of smooth transition zone (350 Hz to 500 Hz)

# Calculate bin indices
k1 = int(N / sample_rate * 5)
k2 = int(N / sample_rate * f_cutoff)
k_trans = int(N / sample_rate * (f_cutoff + trans_width))

# Zero out frequencies from k1 to k2
Xf[k1 : k2 + 1] = 0
Xf[N - k2 : N - k1 + 1] = 0

# Create a smooth Cosine transition (0 to 1) from 350 Hz to 500 Hz
num_taper_bins = k2 - k_trans
# Cosine window formula going smoothly from 0.0 to 1.0
transition_window = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins)))

# Apply transition to positive frequencies
Xf[k2 : k_trans] *= transition_window
# And mirrored zone ([::-1] reverse the transition window for the mirrored zone)
Xf[N - k_trans : N - k2 + 1] *= transition_window[::-1]

#create frequency axis
df = sample_rate / N
freqs = np.arange(0, N) * df

#test plot to see frequency spectrum
plt.figure()
plt.plot(freqs, np.abs(Xf))
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("")
plt.savefig('ewan_recordings/fft_spectrum_filtered_01_5cm.svg', dpi=300, bbox_inches="tight")

#take ifft
xn = (np.fft.ifft(Xf)).real

# 4. Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, xn, color="blue", alpha=0.7)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig('ewan_recordings/time_spectrum_ifft_ewan_01_5cm.svg', dpi=300, bbox_inches="tight")

#write out wavefile
wavfile.write("ewan_recordings/ewan_01_5cm_filtered.wav", sample_rate, xn)

