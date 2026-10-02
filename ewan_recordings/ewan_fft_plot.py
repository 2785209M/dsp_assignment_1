import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Load the wave file
sample_rate, data = wavfile.read("ewan_recordings/ewan_01_5cm.wav")
data = data[:int(len(data)/4)]
N = len(data)

# 1. Fft
freqData = np.fft.fft(data) / N

# 2. Nyquist theorem on fft data
freqDataNyquist = 2 * freqData[:N // 2]

# 3. Create Frequency Axis
df = sample_rate / N
freqsNyquist = np.arange(0, N // 2) * df

#3. Plot Frequency data
plt.figure()
plt.plot(freqsNyquist, np.abs(freqDataNyquist))
plt.title("FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("")
plt.savefig('ewan_recordings/fft_spectrum_silence_ewan_01_5cm.svg', dpi=300, bbox_inches="tight")

# 4. Plot on log-log scale
magnitude_db = 20 * np.log10(np.abs(freqDataNyquist))

plt.figure(figsize=(12, 6))
plt.semilogx(freqsNyquist[4:], magnitude_db[4:])
plt.title("FFT Spectrum (Log-Log Scale)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.savefig('ewan_recordings/fft_loglog_spectrum_silence_ewan_01_5cm.svg', dpi=300, bbox_inches="tight")
plt.close()