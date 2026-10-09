import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Load the wave file
sample_rate, data = wavfile.read("audio_files/peter-pipers-bicycle-5cm.wav")

if data.dtype == np.int16:
        # 16-bit integer PCM
    data = data.astype(np.float32) / 32768.0
elif data.dtype == np.int32:
        # 32-bit integer PCM
    data = data.astype(np.float32) / 2147483648.0

N = len(data)
data = data*6 #divide by 2 at the end
data = np.tanh(data)

# 1. Fft
freqData = np.abs(np.fft.fft(data)) / N

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
plt.savefig('ewan_recordings/fft-peter-piper-tanh.svg', dpi=300, bbox_inches="tight")

# 4. Plot on log-log scale
magnitude_db = 20 * np.log10(np.abs(freqDataNyquist))

plt.figure(figsize=(18, 6))
plt.semilogx(freqsNyquist[69:], magnitude_db[69:])
plt.title("FFT Spectrum (Log-Log Scale)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.savefig('ewan_recordings/fft-log-peter-piper-tanh.svg', dpi=300, bbox_inches="tight")
plt.close()

