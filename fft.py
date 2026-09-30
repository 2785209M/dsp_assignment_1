import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#open the wav file

# 1. load the wav file
sample_rate, data = wavfile.read('dsp_audio_1m.wav')

# 2. print sample rate
print(f'Sample rate: {sample_rate} Hz')

# 3. Calculate time axis in seconds
duration = len(data) / sample_rate
time = np.linspace(0, duration, len(data))

# 4. Plot the audio signal waveform
plt.figure(figsize=(10, 4))
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Signal Waveform')
plt.show()


#1. fft
freqdata = np.fft.fft(data)

#2. Create Frequency Axis
freqs = np.linspace(0, sample_rate, len(freqdata))

freqDataNyquist = freqdata[:len(freqdata)//2]
freqsNyquist = freqs[:len(freqs)//2]

#3. Plot the frequency spectrum
plt.figure(figsize=(10, 4))
plt.plot(freqsNyquist, np.abs(freqDataNyquist))
plt.xlim(0, 1000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.show()