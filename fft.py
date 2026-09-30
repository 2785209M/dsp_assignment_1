import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#open the wav file
def read_wav_file(file):
    # 1. load the wav file
    sample_rate, data = wavfile.read(file)

    # 2. print sample rate
    print(f'Sample rate: {sample_rate} Hz')

    # 3. Calculate time axis in seconds
    duration = len(data) / sample_rate
    time = np.linspace(0, duration, len(data))

    return sample_rate, data, time

def calculate_fft(sample_rate, data):
    #1. fft
    freqdata = np.fft.fft(data)

    #2. Create Frequency Axis
    freqs = np.linspace(0, sample_rate, len(freqdata))

    return freqs, freqdata


def calculate_Nyquist_fft(freqs, freqdata):
    #1. cut mirrored half of data
    freqdatanyquist = freqdata[:len(freqdata)//2]
    #2. cut number of frequencies by half
    freqsnyquist = freqs[:len(freqs)//2]

    return freqsnyquist, freqdatanyquist

sample_rate, data, time = read_wav_file('dsp_audio_1m.wav')
freqs, freqdata = calculate_fft(sample_rate, data)
freqsnyquist, freqdatanyquist = calculate_Nyquist_fft(freqs, freqdata)

plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Signal Waveform')
plt.savefig('audio_signal_waveform.png', dpi=300, bbox_inches='tight')
plt.clf()

plt.plot(freqs, np.abs(freqdata))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.savefig('frequency_spectrum.png', dpi=300, bbox_inches='tight')
plt.clf()

plt.plot(freqsnyquist, np.abs(freqdatanyquist))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.savefig('frequency_spectrum_nyquist.png', dpi=300, bbox_inches='tight')
plt.clf()
