import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave as wv

#open the wav file
def read_wav_file(file):
    sample_rate, data = wavfile.read(file)
    time = np.arange(len(data)) / sample_rate
    data = data/32768

    return sample_rate, data, time

def plot_time_spectrum(time, data, title):
    plt.figure(figsize=(12.8, 4.8))
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_frequency_spectrum(data, sample_rate, title):
    spectrum = np.abs(np.fft.fft(data))
    spectrum = spectrum[:len(spectrum) // 2] # remove all frequencies > Nyquist frequency
    frequencies = np.linspace(0, sample_rate, len(spectrum))

    plt.plot(frequencies, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close()

def plot_frequency_spectrum_log(data, sample_rate, title):
    plt.figure(figsize=(12.8, 4.8))
    spectrum = np.abs(np.fft.fft(data))
    spectrum = spectrum[:len(spectrum) // 2] # remove all frequencies > Nyquist frequency
    frequencies = np.linspace(0, sample_rate, len(spectrum))

    spectrum = 20* np.log10(spectrum)

    plt.semilogx(frequencies, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.savefig(title, dpi=600, bbox_inches="tight")
    plt.close()


sample_rate_5cm, data_5cm, time_5cm = read_wav_file("Untitled.wav")

plot_time_spectrum(time_5cm, data_5cm, 'time_spectrum.png')
plot_frequency_spectrum(data_5cm[:len(data_5cm) // 2], sample_rate_5cm, "frequency_spectrum.png")
plot_frequency_spectrum(data_5cm*np.hamming(len(data_5cm)), sample_rate_5cm, "frequency_spectrum_hamming.png")
plot_frequency_spectrum_log(data_5cm[:len(data_5cm)] // 2, sample_rate_5cm, "frequency_spectrum_log.png")
plot_frequency_spectrum_log(data_5cm*np.hamming(len(data_5cm)), sample_rate_5cm, "frequency_spectrum_hamming_log.png")

# TODO write a function to automate the creation of frequency plots
