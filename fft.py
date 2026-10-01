import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#open the wav file
def read_wav_file(file):
    sample_rate, data = wavfile.read(file)
    time = np.arange(len(data)) / sample_rate

    return sample_rate, data, time

def plot_time_spectrum(time, data, title):
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_frequency_spectrum(data, sample_rate, title):
    spectrum = np.abs(np.fft.fft(data))
    frequencies = np.linspace(0, sample_rate, len(spectrum))

    plt.plot(frequencies, spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close()


sample_rate_5cm, data_5cm, time_5cm = read_wav_file("dsp_audio_5cm.wav")

plot_time_spectrum(time_5cm, data_5cm, 'time_spectrum.png')
plot_frequency_spectrum(data_5cm, sample_rate_5cm, "frequency_spectrum.png")

# TODO write a function to automate the creation of frequency plots
