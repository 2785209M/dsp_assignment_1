import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import wave as wv

IMAGES_DIR = Path("images")

#open the wav file
def read_wav_file(file):
    sample_rate, data = wavfile.read(file)
    time = np.arange(len(data)) / sample_rate
    data = data/2**31
    file_name = file[:-4]

    return sample_rate, data, time, file_name

def plot_time_spectrum(time, data, title, original_file_name = ""):
    plt.figure(figsize=(20, 4))
    plt.plot(time, data)
    plt.title(title + "_" + original_file_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.savefig(IMAGES_DIR / f"{title}_{original_file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_frequency_spectrum(dB, data, sample_rate, title, original_file_name = ""):
    # convert spectrum data to frequency domain
    spectrum = np.abs(np.fft.fft(data))
    # remove all frequencies > Nyquist frequency
    spectrum = spectrum[:len(spectrum) // 2]
    # Create an array of len(spectrum) evenly spaced values from 0 to sample_rate Hz
    frequencies = np.linspace(0, sample_rate, len(spectrum))

    if(dB == True):
        # convert y-axis to decibels
        spectrum = 20* np.log10(spectrum)

    plt.figure(figsize=(20, 4))
    plt.semilogx(frequencies, spectrum)
    plt.title(title + "_" + original_file_name)
    plt.xlabel("Frequency (Hz)")
    if(dB == True):
        plt.ylabel("Magnitude (dB)")
    else:
        plt.ylabel("Magnitude")
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.savefig(IMAGES_DIR / f"{title}_{original_file_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


sample_rate_5cm, data_5cm, time_5cm, file_name = read_wav_file("Untitled(1).wav")

plot_time_spectrum(time_5cm, data_5cm, 'time_spectrum', file_name)
plot_frequency_spectrum(False, data_5cm, sample_rate_5cm, "frequency_spectrum", file_name)
plot_frequency_spectrum(True, data_5cm, sample_rate_5cm, "frequency_spectrum_log", file_name)