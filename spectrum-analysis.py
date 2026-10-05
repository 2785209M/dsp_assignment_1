import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def read_wav_file(file):
    # open the wav file
    sample_rate, data = wavfile.read(file)

    time = np.arange(len(data)) / sample_rate

    file_name = Path(file).stem

    # force the data to mono by only taking the first channel if a stereo input was given
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize the input to a [-1:1] range - 16-bit PCM files are integers from [-32,768:32,767]
    # which is why we have to divide them by 32768
    if data.dtype == np.int16:
        # 16-bit integer PCM
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        # 32-bit integer PCM
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        # 8-bit integer PCM (unsigned, centered at 128)
        data = (data.astype(np.float32) - 128) / 128
    # if data is already np.float32, it is already normalized and sits in the range of [-1:1]

    return sample_rate, data, time, file_name

def plot_time_spectrum(time, data, title, original_file_name = ""):
    plt.figure(figsize=(20, 4))
    plt.plot(time, data)

    plt.title(f"{title}_{original_file_name}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    
    plt.savefig(IMAGES_DIR / f"{title}_{original_file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_frequency_spectrum(dB, data, sample_rate, title, original_file_name = ""):
    N = len(data)

    # convert spectrum data to frequency domain, remove all frequencies > Nyquist frequency, and normalize the magnitudes
    spectrum = np.abs(np.fft.fft(data))[:N // 2] / N
    spectrum[1:-1] *= 2

    # Create an array of len(spectrum) evenly spaced values from 0 to sample_rate Hz
    frequency_resolution = sample_rate / N
    frequencies = np.arange(0, N // 2) * frequency_resolution

    if dB:
        # convert y-axis to decibels
        spectrum = 20* np.log10(spectrum)

    plt.figure(figsize=(20, 4))
    plt.semilogx(frequencies, spectrum)
    plt.title(f"{title}_{original_file_name}")
    plt.xlabel("Frequency (Hz)")
    if dB:
        plt.ylabel("Magnitude (dB)")
    else:
        plt.ylabel("Magnitude")
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.savefig(IMAGES_DIR / f"{title}_{original_file_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    file_paths = ["Untitled(1).wav", "ewan_recordings/ewan_01_1m.wav", "ewan_recordings/ewan_01_5cm.wav"]

    for file_path in file_paths:
        if Path(file_path).exists():
            sample_rate_5cm, data_5cm, time_5cm, file_name = read_wav_file(file_path)
    
            plot_time_spectrum(time_5cm, data_5cm, 'time_spectrum', file_name)
            plot_frequency_spectrum(False, data_5cm, sample_rate_5cm, "frequency_spectrum", file_name)
            plot_frequency_spectrum(True, data_5cm, sample_rate_5cm, "frequency_spectrum_log", file_name)


        else:
            print(f"Error: {file_path} not found") 