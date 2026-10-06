import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def read_wav_file(file):
    # open the wav file
    sample_rate, data = wavfile.read(file)
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

    time = np.arange(len(data)) / sample_rate

    return sample_rate, data, time, file_name

def plot_time_spectrum(time, data, original_file_name, sample_rate=None, title='time_spectrum', is_ifft=False):
    plt.figure(figsize=(20, 4))
    plt.plot(time, data)

    plt.title(f"{title}_{original_file_name}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    output_name = f"{title}_{original_file_name}"
    output_dir = IMAGES_DIR / original_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / f"{output_name}.png", dpi=300, bbox_inches="tight")
    if is_ifft: wavfile.write(output_dir / f"{output_name}.wav", sample_rate, data.astype(np.float32))
    plt.close()

def plot_fft(dB, data, sample_rate, original_file_name):
    N = len(data)
    title = "fft"

    # Compute full complex FFT first (needed for a proper IFFT later)
    complex_spectrum = np.fft.fft(data)

    # convert spectrum data to frequency domain, remove all frequencies > Nyquist frequency, and normalize the magnitudes
    spectrum = np.abs(complex_spectrum)[:N // 2] / N
    spectrum[1:-1] *= 2

    # Create an array of len(spectrum) evenly spaced values from 0 to sample_rate Hz
    frequency_resolution = sample_rate / N
    frequencies = np.arange(0, N // 2) * frequency_resolution

    if dB:
        # convert y-axis to decibels
        with np.errstate(divide='ignore'):
            spectrum = 20 * np.log10(spectrum)
        # Add log to the title
        title += "_log"

    plt.figure(figsize=(20, 4))
    plt.semilogx(frequencies, spectrum)
    plt.title(f"{title}_{original_file_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)" if dB else "Magnitude")
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    output_name = f"{title}_{original_file_name}"
    output_dir = IMAGES_DIR / original_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # return full complex spectrum for use in the ifft
    return complex_spectrum, N

def plot_ifft(complex_spectrum, N, sample_rate, time, original_file_name):
    # copy the aray to avoid modifying by reference
    filtered_spectrum = complex_spectrum.copy()

    k1 = int(N / sample_rate * 1)
    k2 = int(N / sample_rate * 80)      
            
    k1_mirror = N - k1      
    k2_mirror = N - k2 

    # Zero out frequencies between 1-80Hz (high pass filter)
    filtered_spectrum[k1 : k2+1] = 0       
    filtered_spectrum[k2_mirror : k1_mirror + 1] = 0       

    # perform inverse fft to go back to the time domain
    filtered_data = np.real(np.fft.ifft(filtered_spectrum))

    plot_time_spectrum(time, filtered_data, original_file_name, sample_rate, "ifft_filtered", True)

    return filtered_spectrum, filtered_data


if __name__ == "__main__":
    file_paths = ["peter-pipers-bicycle-5cm.wav"]

    for file_path in file_paths:
        if Path(AUDIO_DIR/file_path).exists():
            sample_rate, data, time, file_name = read_wav_file(AUDIO_DIR / file_path)
            spectrum, N = plot_fft(False, data, sample_rate, file_name)
            plot_time_spectrum(time, data, file_name)
            plot_fft(True, data, sample_rate, file_name)
            plot_ifft(spectrum, N, sample_rate, time, file_name)

        else:
            print(f"Error: {file_path} not found") 