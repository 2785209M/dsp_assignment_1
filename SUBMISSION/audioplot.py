import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import numpy as np

def read_wav_file(file):
    # extract the sample rate in Hz and data array from the wav file
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

    # Dividing number of elements in data by the sampling rate to get the total length in time of the sample
    time_array = np.arange(len(data)) / sample_rate

    return sample_rate, data, time_array, file_name

def plot_time_spectrum(time, data, original_file_name, output_directory, title='time_spectrum'):
    plt.figure(figsize=(20, 4))
    plt.plot(time, data)

    plt.title(f"{title}_{original_file_name}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    output_name = f"{title}_{original_file_name}"
    output_dir = output_directory / original_file_name / "time_domain"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / f"{output_name}.svg", dpi=300, bbox_inches="tight")
    plt.close()


def calculate_fft(data):
    # Compute full complex FFT first (needed for a proper IFFT later)
    complex_spectrum = np.fft.fft(data)

    # return full complex spectrum for use in the ifft
    return complex_spectrum


def get_nyquist_spectrum(complex_spectrum, sample_rate):
    N = len(complex_spectrum)

    # Calculate the normalized magnitude spectrum of the filtered data for plotting
    nyquist_spectrum = np.abs(complex_spectrum)[:N // 2] / N
    nyquist_spectrum[1:] *= 2

    # Create an array of len(spectrum) evenly spaced values from 0 to sample_rate Hz
    frequency_resolution = sample_rate / N
    nyquist_frequencies = np.arange(0, N // 2) * frequency_resolution

    return nyquist_spectrum, nyquist_frequencies


def plot_frequency_spectrum(frequencies, spectrum, original_file_name, output_directory, title="",log=False, dB=False):

    if dB:
        # convert y-axis to decibels
        with np.errstate(divide='ignore'):
            spectrum = 20 * np.log10(spectrum)
        # Add log to the title
        title += "_log"
        
    plt.figure(figsize=(20, 4))

    # plot the data
    if log:
        plt.semilogx(frequencies, spectrum)
    else:
        plt.plot(frequencies, spectrum)

    # name the plot and the axes
    plt.title(f"{title}_{original_file_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)" if dB else "Magnitude")

    # Set up the grid
    # ax = plt.gca()
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    # name the file
    if title:
        output_name = f"{title}_{original_file_name}"
    output_dir = output_directory / original_file_name / "frequency_domain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # save the plot
    plt.savefig(output_dir / f"{output_name}.svg", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":

    # put the audio files you want to process into the "audio_files" directory
    # The output will appear in the "images" directory
    OUTPUT_DIR = Path("2785209M-2664680S/part_1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Resolve the input relative to this script so it works regardless of the
    # directory from which the script is launched.
    file_path = Path(__file__).resolve().parent / "Welcome_1m.wav"

    sample_rate, data, time_array, file_name = read_wav_file(file_path)
                
    # Calculate FFT and the positive-frequency spectrum used for plotting
    complex_spectrum = calculate_fft(data)
    nyquist_adjusted_spectrum, nyquist_frequencies = get_nyquist_spectrum(
        complex_spectrum, sample_rate
    )

    # Plot original Audio in Time Domain
    plot_time_spectrum(time_array, data, file_name, OUTPUT_DIR, "Original_Audio")

    # Plot Original Audio in Frequency Domain
    plot_frequency_spectrum(nyquist_frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
    plot_frequency_spectrum(nyquist_frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

    print(f"Successfully processed {file_path}")