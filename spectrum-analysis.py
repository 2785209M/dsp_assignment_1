import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

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


def plot_time_spectrum(time, data, original_file_name, sample_rate=None, title='time_spectrum', is_ifft=False):
    plt.figure(figsize=(20, 4))
    plt.plot(time, data)

    plt.title(f"{title}_{original_file_name}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    output_name = f"{title}_{original_file_name}"
    output_dir = IMAGES_DIR / original_file_name / "time_domain"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / f"{output_name}.svg", dpi=300, bbox_inches="tight")
    if is_ifft: wavfile.write(output_dir / f"{output_name}.wav", sample_rate, data.astype(np.float32))
    plt.close()


def plot_frequency_spectrum(frequencies, spectrum, original_file_name, title="", log=False, dB=False):

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
    output_dir = IMAGES_DIR / original_file_name / "frequency_domain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # save the plot
    plt.savefig(output_dir / f"{output_name}.svg", dpi=300, bbox_inches="tight")
    plt.close()


def calculate_fft(data, sample_rate):
    N = len(data)

    # Compute full complex FFT first (needed for a proper IFFT later)
    complex_spectrum = np.fft.fft(data)

    # Create an array of len(spectrum) evenly spaced values from 0 to sample_rate Hz
    frequency_resolution = sample_rate / N
    frequencies = np.arange(0, N // 2) * frequency_resolution

    # return full complex spectrum for use in the ifft
    return complex_spectrum, frequencies, N

def calculate_ifft(complex_spectrum):
    # copy the aray to avoid modifying by reference
    filtered_complex_spectrum = complex_spectrum.copy()

    # perform inverse fft to go back to the time domain
    filtered_data = np.real(np.fft.ifft(filtered_complex_spectrum))

    return filtered_data


def get_nyquist_spectrum(complex_spectrum):
    # Calculate the normalized magnitude spectrum of the filtered data for plotting
    nyquist_spectrum = np.abs(complex_spectrum)[:N // 2] / N
    nyquist_spectrum[1:-1] *= 2
    return nyquist_spectrum


def raised_cosine_filter(data, sample_rate, f_lower, f_upper, transition_window_width):
    N = len(data)
    nyquist_freq = sample_rate / 2

    if(f_lower < 0 or f_lower > nyquist_freq):
        print(f"WARNING: lower bound frequency {f_lower} is outside of the accepted frequency range. Setting to zero.")
        f_lower = 0
    if(f_upper > nyquist_freq or f_upper < 0):
        print(f"WARNING: upper bound frequency {f_upper} is outside of the accepted frequency range. Setting to sample_rate/2.")
        f_upper = int(nyquist_freq)

    total_time = N / sample_rate

    #calculate the corresponding indices of f_lower and f_upper as well as the ends of the transition windows
    k_lower = int(total_time * f_lower)
    k_upper = int(total_time * f_upper)

    # Calculate transition bounds to prevent negative indices
    k_trans_lower = int(max(0, (total_time * (f_lower - transition_window_width))))
    k_trans_upper = int(min(N // 2, (total_time * (f_upper + transition_window_width))))

    # Zero out frequencies from f_lower to f_upper
    data[k_lower : k_upper + 1] = 0
    # And in mirrored spectrum
    data[N - k_upper : N - k_lower + 1] = 0

    # How many bins between the start and end of the transition windows
    num_taper_bins_upper_transition = k_trans_upper - k_upper
    num_taper_bins_lower_transition = k_lower - k_trans_lower

    # Cosine window formula going smoothly from 0.0 to 1.0
    upper_transition_window = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins_upper_transition)))
    lower_transition_window = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins_lower_transition)))

    # Apply transition to positive frequencies
    data[k_upper : k_trans_upper] *= upper_transition_window
    data[k_trans_lower : k_lower] *= lower_transition_window[::-1]

    # And mirrored spectrum ([::-1] reverse the transition window for the mirrored spectrum)
    data[N - k_trans_upper : N - k_upper] *= upper_transition_window[::-1]
    data[N - k_lower : N - k_trans_lower] *= lower_transition_window

    return data


if __name__ == "__main__":
    file_paths = ["peter-pipers-bicycle-5cm.wav"]

    for file_path in file_paths:
        if Path(AUDIO_DIR/file_path).exists():
            # Extract Audio data from file
            sample_rate, data, time_array, file_name = read_wav_file(AUDIO_DIR / file_path)
            
            # Calculate FFT and IFFT
            complex_spectrum, frequencies, N = calculate_fft(data, sample_rate)
            nyquist_adjusted_spectrum = get_nyquist_spectrum(complex_spectrum)

            # Plot original Audio in Time Domain
            plot_time_spectrum(time_array, data, file_name, title = "Original_Audio")
            
            # Plot Original Audio in Frequency Domain
            plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, "FFT")
            plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, "FFT", True, True)

            # Apply a filtering window to remove specific frequencies from the audio signal
            filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 1, 80, 20)
            filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 4000, 5000, 200)
            filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 9000, 10000, 20)

            # Obtain the nyquist adjusted filtered frequency spectrum for plotting
            filtered_nyquist_spectrum = get_nyquist_spectrum(filtered_complex_spectrum)

            # Revert the complex spectrum to time domain
            filtered_data = calculate_ifft(complex_spectrum)

            # Plot Filtered Audio in Frequency Domain
            plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, "IFFT")
            plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, "IFFT", True, True)
            
            # Plot Filtered Audio in Time Domain
            plot_time_spectrum(time_array, filtered_data, file_name, sample_rate, "Filtered_Audio", True)

            print(f"Successfully processed {file_path}")

        else:
            print(f"Error: {file_path} not found") 