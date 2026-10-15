import SUBMISSION.audioplot as ap # part 1 code
import numpy as np
from pathlib import Path

# put the audio files you want to process into the "audio_files" directory
# The output will appear in the "OUTPUT" directory
OUTPUT_DIR = Path("2785209M-2664680S/part_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def calculate_ifft(complex_spectrum):
    # copy the aray to avoid modifying by reference
    filtered_complex_spectrum = complex_spectrum.copy()

    # perform inverse fft to go back to the time domain
    filtered_data = np.real(np.fft.ifft(filtered_complex_spectrum))

    return filtered_data


# TODO merge raised_cosine functions together
def raised_cosine_filter(data, sample_rate, f_lower, f_upper, transition_window_width, multiplier):
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
    data[k_lower : k_upper + 1] *= multiplier
    # And in mirrored spectrum
    data[N - k_upper : N - k_lower + 1] *= multiplier

    # How many bins between the start and end of the transition windows
    num_taper_bins_upper_transition = k_trans_upper - k_upper
    num_taper_bins_lower_transition = k_lower - k_trans_lower

    # Cosine window formula going smoothly from 0.0 to 1.0
    upper_transition_window = multiplier + (1 - multiplier) * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins_upper_transition)))
    lower_transition_window = multiplier + (1 - multiplier) * 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins_lower_transition)))

    # Apply transition to positive frequencies
    data[k_upper : k_trans_upper] *= upper_transition_window
    data[k_trans_lower : k_lower] *= lower_transition_window[::-1]

    # And mirrored spectrum ([::-1] reverse the transition window for the mirrored spectrum)
    data[N - k_trans_upper : N - k_upper] *= upper_transition_window[::-1]
    data[N - k_lower : N - k_trans_lower] *= lower_transition_window

    return data

def raised_cosine_boost(data, sample_rate, f_lower, f_upper, transition_window_width, multiplier):
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
    data[k_lower : k_upper + 1] *= multiplier
    # And in mirrored spectrum
    data[N - k_upper : N - k_lower + 1] *= multiplier

    # How many bins between the start and end of the transition windows
    num_taper_bins_upper_transition = k_trans_upper - k_upper
    num_taper_bins_lower_transition = k_lower - k_trans_lower

    # Cosine window formula going smoothly from 1 to multiplier
    upper_transition_window = 1 + (multiplier - 1) * (0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins_upper_transition))))
    lower_transition_window = 1 + (multiplier - 1) * (0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, num_taper_bins_lower_transition)))) 

    # Apply transition to positive frequencies
    data[k_upper : k_trans_upper] *= upper_transition_window[::-1]
    data[k_trans_lower : k_lower] *= lower_transition_window

    # And mirrored spectrum ([::-1] reverse the transition window for the mirrored spectrum)
    data[N - k_trans_upper : N - k_upper] *= upper_transition_window
    data[N - k_lower : N - k_trans_lower] *= lower_transition_window[::-1]

    return data

# make sure you include the file name extension (.wav)
file_path = "Welcome_5cm.wav"

# Extract Audio data from file
sample_rate, data, time_array, file_name = ap.read_wav_file(AUDIO_DIR / file_path)

# Calculate FFT and IFFT
complex_spectrum, frequencies, N = ap.calculate_fft(data, sample_rate)
nyquist_adjusted_spectrum = ap.get_nyquist_spectrum(complex_spectrum)

# Plot original Audio in Time Domain
ap.plot_time_spectrum(time_array, data, file_name, OUTPUT_DIR, "Original_Audio")

# Plot Original Audio in Frequency Domain
ap.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
ap.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

# Apply a filtering window to remove specific frequencies from the audio signal
filtered_complex_spectrum = complex_spectrum.copy()
#these are the settings for the 5cm sample
filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 20, 200, 400, 0)
filtered_complex_spectrum = raised_cosine_boost(complex_spectrum, sample_rate, 400, 800, 200, 2)
#and these are the settings for the 1m sample
#filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 20, 100, 20, 0)
#filtered_complex_spectrum = raised_cosine_boost(complex_spectrum, sample_rate, 120, 400, 20, 2)

# Obtain the nyquist adjusted filtered frequency spectrum for plotting
filtered_nyquist_spectrum = ap.get_nyquist_spectrum(filtered_complex_spectrum)

# Revert the complex spectrum to time domain
filtered_data = calculate_ifft(filtered_complex_spectrum)

# Plot Filtered Audio in Frequency Domain
ap.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT")
ap.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT", True, True)

# Plot Filtered Audio in Time Domain
ap.plot_time_spectrum(time_array, filtered_data, file_name, OUTPUT_DIR, "Filtered_Audio")
ap.generate_wav_file(filtered_data, sample_rate, file_name, OUTPUT_DIR, "Filtered_Audio")

print(f"Successfully processed {file_path}")