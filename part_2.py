import audio_utilities as au
from pathlib import Path

# put the audio files you want to process into the "audio_files" directory
# The output will appear in the "OUTPUT" directory
AUDIO_DIR = Path("audio_files")
OUTPUT_DIR  = Path("part_1")

# make sure you include the file name extension (.wav)
file_path = "peter-piper-1m.wav"

# Extract Audio data from file
sample_rate, data, time_array, file_name = au.read_wav_file(AUDIO_DIR / file_path)

# Calculate FFT and IFFT
complex_spectrum, frequencies, N = au.calculate_fft(data, sample_rate)
nyquist_adjusted_spectrum = au.get_nyquist_spectrum(complex_spectrum)

# Plot original Audio in Time Domain
au.plot_time_spectrum(time_array, data, sample_rate, file_name, OUTPUT_DIR, title = "Original_Audio")

# Plot Original Audio in Frequency Domain
au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

# Apply a filtering window to remove specific frequencies from the audio signal
filtered_complex_spectrum = complex_spectrum.copy()
# filter out noise, loud plosives, and loud sibilance
filtered_complex_spectrum = au.raised_cosine_filter(complex_spectrum, sample_rate, 20, 80, 20, 0)
filtered_complex_spectrum = au.raised_cosine_filter(complex_spectrum, sample_rate, 5000, 6000, 1000, 0.2)
filtered_complex_spectrum = au.raised_cosine_filter(complex_spectrum, sample_rate, 10000, sample_rate/2, 1000, 0.01)
# boost bass frequencies
filtered_complex_spectrum = au.raised_cosine_boost(complex_spectrum, sample_rate, 300, 800, 300, 2)

# Obtain the nyquist adjusted filtered frequency spectrum for plotting
filtered_nyquist_spectrum = au.get_nyquist_spectrum(filtered_complex_spectrum)

# Revert the complex spectrum to time domain
filtered_data = au.calculate_ifft(filtered_complex_spectrum)

# Plot Filtered Audio in Frequency Domain
au.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT")
au.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT", True, True)

# Plot Filtered Audio in Time Domain
au.plot_time_spectrum(time_array, filtered_data, sample_rate, file_name, OUTPUT_DIR, "Filtered_Audio")
au.generate_wav_file(filtered_data, sample_rate, file_name, OUTPUT_DIR, "Filtered_Audio")

print(f"Successfully processed {file_path}")