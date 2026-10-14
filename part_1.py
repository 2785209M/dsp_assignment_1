import audio_utilities as au
from pathlib import Path

# put the audio files you want to process into the "audio_files" directory
# The output will appear in the "images" directory
AUDIO_DIR = Path("audio_files")
OUTPUT_DIR  = Path("part_1")

# make sure you include the file name extension (.wav)
file_path = "peter-piper-1m.wav"

sample_rate, data, time_array, file_name = au.read_wav_file(f"{AUDIO_DIR}/{file_path}")
            
# Calculate FFT
complex_spectrum, frequencies, N = au.calculate_fft(data, sample_rate)
nyquist_adjusted_spectrum = au.get_nyquist_spectrum(complex_spectrum)

# Plot original Audio in Time Domain
au.plot_time_spectrum(time_array, data, sample_rate, file_name, OUTPUT_DIR, title = "Original_Audio")

# Plot Original Audio in Frequency Domain
au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

print(f"Successfully processed {file_path}")