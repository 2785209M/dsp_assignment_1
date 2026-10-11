import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import spectrum_analysis as au #audio utilities

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    file_paths = ["aural-exciter-test.wav"]

    for file_path in file_paths:
        if Path(AUDIO_DIR/file_path).exists():
            # 1. Remove noise from original signal

            # Extract Audio data from file
            sample_rate, data, time_array, file_name = au.read_wav_file(AUDIO_DIR / file_path)
            
            # Calculate FFT and IFFT
            complex_spectrum, frequencies, N = au.calculate_fft(data, sample_rate)
            nyquist_adjusted_spectrum = au.get_nyquist_spectrum(complex_spectrum)

            # Plot original Audio in Time Domain
            au.plot_time_spectrum(time_array, data, sample_rate, file_name, title = "Original_Audio")
            
            # Plot Original Audio in Frequency Domain
            au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, "FFT")
            au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, "FFT", True, True)

            # Apply a filtering window to remove specific frequencies from the audio signal
            #clear up noise and sibilance before aural exciter
            filtered_complex_spectrum = complex_spectrum.copy()
            filtered_complex_spectrum = au.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 20, 80, 20, 0)
            filtered_complex_spectrum = au.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 5000, 6000, 1000, 0.2)
            filtered_complex_spectrum = au.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 10000, 22000, 1000, 0)

            # Obtain the nyquist adjusted filtered frequency spectrum for plotting
            filtered_nyquist_spectrum = au.get_nyquist_spectrum(filtered_complex_spectrum)

            # Revert the complex spectrum to time domain
            filtered_data = au.calculate_ifft(filtered_complex_spectrum)

            # 2. Aural Exciter Sidechain

            #extract 800-6000Hz
            upper_sidechain = filtered_complex_spectrum.copy()
            upper_sidechain = au.raised_cosine_filter(upper_sidechain, sample_rate, 20, 800, 20, 0)
            upper_sidechain = au.raised_cosine_filter(upper_sidechain, sample_rate, 6000, 20000, 200, 0)

            upper_sidechain_signal = au.calculate_ifft(upper_sidechain)

            #scale upper sidechain to get nonlinearity from tanh
            alpha = 400
            upper_sidechain_signal = np.tanh(alpha * upper_sidechain_signal)
            upper_sidechain_signal /= 400

            #fft after nonlinearity applied
            exciter_spectrum, exciter_freqs, N = au.calculate_fft(upper_sidechain_signal, sample_rate)

            #filter for desired harmonics
            exciter_spectrum = au.raised_cosine_filter(exciter_spectrum, sample_rate, 0, 5500, 200, 0)
            exciter_spectrum = au.raised_cosine_filter(exciter_spectrum, sample_rate, 6500, sample_rate/2, 500, 0)
            
            nyquist_exciter_spectrum = au.get_nyquist_spectrum(exciter_spectrum)

            exciter_harmonics = au.calculate_ifft(exciter_spectrum)

            exciter_harmonics = au.normalise(exciter_harmonics)

            mixer_gain_1 = 0.005
            final_audio = filtered_data + mixer_gain_1 * exciter_harmonics

            final_audio = au.normalise(final_audio)

            #initial filtered audio normalised at the end to compare with the final_audio
            filtered_data = au.normalise(filtered_data)

            # 3. Plots and wav files

            # Plot Filtered Audio in Frequency Domain
            au.plot_frequency_spectrum(exciter_freqs, nyquist_exciter_spectrum, file_name, "FFT_exciter", True, True)
            au.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, "IFFT")
            au.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, "IFFT", True, True)
            
            # Plot Filtered Audio in Time Domain
            au.plot_time_spectrum(time_array, exciter_harmonics, sample_rate, file_name, "exciter_harmonics")
            au.plot_time_spectrum(time_array, filtered_data, sample_rate, file_name, "Filtered_Audio")
            au.plot_time_spectrum(time_array, final_audio, sample_rate, file_name, "excited_audio")

            print(f"Successfully processed {file_path}")

        else:
            print(f"Error: {file_path} not found") 
