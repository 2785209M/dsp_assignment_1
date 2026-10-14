import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import wavfile
import audio_utilities as au #audio utilities

OUTPUT_DIR = Path("part_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    file_paths = ["exciter_Welcome_5cm.wav"]

    for file_path in file_paths:
        if Path(AUDIO_DIR/file_path).exists():
            # 1. Remove noise from original signal

            # Extract Audio data from file
            sample_rate, data, time_array, file_name = au.read_wav_file(AUDIO_DIR / file_path)
            
            # Calculate FFT and IFFT
            complex_spectrum, frequencies, N = au.calculate_fft(data, sample_rate)
            nyquist_adjusted_spectrum = au.get_nyquist_spectrum(complex_spectrum)

            # Plot original Audio in Time Domain
            au.plot_time_spectrum(time_array, data, file_name, OUTPUT_DIR, "Original_Audio")
            
            # Plot Original Audio in Frequency Domain
            au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
            au.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

            # Apply a filtering window to remove specific frequencies from the audio signal
            #clear up noise and sibilance before aural exciter
            filtered_complex_spectrum = complex_spectrum.copy()
            filtered_complex_spectrum = au.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 20, 200, 400, 0)
            filtered_complex_spectrum = au.raised_cosine_boost(filtered_complex_spectrum, sample_rate, 400, 800, 200, 2)
        
            # Obtain the nyquist adjusted filtered frequency spectrum for plotting
            filtered_nyquist_spectrum = au.get_nyquist_spectrum(filtered_complex_spectrum)

            # Revert the complex spectrum to time domain
            filtered_data = au.calculate_ifft(filtered_complex_spectrum)

            # 2. Aural Exciter Sidechain

            #extract 800-6000Hz
            upper_sidechain = filtered_complex_spectrum.copy()
            upper_sidechain = au.raised_cosine_filter(upper_sidechain, sample_rate, 20, 800, 200, 0)
            upper_sidechain = au.raised_cosine_filter(upper_sidechain, sample_rate, 6000, sample_rate/2 - 200, 200, 0)

            upper_sidechain_signal = au.calculate_ifft(upper_sidechain)
            upper_sidechain_signal = au.normalise(upper_sidechain_signal)

            #scale upper sidechain to get nonlinearity from tanh
            print(np.max(upper_sidechain_signal))
            print(np.min(upper_sidechain_signal))
            alpha = 3
            upper_sidechain_signal = np.tanh(alpha * upper_sidechain_signal)

            #fft after nonlinearity applied
            exciter_spectrum, exciter_freqs, N = au.calculate_fft(upper_sidechain_signal, sample_rate)

            #filter for desired harmonics
            exciter_spectrum = au.raised_cosine_filter(exciter_spectrum, sample_rate, 0, 4000, 200, 0)
            exciter_spectrum = au.raised_cosine_filter(exciter_spectrum, sample_rate, 10000, sample_rate/2 - 200, 200, 0)

            #nyquist spectrum for plotting
            nyquist_exciter_spectrum = au.get_nyquist_spectrum(exciter_spectrum)

            exciter_harmonics = au.calculate_ifft(exciter_spectrum)

            # 3. Big Bottom Sidechain

            # #extract 50-190Hz
            # bottom_sidechain = filtered_complex_spectrum.copy()
            # bottom_sidechain = au.raised_cosine_filter(bottom_sidechain, sample_rate, 20, 40, 20, 0)
            # bottom_sidechain = au.raised_cosine_filter(bottom_sidechain, sample_rate, 200, sample_rate/2 - 200, 20, 0)

            # bottom_sidechain_signal = au.calculate_ifft(bottom_sidechain)
            # bottom_sidechain_signal = au.normalise(bottom_sidechain_signal)

            # #scale upper sidechain to get nonlinearity from tanh
            # beta = 5
            # bottom_sidechain_signal = np.tanh(beta * bottom_sidechain_signal)

            # #fft after nonlinearity applied
            # bottom_spectrum, bottom_freqs, N = au.calculate_fft(bottom_sidechain_signal, sample_rate)

            # #filter for desired harmonics
            # bottom_spectrum = au.raised_cosine_filter(bottom_spectrum, sample_rate, 20, 150, 200, 0)
            # bottom_spectrum = au.raised_cosine_filter(bottom_spectrum, sample_rate, 700, sample_rate/2 - 200, 200, 0)

            # #nyquist spectrum for plotting
            # nyquist_bottom_spectrum = au.get_nyquist_spectrum(bottom_spectrum)

            # bottom_harmonics = au.calculate_ifft(bottom_spectrum)

            # bottom_harmonics = au.normalise(bottom_harmonics)

            # 4. Mixing

            #mixer gain scaled by some fraction of the original audio
            mixer_gain_1 = np.max(np.abs(filtered_data)) * 0.4

            # mixer_gain_2 = np.max(np.abs(filtered_data)) * 0.1

            final_audio = filtered_data + mixer_gain_1 * exciter_harmonics# + mixer_gain_2 * bottom_harmonics

            final_audio = (final_audio)

            final_spectrum, final_freqs, N = au.calculate_fft(final_audio, sample_rate) 
            nyquist_final_spectrum = au.get_nyquist_spectrum(final_spectrum)

            #initial filtered audio normalised at the end to compare with the final_audio
            filtered_data = (filtered_data)

            # 5. Plots and wav files

            # Plot Filtered Audio in Frequency Domain
            au.plot_frequency_spectrum(exciter_freqs, nyquist_exciter_spectrum, file_name, OUTPUT_DIR, "FFT_exciter", True, True)
            #au.plot_frequency_spectrum(bottom_freqs, nyquist_bottom_spectrum, file_name, OUTPUT_DIR, "FFT_bottom", True, True)
            au.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT", True, True)
            au.plot_frequency_spectrum(final_freqs, nyquist_final_spectrum, file_name, OUTPUT_DIR, "FFT_final", True, True)
            
            # Plot Filtered Audio in Time Domain
            au.plot_time_spectrum(time_array, exciter_harmonics, file_name, OUTPUT_DIR, "exciter_harmonics")
            #au.plot_time_spectrum(time_array, bottom_harmonics, file_name, OUTPUT_DIR, "bottom_harmonics")
            au.plot_time_spectrum(time_array, filtered_data, file_name, OUTPUT_DIR, "Filtered_Audio")
            au.plot_time_spectrum(time_array, final_audio, file_name, OUTPUT_DIR,"excited_audio")

            # Generate wav files
            au.generate_wav_file(final_audio, sample_rate, file_name, OUTPUT_DIR,"Final_Audio")
            au.generate_wav_file(filtered_data, sample_rate, file_name, OUTPUT_DIR,"Filtered_Audio")

            print(f"Successfully processed {file_path}")

        else:
            print(f"Error: {file_path} not found") 
