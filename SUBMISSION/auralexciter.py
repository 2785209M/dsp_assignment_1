import numpy as np
from pathlib import Path
import audioplot as ap # part 1 code
import audioenhancer as ae # part 2 code

OUTPUT_DIR = Path("2785209M-2664680S/part_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    file_paths = ["Welcome_5cm.wav"]

    for file_path in file_paths:
        if Path(file_path).exists():
            # 1. Remove noise from original signal

            # Extract Audio data from file
            sample_rate, data, time_array, file_name = ap.read_wav_file(file_path)
            
            # Calculate FFT and the positive-frequency spectrum used for plotting
            complex_spectrum = ap.calculate_fft(data)
            nyquist_adjusted_spectrum, frequencies = ap.get_nyquist_spectrum(complex_spectrum, sample_rate)

            # Plot original Audio in Time Domain
            ap.plot_time_spectrum(time_array, data, file_name, OUTPUT_DIR, "Original_Audio")
            
            # Plot Original Audio in Frequency Domain
            ap.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
            ap.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

            # Apply a filtering window to remove specific frequencies from the audio signal
            #clear up noise and sibilance before aural exciter
            filtered_complex_spectrum = ae.raised_cosine_filter(complex_spectrum, sample_rate, 20, 100, 20, 0)

            # Obtain the nyquist adjusted filtered frequency spectrum for plotting
            filtered_nyquist_spectrum, filtered_frequencies = ap.get_nyquist_spectrum(filtered_complex_spectrum, sample_rate)

            # Revert the complex spectrum to time domain
            filtered_data = ae.calculate_ifft(filtered_complex_spectrum)

            # 2. Big Bottom Sidechain

            # Boost 100-200 Hz (primary male human vocal range)
            big_bottom_spectrum = ae.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 100, 200, 20, 2)

            big_bottom_signal = ae.calculate_ifft(big_bottom_spectrum)

            # 3. Aural Exciter Sidechain

            #extract 800-6000Hz
            upper_sidechain = ae.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 20, 2000, 20, 0)
            #upper_sidechain = ae.raised_cosine_filter(upper_sidechain, sample_rate, 2100, sample_rate/2 - 20, 20, 0)

            upper_sidechain_signal = ae.calculate_ifft(upper_sidechain)

            #scale upper sidechain to get nonlinearity from tanh
            upper_sidechain_signal /= np.max(np.abs(upper_sidechain_signal))
    
            alpha = 2
            upper_sidechain_signal = np.tanh(alpha * upper_sidechain_signal)

            #fft after nonlinearity applied
            exciter_spectrum = ap.calculate_fft(upper_sidechain_signal)

            #filter for desired harmonics
            exciter_spectrum = ae.raised_cosine_filter(exciter_spectrum, sample_rate, 0, 1000, 200, 0)
            exciter_spectrum = ae.raised_cosine_filter(exciter_spectrum, sample_rate, 10000, sample_rate/2 - 200, 200, 0)

            #nyquist spectrum for plotting
            nyquist_exciter_spectrum, exciter_freqs = ap.get_nyquist_spectrum(exciter_spectrum, sample_rate)

            exciter_harmonics = ae.calculate_ifft(exciter_spectrum)

            # 4. Mixing

            #mixer gain scaled by some fraction of the original audio
            mixer_gain_1 = 0.1

            final_audio = filtered_data + mixer_gain_1 * (exciter_harmonics + big_bottom_signal)

            final_spectrum = ap.calculate_fft(final_audio)
            nyquist_final_spectrum, final_freqs = ap.get_nyquist_spectrum(final_spectrum, sample_rate)

            # 5. Plots and wav files

            # Plot Filtered Audio in Frequency Domain
            ap.plot_frequency_spectrum(exciter_freqs, nyquist_exciter_spectrum, file_name, OUTPUT_DIR, "FFT_exciter", True, True)
            ap.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT", True, True)
            ap.plot_frequency_spectrum(final_freqs, nyquist_final_spectrum, file_name, OUTPUT_DIR, "FFT_final", True, True)
            
            # Plot Filtered Audio in Time Domain
            ap.plot_time_spectrum(time_array, exciter_harmonics, file_name, OUTPUT_DIR, "exciter_harmonics")
            ap.plot_time_spectrum(time_array, filtered_data, file_name, OUTPUT_DIR, "Filtered_Audio")
            ap.plot_time_spectrum(time_array, final_audio, file_name, OUTPUT_DIR,"excited_audio")

            # Generate wav files
            ae.generate_wav_file(final_audio, sample_rate, file_name, OUTPUT_DIR,"Final_Audio")
            ae.generate_wav_file(filtered_data, sample_rate, file_name, OUTPUT_DIR,"Filtered_Audio")

            print(f"Successfully processed {file_path}")

        else:
            print(f"Error: {file_path} not found") 
