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
            
            # Calculate FFT and IFFT
            complex_spectrum, frequencies, N = ap.calculate_fft(data, sample_rate)
            nyquist_adjusted_spectrum = ap.get_nyquist_spectrum(complex_spectrum)

            # Plot original Audio in Time Domain
            ap.plot_time_spectrum(time_array, data, file_name, OUTPUT_DIR, "Original_Audio")
            
            # Plot Original Audio in Frequency Domain
            ap.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT")
            ap.plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, OUTPUT_DIR, "FFT", True, True)

            # Apply a filtering window to remove specific frequencies from the audio signal
            #clear up noise and sibilance before aural exciter
            filtered_complex_spectrum = complex_spectrum.copy()
            filtered_complex_spectrum = ae.raised_cosine_filter(filtered_complex_spectrum, sample_rate, 20, 200, 100, 0)
            filtered_complex_spectrum = ae.raised_cosine_boost(filtered_complex_spectrum, sample_rate, 300, 800, 100, 2)
        
            # Obtain the nyquist adjusted filtered frequency spectrum for plotting
            filtered_nyquist_spectrum = ap.get_nyquist_spectrum(filtered_complex_spectrum)

            # Revert the complex spectrum to time domain
            filtered_data = ae.calculate_ifft(filtered_complex_spectrum)

            # 2. Aural Exciter Sidechain

            #extract 800-6000Hz
            upper_sidechain = filtered_complex_spectrum.copy()
            upper_sidechain = ae.raised_cosine_filter(upper_sidechain, sample_rate, 20, 800, 200, 0)
            upper_sidechain = ae.raised_cosine_filter(upper_sidechain, sample_rate, 6000, sample_rate/2 - 200, 200, 0)

            upper_sidechain_signal = ae.calculate_ifft(upper_sidechain)

            #scale upper sidechain to get nonlinearity from tanh
            print(np.max(upper_sidechain_signal))
            print(np.min(upper_sidechain_signal))
            alpha = 2
            upper_sidechain_signal = np.tanh(alpha * upper_sidechain_signal)

            #fft after nonlinearity applied
            exciter_spectrum, exciter_freqs, N = ap.calculate_fft(upper_sidechain_signal, sample_rate)

            #filter for desired harmonics
            exciter_spectrum = ae.raised_cosine_filter(exciter_spectrum, sample_rate, 0, 4000, 500, 0)
            exciter_spectrum = ae.raised_cosine_filter(exciter_spectrum, sample_rate, 8000, sample_rate/2 - 200, 500, 0)

            #nyquist spectrum for plotting
            nyquist_exciter_spectrum = ap.get_nyquist_spectrum(exciter_spectrum)

            exciter_harmonics = ae.calculate_ifft(exciter_spectrum)

            # 3. Big Bottom Sidechain

            # #extract 50-190Hz
            # bottom_sidechain = filtered_complex_spectrum.copy()
            # bottom_sidechain = ae.raised_cosine_filter(bottom_sidechain, sample_rate, 20, 40, 20, 0)
            # bottom_sidechain = ae.raised_cosine_filter(bottom_sidechain, sample_rate, 200, sample_rate/2 - 200, 20, 0)


            # #scale upper sidechain to get nonlinearity from tanh
            # beta = 5
            # bottom_sidechain_signal = np.tanh(beta * bottom_sidechain_signal)

            # #fft after nonlinearity applied

            # #filter for desired harmonics
            # bottom_spectrum = ae.raised_cosine_filter(bottom_spectrum, sample_rate, 20, 150, 200, 0)
            # bottom_spectrum = ae.raised_cosine_filter(bottom_spectrum, sample_rate, 700, sample_rate/2 - 200, 200, 0)

            # #nyquist spectrum for plotting
            # nyquist_bottom_spectrum = ap.get_nyquist_spectrum(bottom_spectrum)



            # 4. Mixing

            #mixer gain scaled by some fraction of the original audio
            mixer_gain_1 = 0.15

            # mixer_gain_2 = np.max(np.abs(filtered_data)) * 0.1

            final_audio = filtered_data + mixer_gain_1 * exciter_harmonics# + mixer_gain_2 * bottom_harmonics

            final_spectrum, final_freqs, N = ap.calculate_fft(final_audio, sample_rate) 
            nyquist_final_spectrum = ap.get_nyquist_spectrum(final_spectrum)

            # 5. Plots and wav files

            # Plot Filtered Audio in Frequency Domain
            ap.plot_frequency_spectrum(exciter_freqs, nyquist_exciter_spectrum, file_name, OUTPUT_DIR, "FFT_exciter", True, True)
            #ap.plot_frequency_spectrum(bottom_freqs, nyquist_bottom_spectrum, file_name, OUTPUT_DIR, "FFT_bottom", True, True)
            ap.plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, OUTPUT_DIR, "IFFT", True, True)
            ap.plot_frequency_spectrum(final_freqs, nyquist_final_spectrum, file_name, OUTPUT_DIR, "FFT_final", True, True)
            
            # Plot Filtered Audio in Time Domain
            ap.plot_time_spectrum(time_array, exciter_harmonics, file_name, OUTPUT_DIR, "exciter_harmonics")
            #ap.plot_time_spectrum(time_array, bottom_harmonics, file_name, OUTPUT_DIR, "bottom_harmonics")
            ap.plot_time_spectrum(time_array, filtered_data, file_name, OUTPUT_DIR, "Filtered_Audio")
            ap.plot_time_spectrum(time_array, final_audio, file_name, OUTPUT_DIR,"excited_audio")

            # Generate wav files
            ae.generate_wav_file(final_audio, sample_rate, file_name, OUTPUT_DIR,"Final_Audio")
            ae.generate_wav_file(filtered_data, sample_rate, file_name, OUTPUT_DIR,"Filtered_Audio")

            print(f"Successfully processed {file_path}")

        else:
            print(f"Error: {file_path} not found") 
