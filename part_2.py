
file_paths = ["Welcome_5cm.wav"]

for file_path in file_paths:
    if Path(AUDIO_DIR/file_path).exists():
        # Extract Audio data from file
        sample_rate, data, time_array, file_name = read_wav_file(AUDIO_DIR / file_path)
        
        # Calculate FFT and IFFT
        complex_spectrum, frequencies, N = calculate_fft(data, sample_rate)
        nyquist_adjusted_spectrum = get_nyquist_spectrum(complex_spectrum)

        # Plot original Audio in Time Domain
        plot_time_spectrum(time_array, data, sample_rate, file_name, title = "Original_Audio")
        
        # Plot Original Audio in Frequency Domain
        plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, "FFT")
        plot_frequency_spectrum(frequencies, nyquist_adjusted_spectrum, file_name, "FFT", True, True)

        # Apply a filtering window to remove specific frequencies from the audio signal
        filtered_complex_spectrum = complex_spectrum.copy()
        filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 20, 200, 400, 0)
        #filtered_complex_spectrum = raised_cosine_filter(complex_spectrum, sample_rate, 200, 400, 20, 0.1)

        # Obtain the nyquist adjusted filtered frequency spectrum for plotting
        filtered_nyquist_spectrum = get_nyquist_spectrum(filtered_complex_spectrum)

        # Revert the complex spectrum to time domain
        filtered_data = calculate_ifft(filtered_complex_spectrum)

        # Plot Filtered Audio in Frequency Domain
        plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, "IFFT")
        plot_frequency_spectrum(frequencies, filtered_nyquist_spectrum, file_name, "IFFT", True, True)
        
        # Plot Filtered Audio in Time Domain
        plot_time_spectrum(time_array, filtered_data, sample_rate, file_name, "Filtered_Audio")
        generate_wav_file(filtered_data, sample_rate, file_name, "Filtered_Audio")

        print(f"Successfully processed {file_path}")

    else:
        print(f"Error: {file_path} not found") 