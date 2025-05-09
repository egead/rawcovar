import numpy as np
import pandas as pd
import os
import obspy
from obspy import UTCDateTime

def trim_stream_to_common_time(stream):
    """
    Find the latest start time and earliest end time and 
    trim all traces in the stream to the common time range

    Args:
    stream (obspy.Stream): Input ObsPy stream
    
    Returns:
    trimmed_stream (obspy.Steam): Trimmed ObsPy stream
    """
    start_times = [tr.stats.starttime for tr in stream.traces]
    end_times = [tr.stats.endtime for tr in stream.traces]
    
    latest_start_time = max(start_times)
    earliest_end_time = min(end_times)
    print("Latest start time: ", latest_start_time)
    print("Earliest end time: ", earliest_end_time)
    
    trimmed_stream = stream.copy()
    for tr in trimmed_stream.traces:
        tr.trim(starttime=latest_start_time, endtime=earliest_end_time)
    
    return trimmed_stream

def normalize(x, axis):
    """
    Normalize array along specified axis, handling both regular and masked arrays.
    
    Args:
    x (np.ndarray): Array to be normalized.
    axis (int): Axis to be normalized.
    
    Returns:
    np.ndarray: Normalized array.
    """
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / (1e-37 + norm)

def create_synchronized_labels(station_arrivals_path, start_time, sampling_rate, 
                                window_length=30, samples_per_window=3000,
                                n_windows=None, specific_station=None):
    """
    Create earthquake labels synchronized with preprocessed X data windows.
    
    Parameters:
    -----------
    station_arrivals_path : str
        Path to station arrivals CSV
    start_time : UTCDateTime
        Start time of the preprocessed data
    sampling_rate : float
        Sampling rate of the data
    window_length : int, optional
        Length of each time window in seconds (default: 30)
    samples_per_window : int, optional
        Number of samples in each window (default: 3000)
    n_windows : int, optional
        Number of windows to create labels for
    specific_station : str, optional
        Specific station to filter arrivals
    
    Returns:
    --------
    precise_labels : np.ndarray
        2D array of precise labels for each window
    condensed_labels : np.ndarray
        1D array of binary labels for each window
    """
    station_arrivals = pd.read_csv(station_arrivals_path)
    station_arrivals['arrival_time'] = pd.to_datetime(station_arrivals['arrival_time'])
    
    if specific_station:
        station_arrivals = station_arrivals[station_arrivals['station_name'] == specific_station]
    
    precise_labels = np.zeros((n_windows, samples_per_window), dtype=int)
    condensed_labels = np.zeros(n_windows, dtype=int)
    
    sampling_rate_windows = samples_per_window / window_length
    
    #Mark the precise and condensed labels
    for _, arrival in station_arrivals.iterrows():
        arrival_time = UTCDateTime(arrival['arrival_time'])
        
        time_since_start = arrival_time - start_time
        
        #Determine which window this arrival falls into
        time_window_index = int(time_since_start // window_length)
        
        #Skip if outside the valid window range
        if time_window_index < 0 or time_window_index >= n_windows:
            continue
        
        # Calculate precise sample within the time window
        time_within_window = time_since_start % window_length
        sample_index = int(time_within_window * sampling_rate_windows)
        
        if 0 <= sample_index < samples_per_window:
            precise_labels[time_window_index, sample_index] = 1
            condensed_labels[time_window_index] = 1
    
    return precise_labels, condensed_labels

def filter_labels_by_nan_windows(X_data, y_precise, y_condensed):
    """
    Filter out windows from Y labels that correspond to NaN windows in X data.
    
    Parameters:
    -----------
    X_data : np.ndarray
        Preprocessed seismic data with potential NaN windows
    y_precise : np.ndarray
        Original precise labels for all windows
    y_condensed : np.ndarray
        Original condensed labels for all windows
    
    Returns:
    --------
    X_filtered : np.ndarray
        Filtered X data without NaN windows
    y_precise_filtered : np.ndarray
        Filtered precise labels corresponding to non-NaN windows
    y_condensed_filtered : np.ndarray
        Filtered condensed labels corresponding to non-NaN windows
    """
    nan_windows = np.isnan(X_data).any(axis=(1, 2)) #MAJOR PROBLEM HERE
    print(nan_windows)
    valid_window_mask = ~nan_windows #~ inverts the mask
    
    #Filter data by boolean masking
    X_filtered = X_data[valid_window_mask]
    y_precise_filtered = y_precise[valid_window_mask]
    y_condensed_filtered = y_condensed[valid_window_mask]
    
    return X_filtered, y_precise_filtered, y_condensed_filtered

def preprocess_stream(stream, output_dir=None, window_length=30, freqmin=1, freqmax=20):
    """
    Preprocess seismic waveform stream with advanced windowing and processing.
    
    Args:
    stream (obspy.Stream): Input ObsPy stream
    output_dir (str, optional): Directory to save numpy file
    window_length (int, optional): Length of time window in seconds
    freqmin (float, optional): Minimum frequency for bandpass filter
    freqmax (float, optional): Maximum frequency for bandpass filter
    
    Returns:
    np.ndarray: Preprocessed and windowed waveform data
    """
    stream = trim_stream_to_common_time(stream)

    sampling_rates = [tr.stats.sampling_rate for tr in stream.traces]
    if len(set(sampling_rates)) > 1:
        raise ValueError("Inconsistent sampling rates across traces")
    sampling_rate = sampling_rates[0]
    
    start_times = [tr.stats.starttime for tr in stream.traces]
    end_times = [tr.stats.endtime for tr in stream.traces]
    
    # Use the latest start time and earliest end time
    latest_start_time = max(start_times)
    earliest_end_time = min(end_times)
    
    # Compute window parameters
    common_duration = earliest_end_time - latest_start_time
    n_windows = int(common_duration / window_length)
    
    # Collect aligned windows
    aligned_windows_by_trace = []
    for tr in stream.traces:
        windows = []
        for i in range(n_windows):
            window_start = latest_start_time + (i * window_length)
            window_end = window_start + window_length - (1 / sampling_rate)
            
            window_trace = tr.slice(window_start, window_end)
            window_data = window_trace.data
            
            # Compute frequency array
            f = np.fft.fftfreq(len(window_data), d=1/sampling_rate)
            
            # Convert to Fourier domain
            xw = np.fft.fft(window_data)
            
            # Apply bandpass filtering in frequency domain
            mask = (np.abs(f) < freqmin) | (np.abs(f) > freqmax)
            xw[mask] = 0
            
            # Convert back to time domain
            processed_window = np.real(np.fft.ifft(xw)).astype(np.float32)
            
            # Demean
            processed_window -= np.mean(processed_window)
            
            # Normalize
            processed_window = normalize(processed_window, axis=0)
            
            windows.append(processed_window)
        
        aligned_windows_by_trace.append(np.array(windows))
    
    # Stack all time windows
    stream_numpy = np.stack(aligned_windows_by_trace, axis=-1)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        new_filename = (f"{stream[0].stats.network}.{stream[0].stats.station}."
                        f"{latest_start_time.strftime('%Y%m%d_%H%M%S')}.npy")
        save_path = os.path.join(output_dir, new_filename)
        np.save(save_path, stream_numpy)
        print('Stream saved at:', save_path)
    
    print(f'Shape of saved data: {stream_numpy.shape}')
    
    return stream_numpy

def integrate_preprocessing_and_labeling(stream, station_arrivals_path, output_dir=None):
    """
    Comprehensive preprocessing function that combines MiniSEED processing, 
    label creation, and NaN window filtering.
    
    Parameters:
    -----------
    stream : obspy.Stream
        Input seismic waveform stream
    station_arrivals_path : str
        Path to station arrivals CSV
    output_dir : str, optional
        Directory to save processed data
    
    Returns:
    --------
    X_data : np.ndarray
        Preprocessed and synchronized seismic data
    y_precise : np.ndarray
        Precise earthquake labels
    y_condensed : np.ndarray
        Condensed earthquake labels
    """
    start_time = stream[0].stats.starttime
    sampling_rate = stream[0].stats.sampling_rate
    
    X_data = preprocess_stream(
        stream, 
        output_dir=output_dir, 
        window_length=30, 
        freqmin=1, 
        freqmax=20
    )
    
    y_precise, y_condensed = create_synchronized_labels(
        station_arrivals_path, 
        start_time, 
        sampling_rate,
        n_windows=X_data.shape[0],
        samples_per_window=X_data.shape[1]
    )
    
    X_filtered, y_precise_filtered, y_condensed_filtered = filter_labels_by_nan_windows(
        X_data, y_precise, y_condensed
    )
    
    return X_filtered, y_precise_filtered, y_condensed_filtered