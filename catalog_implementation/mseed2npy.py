import numpy as np
import obspy
from obspy import UTCDateTime
import os
import pandas as pd


def preprocess_stream(stream, output_dir=None, window_length=30, freqmin=1, freqmax=20):

    stream = trim_stream_to_common_time(stream)
    
    sampling_rate = stream[0].stats.sampling_rate
    latest_start_time = max([tr.stats.starttime for tr in stream])
    earliest_end_time = min([tr.stats.endtime for tr in stream])
    
    common_duration = earliest_end_time - latest_start_time
    n_potential_windows = int(common_duration / window_length)
    
    print(f"Processing {n_potential_windows} potential windows...")
    
    windows_good_for_all_traces = []
    
    for i in range(n_potential_windows):
        window_is_good_for_all = True
        
        # Check the window for all traces
        for tr in stream.traces:
            window_start = latest_start_time + (i * window_length)
            window_end = window_start + window_length - (1 / sampling_rate)
            
            window_trace = tr.slice(window_start, window_end)
            window_data = window_trace.data
            
            if not check_if_window_is_good(window_data, expected_length=int(window_length * sampling_rate)):
                window_is_good_for_all = False
                break
        
        if window_is_good_for_all:
            windows_good_for_all_traces.append(i)
    
    print(f"  Found {len(windows_good_for_all_traces)} windows that are good for all traces")
    
    # Process the windows that are good for all traces
    good_windows_by_trace = []
    
    for tr_idx, tr in enumerate(stream.traces):
        trace_windows = []
        
        for window_idx in windows_good_for_all_traces:
            window_start = latest_start_time + (window_idx * window_length)
            window_end = window_start + window_length - (1 / sampling_rate)
            
            window_trace = tr.slice(window_start, window_end)
            window_data = window_trace.data
            
            # Extract data if it's a masked array
            if np.ma.is_masked(window_data):
                window_data = window_data.data
            
            try:
                f = np.fft.fftfreq(len(window_data), d=1/sampling_rate)
                xw = np.fft.fft(window_data)
                mask = (np.abs(f) < freqmin) | (np.abs(f) > freqmax)
                xw[mask] = 0
                processed_window = np.real(np.fft.ifft(xw)).astype(np.float32)
                processed_window -= np.mean(processed_window)
                processed_window = normalize(processed_window, axis=0)
                
                trace_windows.append(processed_window)
                    
            except Exception as e:
                print(f"  Processing failed for window {window_idx} from trace {tr_idx}: {e}")
                raise e  # Reraise to debug issues
        
        good_windows_by_trace.append(np.array(trace_windows))
    
    # Stack into final array
    stream_numpy = np.stack(good_windows_by_trace, axis=-1)
    
    print(f"\nSummary:")
    print(f"  Started with {n_potential_windows} potential windows")
    print(f"  Kept {len(windows_good_for_all_traces)} good windows ({100*len(windows_good_for_all_traces)/n_potential_windows:.1f}%)")
    print(f"  Final shape: {stream_numpy.shape}")
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{stream[0].stats.network}.{stream[0].stats.station}.{latest_start_time.strftime('%Y%m%d_%H%M%S')}.npy"
        save_path = os.path.join(output_dir, filename)
        np.save(save_path, stream_numpy)
        print(f'  Saved to: {save_path}')
    
    # Return the indices of windows we kept
    return stream_numpy, windows_good_for_all_traces


def check_if_window_is_good(window_data, expected_length):
    """
    Checks to see if a window is usable.
    Returns True if the window is "good", False if it should be skipped.
    """
    # Check 1: Is it a masked array with masked values?
    if np.ma.is_masked(window_data):
        if window_data.mask.any():
            return False
        # If it's masked but has no actual masked values, extract the data
        window_data = window_data.data
    
    # Check 2: Does it have NaN values?
    if np.isnan(window_data).any():
        return False
    
    # Check 3: Does it have the expected length?
    if len(window_data) != expected_length:
        return False
    
    # Check 4: Is it all zeros? (indicates missing data)
    if np.all(window_data == 0):
        return False
    
    # Check 5: Does it have infinite values?
    if np.isinf(window_data).any():
        return False
    
    return True


def create_synchronized_labels(station_arrivals_path, start_time, sampling_rate, 
                                    window_indices_kept, window_length=30, 
                                    samples_per_window=3000, specific_station=None):

    station_arrivals = pd.read_csv(station_arrivals_path)
    station_arrivals['arrival_time'] = pd.to_datetime(station_arrivals['arrival_time'])
    
    if specific_station:
        station_arrivals = station_arrivals[station_arrivals['station_name'] == specific_station]
    
    n_kept_windows = len(window_indices_kept)
    precise_labels = np.zeros((n_kept_windows, samples_per_window), dtype=int)
    
    for _, arrival in station_arrivals.iterrows():
        arrival_time = UTCDateTime(arrival['arrival_time'])
        
        # Calculate which window this arrival falls into
        time_since_start = arrival_time - start_time
        window_index = int(time_since_start / window_length)
        
        # Check if this window was kept
        if window_index in window_indices_kept:
            # Find the position in our kept windows array
            kept_position = window_indices_kept.index(window_index)
            
            # Calculate sample position within the window
            time_in_window = time_since_start - (window_index * window_length)
            sample_in_window = int(time_in_window * sampling_rate)
            
            # Mark the arrival if it's within bounds
            if 0 <= sample_in_window < samples_per_window:
                precise_labels[kept_position, sample_in_window] = 1
    
    condensed_labels = (precise_labels.sum(axis=1) > 0).astype(int)
    
    print(f"\nLabel summary:")
    print(f"  Created labels for {n_kept_windows} windows")
    print(f"  Windows with earthquakes: {condensed_labels.sum()}")
    
    return precise_labels, condensed_labels


def integrate_preprocessing_and_labeling(stream, station_arrivals_path, output_dir=None):
    start_time = stream[0].stats.starttime
    sampling_rate = stream[0].stats.sampling_rate
    
    # Process the stream
    X_data, window_indices_kept = preprocess_stream(
        stream, 
        output_dir=output_dir, 
        window_length=30, 
        freqmin=1, 
        freqmax=20
    )
    
    # Create labels
    y_precise, y_condensed = create_synchronized_labels(
        station_arrivals_path, 
        start_time, 
        sampling_rate,
        window_indices_kept,
        #n_windows=X_data.shape[0],
        samples_per_window=X_data.shape[1]
    )
    
    print(f"\nFinal results:")
    print(f"  X shape: {X_data.shape}")
    print(f"  y_precise shape: {y_precise.shape}")
    print(f"  y_condensed shape: {y_condensed.shape}")
    print(f"  Everything is aligned and ready.")
    
    return X_data, y_precise, y_condensed


def trim_stream_to_common_time(stream):
    start_times = [tr.stats.starttime for tr in stream.traces]
    end_times = [tr.stats.endtime for tr in stream.traces]
    
    latest_start_time = max(start_times)
    earliest_end_time = min(end_times)
    
    trimmed_stream = stream.copy()
    for tr in trimmed_stream.traces:
        tr.trim(starttime=latest_start_time, endtime=earliest_end_time)
    
    return trimmed_stream


def normalize(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / (1e-37 + norm)