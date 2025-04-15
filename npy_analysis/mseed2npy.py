import obspy
import numpy as np
import os

def merge_stream(stream):
    """
    Merge channels in an ObsPy stream.
    
    Args:
        stream (obspy.Stream): The input ObsPy stream
        
    Returns:
        obspy.Stream: The merged stream
    """
    # Create a copy to avoid modifying the original
    merged_stream = stream.copy()
    
    # Merge the streams
    merged_stream.merge(method=0)  # Uses method 0 for merging
    
    return merged_stream
def preprocess_stream(stream):
    """
    Preprocess the stream with proper handling of split traces
    """
    # Create a copy
    stream_copy = stream.copy()
    
    # Handle masked data
    split_stream = stream_copy.split()
    
    # Instead of keeping all split traces, group them by channel and merge them
    channels = {}
    for tr in split_stream:
        channel_id = tr.id
        if channel_id not in channels:
            channels[channel_id] = obspy.Stream()
        channels[channel_id] += tr
    
    # Merge each channel's traces
    merged_stream = obspy.Stream()
    for channel_id, channel_stream in channels.items():
        channel_stream.merge(method=0, fill_value =0)  # Merge traces for each channel
        merged_stream += channel_stream
    
    # Now we should have only 3 traces again (one per component)
    print(f"After merging split traces: {merged_stream}")
    
    # Apply filters
    processed_stream = merged_stream.copy()
    processed_stream.filter("bandpass", freqmin=1, freqmax=20)
    processed_stream.normalize()
    
    return processed_stream

def create_time_windows(tr, start_time, end_time):
    """
    Create 30-second time windows from a trace within the given time range.
    """
    time_windows = []
    n_of_windows = int(end_time-start_time)/30

    for i in range(int(n_of_windows)):
        time_window = tr.slice(start_time+(i)*30, start_time + (i + 1) * 30 - (1 / tr.stats.sampling_rate)).data
        time_windows.append(time_window) 
        
    return time_windows

def tw_2_npy(processed_stream, output_dir):
    '''
    Time Window to .NPY 
    Saves created timewindows to a .NPY file.
    Uses the largest common time range across all channels.
    Discards any incomplete windows or windows with NaN values.
    '''
    latest_start_time = max(tr.stats.starttime for tr in processed_stream.traces)
    earliest_end_time = min(tr.stats.endtime for tr in processed_stream.traces)
    
    if earliest_end_time <= latest_start_time:
        print("Error: No common time range across channels")
        return
    
    print(f"Common time range: {latest_start_time} to {earliest_end_time}")
    
    common_duration = earliest_end_time - latest_start_time
    n_windows = int(common_duration / 30)
    
    if n_windows <= 0:
        print("Error: Common time range too short for 30-second windows")
        return
    
    print(f"Processing {n_windows} potential time windows of 30 seconds each")
    
    sampling_rate = processed_stream.traces[0].stats.sampling_rate
    expected_samples = int(30 * sampling_rate)
    
    valid_windows_by_trace = []
    valid_window_indices = []
    
    # First pass: collect all windows for each trace
    for tr_idx, tr in enumerate(processed_stream.traces):
        windows = []
        window_indices = []
        
        for i in range(n_windows):
            start = latest_start_time + i * 30
            end = start + 30 - (1 / sampling_rate)
            
            try:
                window = tr.slice(start, end).data
                
                # Check if window is the expected length and has no NaN values
                if len(window) == expected_samples and not np.isnan(window).any():
                    windows.append(window)
                    window_indices.append(i)
                else:
                    if len(window) != expected_samples:
                        print(f"Skipping window {i} for trace {tr_idx}: expected {expected_samples} samples, got {len(window)}")
                    if np.isnan(window).any():
                        print(f"Skipping window {i} for trace {tr_idx}: contains NaN values")
            except Exception as e:
                print(f"Error slicing window {i} for trace {tr_idx}: {e}")
        
        valid_windows_by_trace.append(windows)
        valid_window_indices.append(set(window_indices))
    
    # Find common valid window indices across all traces
    if not valid_window_indices:
        print("Error: No valid windows found")
        return
        
    common_valid_indices = set.intersection(*valid_window_indices) if valid_window_indices else set()
    
    if not common_valid_indices:
        print("Error: No common valid windows across all channels")
        return
    
    print(f"Found {len(common_valid_indices)} complete windows with no NaN values across all channels")
    
    # Convert to sorted list
    common_valid_indices = sorted(list(common_valid_indices))
    
    # Second pass: build aligned windows using only common valid indices
    aligned_windows_by_trace = []
    
    for tr_idx, tr in enumerate(processed_stream.traces):
        windows = []
        
        for i in common_valid_indices:
            start = latest_start_time + i * 30
            end = start + 30 - (1 / sampling_rate)
            window = tr.slice(start, end).data
            windows.append(window)
        
        aligned_windows_by_trace.append(np.array(windows))
    
    # Stack all time windows using numpy, shape of the data will be [windows, samples, channels]
    stream_numpy = np.stack(aligned_windows_by_trace, axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    new_filename = f"{processed_stream[0].stats.network}.{processed_stream[0].stats.station}.{latest_start_time}-{earliest_end_time}.npy"
    save_path = os.path.join(output_dir, new_filename)
    
    np.save(save_path, stream_numpy)
    print('Stream saved at:', save_path)
    print(f'Shape of saved data: {stream_numpy.shape}')
def stream2np(stream, output_dir):
    """
    Convert an ObsPy stream directly to numpy arrays of 30-second time windows
    
    Args:
        stream (obspy.Stream): The input ObsPy stream
        output_dir (str): Directory to save the numpy files
    """
    # First check for NaNs in the original stream
    print("Checking original stream for NaNs:")
    check_stream_for_nans(stream)
    
    # Process the stream
    processed_stream = preprocess_stream(stream)
    
    # Check processed stream for common time range before proceeding
    start_times = [tr.stats.starttime for tr in processed_stream]
    end_times = [tr.stats.endtime for tr in processed_stream]
    
    latest_start = max(start_times)
    earliest_end = min(end_times)
    
    if earliest_end <= latest_start:
        print("Error: No common time range after processing")
        print(f"Latest start: {latest_start}")
        print(f"Earliest end: {earliest_end}")
        # Try to trim the stream to the time range of the component with the shortest duration
        min_duration_idx = np.argmin([(end - start) for start, end in zip(start_times, end_times)])
        min_start = start_times[min_duration_idx]
        min_end = end_times[min_duration_idx]
        print(f"Attempting to trim all traces to: {min_start} - {min_end}")
        processed_stream.trim(min_start, min_end)
    
    # Convert to numpy and save
    tw_2_npy(processed_stream, output_dir)


def check_stream_for_nans(stream):
    """
    Check an ObsPy stream for NaN values trace by trace
    
    Args:
        stream (obspy.Stream): The input ObsPy stream
        
    Returns:
        bool: True if NaNs are found, False otherwise
    """
    has_nans = False
    
    for i, tr in enumerate(stream):
        nan_count = np.isnan(tr.data).sum()
        if nan_count > 0:
            print(f"Trace {i}: {tr.id} has {nan_count} NaN values")
            has_nans = True
        else:
            print(f"Trace {i}: {tr.id} has NO NaN values")
    
    if not has_nans:
        print("No NaN values found in any trace of the stream")
    
    return has_nans