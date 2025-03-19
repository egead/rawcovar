import obspy
from obspy import read
from obspy.core import UTCDateTime
import numpy as np
import os
'''
Functions in this module seperates any given mseed data into default 30 seconds waveforms, 
applies preprocessing, and saves it as .npy file 

ABOUT PRE-PROCESSING: 
We aimed to make our evaluation procedures compatiblewith the review article where arange of models have been compared on various datasets.

For this purpose, we chose the model input time window to be 30 seconds, applied bandpass filtering (1 − 20Hz) 
and normalized the waveforms along the time axis to make the standard deviation of each channel equal to 1.
'''

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
    Discards any incomplete windows.
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
    
    for tr_idx, tr in enumerate(processed_stream.traces):
        windows = []
        
        for i in range(n_windows):
            start = latest_start_time + i * 30
            end = start + 30 - (1 / sampling_rate)
            
            try:
                window = tr.slice(start, end).data
                
                # Only keep windows with exactly the expected number of samples
                if len(window) == expected_samples:
                    windows.append(window)
                else:
                    print(f"Skipping window {i} for trace {tr_idx}: expected {expected_samples} samples, got {len(window)}")
            except Exception as e:
                print(f"Error slicing window {i} for trace {tr_idx}: {e}")
        
        valid_windows_by_trace.append(windows)
    
    valid_window_counts = [len(windows) for windows in valid_windows_by_trace]
    min_valid_windows = min(valid_window_counts)
    
    if min_valid_windows == 0:
        print("Error: No valid complete windows found across all channels")
        return
    
    print(f"Found {min_valid_windows} complete windows across all channels")
    
    aligned_windows_by_trace = []
    for windows in valid_windows_by_trace:
        aligned_windows_by_trace.append(np.array(windows[:min_valid_windows]))
    
    #Stack all time windows using numpy, shape of the data will be [windows, samples, channels]
    stream_numpy = np.stack(aligned_windows_by_trace, axis=-1)

    os.makedirs(output_dir,exist_ok=True)
    new_filename= f"{processed_stream[0].stats.network}.{processed_stream[0].stats.station}.{latest_start_time}-{earliest_end_time}.npy"
    save_path=os.path.join(output_dir,new_filename)
    
    np.save(save_path, stream_numpy)
    print('Stream saved at:', save_path)
    print(f'Shape of saved data: {stream_numpy.shape}')

def merge_channels(file_list):
    merged_stream = obspy.Stream()

    for file_path in file_list:
        stream = obspy.read(file_path)
        merged_stream += stream

    merged_stream.merge(method=0) # Uses method 0 for merging. Look at obspy documentation for more information.
    
    return merged_stream

def ms2np(file_list, output_dir):
    stream_copy = merge_channels(file_list)

    # Resample all traces to a fixed rate if they're inconsistent 
    for tr in stream_copy:
        tr.resample(100)
    print("Initial Stream:", stream_copy)
    
    # Re-create a stream with traces that are not masked. Masked traces were present in some data.
    split_stream = stream_copy.split()
    chosen_stream = obspy.Stream(traces=[tr for tr in split_stream if not np.ma.is_masked(tr.data)])

    print("Stream chosen for preprocessing:", chosen_stream)
    
    processed_stream = chosen_stream.copy()
    processed_stream.filter("bandpass", freqmin=1, freqmax=20)
    processed_stream.normalize() 
    
    tw_2_npy(processed_stream, output_dir)