import pandas as pd
import numpy as np
from obspy import UTCDateTime
import os

def create_earthquake_labels(station_arrivals_path, specific_station=None,start_time=None,end_time=None,samples_per_window=3000):
    """
    Create earthquake labels based on station arrivals with both detailed and condensed formats.
    
    Parameters:
    - station_arrivals_path: Path to station_arrivals.csv
    - specific_station: Optional. Station name to filter arrivals
    - start_time: Optional. Start time for filtering arrivals (UTC datetime string)
    - end_time: Optional. End time for filtering arrivals (UTC datetime string)
    - samples_per_window: Number of samples in each time window (default: 3000 for 30s at 100Hz)
    
    Returns:
    - precise_labels: 2D array (num_windows, samples_per_window) with precise arrival times
    - condensed_labels: 1D array (num_windows) with binary classification per window
    """
    station_arrivals = pd.read_csv(station_arrivals_path)
    station_arrivals['arrival_time'] = pd.to_datetime(station_arrivals['arrival_time'])
    
    # Filtering by station and time period
    if specific_station:
        station_arrivals = station_arrivals[station_arrivals['station_name'] == specific_station]
    
    if start_time and end_time:
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        station_arrivals = station_arrivals[
            (station_arrivals['arrival_time'] >= start_time) &
            (station_arrivals['arrival_time'] <= end_time)
        ]

    if len(station_arrivals) == 0:
        raise ValueError("No station arrivals found after filtering.")
    
    first_trace_start = UTCDateTime(station_arrivals['arrival_time'].min())
    last_trace_end = UTCDateTime(station_arrivals['arrival_time'].max())
    time_window_size = 30
    
    total_duration = last_trace_end - first_trace_start
    n_time_windows = int(total_duration // time_window_size) + 1
    
    # Create label arrays
    precise_labels = np.zeros((n_time_windows, samples_per_window), dtype=int)
    condensed_labels = np.zeros(n_time_windows, dtype=int)
    
    sampling_rate = samples_per_window / time_window_size  # samples per second
    
    # Convert station arrival times to precise sample indices
    for _, arrival in station_arrivals.iterrows():
        arrival_time = UTCDateTime(arrival['arrival_time'])
        
        # Calculate which time window this arrival falls into
        time_since_start = arrival_time - first_trace_start
        time_window_index = int(time_since_start // time_window_size)
        
        # Calculate the precise sample within the time window
        time_within_window = time_since_start % time_window_size
        sample_index = int(time_within_window * sampling_rate)
        
        # Mark the precise array
        if 0 <= time_window_index < n_time_windows and 0 <= sample_index < samples_per_window:
            precise_labels[time_window_index, sample_index] = 1
            # Also mark the condensed array for this window
            condensed_labels[time_window_index] = 1
    
    return precise_labels, condensed_labels

