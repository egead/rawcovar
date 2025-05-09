import os
import pandas as pd
from obspy import read_events

def cat2csv(nordic_file, events_output_file, stations_output_file):
    """
    Read Nordic Seisan catalog data using ObsPy and convert to CSV using pandas.
    Modified to only save the first arrival from each station.
    
    Parameters:
    -----------
    nordic_file : str
        Path to the input file in Nordic format
    events_output_file : str
        Path to output CSV file for event information
    stations_output_file : str
        Path to output CSV file for station information
    """
    print(f"Reading catalog from {nordic_file}...")
    
    catalog = read_events(nordic_file)
    
    print(f"Successfully read {len(catalog)} events from catalog")
    
    events_data = []
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0] if event.origins else None
        magnitude = event.preferred_magnitude() or event.magnitudes[0] if event.magnitudes else None
        
        if origin:
            events_data.append({
                'datetime': origin.time.datetime if origin.time else None,
                'latitude': origin.latitude,
                'longitude': origin.longitude,
                'depth': origin.depth / 1000 if origin.depth is not None else None,  # Convert m to km
                'magnitude_type': magnitude.magnitude_type if magnitude else None,
                'magnitude': magnitude.mag if magnitude else None
            })
    
    events_df = pd.DataFrame(events_data)
    events_df.to_csv(events_output_file, index=False)
    print(f"Events written to {events_output_file}")
    
    stations_data = []
    
    for event in catalog:
        origin = event.preferred_origin() or event.origins[0] if event.origins else None
        if not origin or not origin.time:
            continue
        
        event_datetime = origin.time.datetime
        
        # Track which stations we've already processed for this event
        processed_stations = set()
        
        # Process picks (arrival times)
        for pick in event.picks:
            station = pick.waveform_id.station_code if pick.waveform_id else None
            
            # Skip this station if we've already processed it (only keep first arrival)
            if station in processed_stations:
                continue
            
            processed_stations.add(station)
            
            amplitude = None
            period = None
            for amp in event.amplitudes:
                if amp.pick_id == pick.resource_id:
                    amplitude = amp.generic_amplitude
                    period = amp.period
            
            stations_data.append({
                'datetime': event_datetime,
                'station_name': station,
                'phase': pick.phase_hint,
                'arrival_time': pick.time.datetime if pick.time else None,
                'amplitude': amplitude,
                'period': period
            })
    
    stations_df = pd.DataFrame(stations_data)
    stations_df.to_csv(stations_output_file, index=False)
    print(f"Station information written to {stations_output_file} (first arrival only)")