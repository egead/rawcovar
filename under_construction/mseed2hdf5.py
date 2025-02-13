# This script seperates any given mseed data into default 30 seconds waveforms. 
# This is an old script, that saves mseed files as hdf5 files. Parts of it can be integrated to our code
import obspy
from obspy import read
from obspy.core import UTCDateTime
import h5py
import numpy as np

stream = read("/home/ege/Documents/EARTH-ML/06to13Feb2023Data/Package_1718468318642.mseed")

segment_length = 30 # seconds
print(f'Preparing waveforms... {segment_length} time windows.')

with h5py.File(f'waveforms{segment_length}s.hdf5', 'w') as f:
    grp = f.create_group('data') # Create `data` group to be consistent with STEAD to reduce code labor 
    
    # Was added due to repeated name error when saving datasets. 
    # (Could also be solved if starttime was to miliseconds maybe? maybe not) Worth to look again 
    segment_number=0 
    
    for trace in stream:
        start_time = trace.stats.starttime
        end_time = trace.stats.endtime
        while start_time + segment_length <= end_time:
            segment = trace.slice(start_time, start_time + segment_length)

            # Add all segments to the `data` group as datasets with appropriate key names
            dset=grp.create_dataset(f'{trace}{segment.stats.network}{segment.stats.station}{segment.stats.channel}{segment.stats.starttime.timestamp}-{segment_number}',data=segment.data)
            start_time += segment_length
            segment_number+=1

print('Done')