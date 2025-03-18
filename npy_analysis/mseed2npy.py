import obspy
from obspy import read
from obspy.core import UTCDateTime
import h5py
import numpy as np

'''
Functions in this module seperates any given mseed data into default 30 seconds waveforms, 
applies preprocessing, and saves it as .npy file 

ABOUT PRE-PROCESSING: 
We aimed to make our evaluation procedures compatiblewith the review article where arange of models have been compared on various datasets.

For this purpose, we chose the model input time window to be 30 seconds, applied bandpass filtering (1 − 20Hz) 
and normalized the waveforms along the time axis to make the standard deviation of each channel equal to 1.
'''

def create_time_windows(tr, start_time, end_time):

    time_windows=[]
    n_of_windows= int(end_time-start_time)/30

    for i in range(int(n_of_windows)):

        time_window=tr.slice(start_time+(i)*30,start_time + (i + 1) * 30 - (1 / tr.stats.sampling_rate)).data
        time_windows.append(time_window) 
        
    return time_windows

def tw_2_npy(processed_stream,save_path):
    '''
    Time Window to .NPY 
    Saves created timewindows to a .NPY file.
    '''

    tw_lst=[]
    for i in range(len(processed_stream.traces)):
        i_time_windows=np.array(create_time_windows(processed_stream.traces[i], start_time=processed_stream.traces[i].stats.starttime,end_time=processed_stream.traces[i].stats.endtime))
        tw_lst.append(i_time_windows)

    stream_numpy=np.stack(tw_lst, axis=-1)

    np.save(save_path,stream_numpy)
    print('Stream saved at: ', save_path)

def merge_channels(file_list):
    merged_stream = obspy.Stream()

    for file_path in file_list:
        stream = obspy.read(file_path)
        merged_stream += stream

    merged_stream.merge(method=0) # Uses method 0 for merging. Look at obspy documentation for more information.
    
    return merged_stream

def ms2np(file_list, save_path):
    processed_stream = merge_channels(file_list)

    # Resample all traces to a fixed rate if they’re inconsistent 
    #for tr in stream_copy:
    #    tr.resample(100)
    #print("Initial Stream:",stream_copy)
    
    # Re-create a stream with traces that are not masked. Masked traces were present in some data.
    #split_stream = stream_copy.split()
    #chosen_stream=obspy.Stream(traces=[tr for tr in split_stream if not np.ma.is_masked(tr.data)])

    #print("Stream chosen for preprocessing:", chosen_stream)
    
    processed_stream=chosen_stream.copy()
    processed_stream.filter("bandpass", freqmin=1,freqmax=20)
    processed_stream.normalize()
    
    tw_2_npy(processed_stream,save_path)