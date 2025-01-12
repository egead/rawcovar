'''
Functions in this module seperates any given mseed data into default 30 seconds waveforms, 
applies preprocessing, and saves it as .npy file 

ABOUT PRE-PROCESSING: 
We aimed to make our evaluation procedures compatiblewith the review article where arange of models have been compared on various datasets.

For this purpose, we chose the model input time window to be 30 seconds, applied bandpass filtering (1 − 20Hz) 
and normalized the waveforms along the time axis to make the standard deviation of each channel equal to 1.
'''
import obspy
from obspy import read
from obspy.core import UTCDateTime
import h5py
import numpy as np


def create_time_windows(tr, start_time, end_time):

    time_windows=[]
    n_of_windows= int(end_time-start_time)/30
    for i in range(int(n_of_windows)):
        time_window=tr.slice(start_time+(i)*30,start_time + (i + 1) * 30 - (1 / tr.stats.sampling_rate)).data
        time_windows.append(time_window) 
    return time_windows

def tw_2_npy(processed_stream,stream_path,save_path):

    tw_lst=[]
    for i in range(len(processed_stream.traces)):
        i_time_windows=np.array(create_time_windows(processed_stream.traces[i], start_time=processed_stream.traces[i].stats.starttime,end_time=processed_stream.traces[i].stats.endtime))
        tw_lst.append(i_time_windows)

    stream_numpy=np.stack(tw_lst, axis=-1)

    save_path=stream_path.split('/')[-1].split('.')[-2]+'.npy'
    np.save(save_path,stream_numpy)
    print('Stream saved at: ', save_path)


def ms2np(stream_path,save_path):
    '''
    PATH Parameter Examples:
    stream_path="/home/ege/KAVV2324.mseed"
    save_path=stream_path.split('/')[-1].split('.')[-2]+'.npy'
    '''
    
    # PRE-PROCESSING 
    stream = read(stream_path)
    stream_copy=stream.copy()
    stream_copy.merge()
    print("Initial Stream:",stream_copy)
    
    #Re-create a stream with chosen traces. Uncomment the line below if you rather use all traces in your stream.
    chosen_stream = stream_copy.copy()
    #chosen_stream=obspy.Stream(traces=[stream_copy[3],stream_copy[4],stream_copy[5]])
    print("Stream chosen for preprocessing:", chosen_stream)
    
    processed_stream=chosen_stream.copy()
    processed_stream.filter("bandpass", freqmin=1,freqmax=20)
    processed_stream.normalize()
    
    # SAVE STREAM AS .npy
    tw_2_npy(processed_stream,stream_path,save_path)