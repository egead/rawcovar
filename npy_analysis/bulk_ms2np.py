import os
from os.path import join as opj
from pathlib import Path
from mseed2npy import ms2np

folder_path='/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30'
output_path='./data/silivri/'

prepared_waveforms_path=opj(folder_path,'prepared_waveforms')
prepared_wfs_daily_path = opj(prepared_waveforms_path,'day_by_day')

source_path=prepared_wfs_daily_path
file_list = []

for root, dirs, files in os.walk(source_path):
    for filename in files: 
        if ext=='.mseed':
            file_path=opj(root,filename)

            rel_path=os.path.relpath(root,source_path)
            out_folder=opj(output_path,rel_path)
            file_list.append(file_path)

            os.makedirs(out_folder,exist_ok=True)
            new_filename= base+'_processed'+'.npy'
            save_path=opj(out_folder,new_filename)
            base,ext=os.path.splitext(filename)

            #   I need to modify ms2np so that it takes the merged_stream object. 
            # A new function that takes the file_list and output_path would suffice ? 

            ms2np(stream_path=file_path, save_path=save_path)