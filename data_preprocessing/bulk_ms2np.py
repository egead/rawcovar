import os
from os.path import join as opj
import pandas as pd
from pathlib import Path
from mseed2npy import ms2np

folder_path='/home/boxx/Public/earthquake_model_evaluations/data/SilivriPaper_2019-09-01__2019-11-30'

prepared_waveforms_path=opj(folder_path,'prepared_waveforms')
processed_catalogs_path=opj(folder_path,'processed_catalogs')
raw_catalogs_path=opj(folder_path,'raw_catalogs')
raw_waveforms_path=opj(folder_path,'raw_waveforms')

prepared_wfs_daily_path = opj(prepared_waveforms_path,'day_by_day')
raw_wfs_daily_path = opj(raw_waveforms_path,'day_by_day')

source_path=prepared_wfs_daily_path
output_path='/home/ege/rawcovar/data/silivri/'

for root, dirs, files in os.walk(source_path):
    for filename in files: 
        file_path=opj(root,filename)

        rel_path=os.path.relpath(root,source_path)
        out_folder=opj(output_path,rel_path)

        os.makedirs(out_folder,exist_ok=True)

        base,ext=os.path.splitext(filename)
        if ext=='.mseed':
            new_filename= base+'_processed'+'.npy'
    
            outfile_path=opj(out_folder,new_filename)
    
            ms2np(file_path,outfile_path)