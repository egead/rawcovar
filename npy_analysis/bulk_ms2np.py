import os
from os.path import join as opj
from mseed2npy import ms2np

folder_path='/home/ege/rawcovar/ADVT/'
output_dir='/home/ege/rawcovar/ADVT_NPYS/'

source_path=folder_path
file_list = []

for root, dirs, files in os.walk(source_path):
    for filename in files: 
        base,ext=os.path.splitext(filename)
        if ext=='.mseed':
            file_path=opj(root,filename)
            file_list.append(file_path)

        ms2np(file_list=file_list, output_dir=output_dir)