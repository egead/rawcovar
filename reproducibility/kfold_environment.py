from config import (
    BATCH_SIZE,
    KFOLD_SPLITS,
    DATASET_CHUNKS,
    INSTANCE_TIME_WINDOW,
    STEAD_TIME_WINDOW,
    RAW_TIME_WINDOW,
    SAMPLING_FREQ,
    FREQMIN,
    FREQMAX,
    TRAIN_VALIDATION_SPLIT,
    SUBSAMPLING_FACTOR,
    PHASE_PICK_ENSURED_CROP_RATIO,
    PHASE_ENSURING_MARGIN,
)
from directory import (
    STEAD_WAVEFORMS_HDF5_PATH,
    STEAD_METADATA_CSV_PATH,
    INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
    INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
    INSTANCE_EQ_METADATA_CSV_PATH,
    INSTANCE_NOISE_METADATA_CSV_PATH,
    RAW_WAVEFORMS_HDF5_PATH,
    RAW_WAVEFORMS_MSEED_DIR,
    RAW_WAVEFORMS_METADATA_CSV_DIR, 
    PREPROCESSED_DATASET_DIRECTORY,
)

from data_generator import (
    DataGenerator,
    GeneratorWrapper,
    PredictGenerator,
)
from os.path import join, exists
from os import makedirs
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import KFold
import obspy
import h5py
import os

class KFoldEnvironment:
    """
    This class is used to create the environment for training, testing and validation.
    It creates the chunk dataframes and the generators.

    Parameters
    ----------
    dataset : str
        The dataset. It can be either "stead" or "instance".
    preprocessed_dataset_directory : str
        The directory where the preprocessed data is saved.
    batch_size : int
        The batch size.
    stead_time_window : float
        The time window of the STEAD dataset.
    instance_time_window : float
        The time window of the instance dataset.
    raw_time_window : float
        The time window of the raw dataset.
    stead_waveforms_hdf5 : str
        The path of the hdf5 file for the STEAD dataset that contains the waveforms of the events.
    stead_metadata_csv : str
        The path of the csv file for the STEAD dataset that contains the metadata of the events.
    instance_eq_waveforms_hdf5 : str
        The path of the hdf5 file for the instance dataset that contains the waveforms of the events.
    instance_no_waveforms_hdf5 : str
        The path of the hdf5 file for the instance dataset that contains the waveforms of the noise.
    raw_waveforms_hdf5 : str
        The path of the hdf5 file for the raw dataset that contains the waveforms of the events.
    instance_eq_metadata_csv : str
        The path of the csv file for the instance dataset that contains the metadata of the events.
    instance_no_metadata_csv : str
        The path of the csv file for the instance dataset that contains the metadata of the noise.
    model_time_window : float
        The time window of the model.
    phase_ensured_crop_ratio : float
        Ratio of the samples which at least one pick(P or S) is ensured to be included in the window.
        This ratio is used just for training.
    phase_ensuring_margin : float
        Margin(in seconds) which is used to ensure that the picks are included in the window.
    n_splits : int
        The number of splits.
    n_chunks : int
        The number of chunks.
    subsampling_factor : float
        The subsampling factor.
    sampling_freq : float
        The sampling frequency.
    train_val_ratio : float
        The ratio of the training set size to the validation set size.
    freqmin : float
        The lower frequency of the bandpass filter.
    freqmax : float
        The upper frequency of the bandpass filter.
    """

    def __init__(
        self,
        dataset,
        preprocessed_dataset_directory=PREPROCESSED_DATASET_DIRECTORY,
        batch_size=BATCH_SIZE,
        stead_time_window=STEAD_TIME_WINDOW,
        instance_time_window=INSTANCE_TIME_WINDOW,
        raw_time_window=RAW_TIME_WINDOW,
        stead_waveforms_hdf5=STEAD_WAVEFORMS_HDF5_PATH,
        stead_metadata_csv=STEAD_METADATA_CSV_PATH,
        instance_eq_waveforms_hdf5=INSTANCE_EQ_WAVEFORMS_HDF5_PATH,
        instance_no_waveforms_hdf5=INSTANCE_NOISE_WAVEFORMS_HDF5_PATH,
        instance_eq_metadata_csv=INSTANCE_EQ_METADATA_CSV_PATH,
        instance_no_metadata_csv=INSTANCE_NOISE_METADATA_CSV_PATH,
        raw_waveforms_hdf5=RAW_WAVEFORMS_HDF5_PATH,
        raw_waveforms_mseed=RAW_WAVEFORMS_MSEED_DIR,
        model_time_window=30.0,
        phase_ensured_crop_ratio=PHASE_PICK_ENSURED_CROP_RATIO,
        phase_ensuring_margin=PHASE_ENSURING_MARGIN,
        n_splits=KFOLD_SPLITS,
        n_chunks=DATASET_CHUNKS,
        subsampling_factor=SUBSAMPLING_FACTOR,
        sampling_freq=SAMPLING_FREQ,
        train_val_ratio=TRAIN_VALIDATION_SPLIT,
        freqmin=FREQMIN,
        freqmax=FREQMAX,
    ):
        self.preprocessed_dataset_directory = preprocessed_dataset_directory
        self.model_time_window = model_time_window
        self.phase_ensured_crop_ratio = phase_ensured_crop_ratio
        self.phase_ensuring_margin = phase_ensuring_margin
        self.stead_time_window = stead_time_window
        self.instance_time_window = instance_time_window
        self.raw_time_window=raw_time_window
        self.subsampling_factor = subsampling_factor
        self.sampling_freq = sampling_freq
        self.train_val_ratio = train_val_ratio
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.n_chunks = n_chunks
        self._batch_size = batch_size
        self._n_splits = n_splits
        self._dataset = dataset

        if dataset == "stead":
            metadata = self._parse_stead_metadata(stead_metadata_csv)

            self.eq_hdf5_path = stead_waveforms_hdf5
            self.no_hdf5_path = stead_waveforms_hdf5
            self.raw_hdf5_path = None
            self.last_axis = "channels"
            self.dataset_time_window = self.stead_time_window

        if dataset == "instance":
            metadata = self._parse_instance_metadata(
                instance_eq_metadata_csv, instance_no_metadata_csv
            )

            self.eq_hdf5_path = instance_eq_waveforms_hdf5
            self.no_hdf5_path = instance_no_waveforms_hdf5
            self.raw_hdf5_path = None
            self.last_axis = "timesteps"
            self.dataset_time_window = self.instance_time_window
        
        if dataset == "raw":
            self.dataset_type = "raw"
            self.eq_hdf5_path = None
            self.no_hdf5_path = None
            self.raw_hdf5_path = self._create_raw_hdf5(raw_waveforms_mseed)
            metadata = self._parse_raw_metadata(self.raw_hdf5_path)
            self.last_axis = "timesteps"
            self.dataset_time_window = self.raw_time_window
        

        # This function returns two list of lists. Each list is a chunk list for a split.
        # First list is for training and validation while the second list is for testing.
        train_and_val_split_chunks, test_split_chunks = self._form_kfold_splits()

        # Seperate train and validation splits.
        (
            self.train_splits,
            self.validation_splits,
        ) = self._seperate_train_and_validation_chunks(train_and_val_split_chunks)

        self.test_splits = test_split_chunks

        # Split the dataset into chunks.
        chunk_metadata_list = self._split_dataset_to_chunks(metadata, "source_id")

        # Subsample the chunks.
        chunk_metadata_list = self._subsample_chunk_metadata(chunk_metadata_list)

        # Make the chunk dataframe lengths multiple of batch size by cropping the ends.
        chunk_metadata_list = self._make_chunk_metadata_multiple_of_batch_size(
            chunk_metadata_list
        )

        # Assign crop offsets to metadata.
        chunk_metadata_list = [
            self._assign_crop_offsets(chunk_metadata)
            for chunk_metadata in chunk_metadata_list
        ]

        # Assign chunk indexes to waveforms.
        chunk_metadata_list = [
            self._assign_chunk_idx(chunk_metadata, chunk_idx)
            for chunk_idx, chunk_metadata in enumerate(chunk_metadata_list)
        ]

        # Saves the chunk dataframes.
        makedirs(join(self.preprocessed_dataset_directory, self.dataset), exist_ok=True)
        metadata_path = join(
            self.preprocessed_dataset_directory, self.dataset, "metadata.csv"
        )
        if not exists(metadata_path):
            pd.concat(chunk_metadata_list).to_csv(metadata_path)

        # Sets the chunk_metadata_list variable.
        self.chunk_metadata_list = chunk_metadata_list

    @property
    def dataset(self):
        """
        Used dataset. It can be either "stead" or "instance".
        """
        return self._dataset

    @property
    def n_splits(self):
        """
        The number of splits.
        """
        return self._n_splits

    @property
    def batch_size(self):
        """
        The batch size.
        """
        return self._batch_size

    def get_generators(self, split):
        """
        Returns the training generator for a specific split.

        Parameters
        ----------
        split : int
            The split index.

        Returns
        -------
        TrainingGenerator
            The training generator.
        """
        # Get datagenerator objects.
        train_datagen = self._get_datagen(self.train_splits[split])
        validation_datagen = self._get_datagen(self.validation_splits[split])
        test_datagen = self._get_datagen(self.test_splits[split])
        predict_datagen = self._get_datagen(self.test_splits[split])

        train_gen = GeneratorWrapper(train_datagen)
        validation_gen = GeneratorWrapper(validation_datagen)
        test_gen = GeneratorWrapper(test_datagen)
        predict_gen = PredictGenerator(predict_datagen)

        return train_gen, validation_gen, test_gen, predict_gen

    def get_split_metadata(self, split):
        """
        Returns the train, validation and test metadata for a specific split.

        Parameters
        ----------
        split : int
            The split index.

        Returns
        -------
        train_metadata : pandas.DataFrame
            Metadata of the training data in proper order for a split.
        validation_metada : pandas.DataFrame
            Metadata of the validation data in proper order for a split.
        test_metadata : pandas.DataFrame
            Metadata of the test data in proper order for a split.
        """
        train_metadata = self.get_chunklist_metadata(self.train_splits[split])
        validation_metadata = self.get_chunklist_metadata(self.validation_splits[split])
        test_metadata = self.get_chunklist_metadata(self.test_splits[split])

        return train_metadata, validation_metadata, test_metadata

    def get_batch_metadata(self, split, operation, batch_idx):
        """
        Returns the metadata of a batch.

        Parameters
        ----------
        split : int
            The split index.
        operation : str
            The operation. It can be either "train", "validation" or "test".
        batch_idx : int
            The batch index.

        Returns
        -------
        df : pandas.DataFrame
            Metadata of the batch.
        """
        if operation == "train":
            split_chunks = self.train_splits[split]
        elif operation == "validation":
            split_chunks = self.validation_splits[split]
        elif operation == "test":
            split_chunks = self.test_splits[split]

        datagen = self._get_datagen(split_chunks)

        chunk_idx, batch_offset = datagen.get_chunk_idx_and_batch_offset(batch_idx)
        return self.chunk_metadata_list[chunk_idx].iloc[
            batch_offset * self.batch_size : (batch_offset + 1) * self.batch_size
        ]

    def get_chunklist_metadata(self, chunks):
        """
        Returns the dataframe of a list of chunks.

        Parameters
        ----------
        chunks : list
            A list of chunks.

        Returns
        -------
        metadata : pandas.DataFrame
            Merged dataframe of the chunks.
        """
        metadata_list = []
        for chunk in chunks:
            metadata_list.append(self.chunk_metadata_list[chunk])

        return pd.concat(metadata_list)

    def _split_dataset_to_chunks(self, dataset_metadata, colname, random_state=0):
        """
        Splits the dataset metadata into chunks based on the unique values of the column named colname.

        :param dataset_metadata: The dataframe.
        :param colname: The column name.
        :param n_chunks: The number of chunks.
        :param random_state: The random state.
        :return: A list of dataframes. Each dataframe will be used to create a chunk.
        """
        unique_vals = list(set(list(dataset_metadata[colname].values)))
        unique_vals.sort()

        # Chunks are formed both randomly but also deterministically.
        random.seed(random_state)
        sampling_size = len(unique_vals) // self.n_chunks

        # Form chunks from the unique values. The last chunk will be smaller than the rest.
        chunks = []
        for i in range(self.n_chunks):
            chunks.append(
                random.sample(unique_vals, min(sampling_size, len(unique_vals)))
            )
            unique_vals = list(set(unique_vals) - set(chunks[-1]))
            unique_vals.sort()

        chunk_metadata_list = []
        for chunk in chunks:
            chunk_metadata_list.append(
                dataset_metadata[dataset_metadata[colname].isin(chunk)]
            )

        return chunk_metadata_list

    def _form_kfold_splits(self, random_state=0):
        """
        Splits the chunks into folds. The folds are returned in a list and each fold
        contains indexes of the chunks for training or testing.

        :param random_state: The random state.
        :return: Two lists containing lists of chunk indexes. The first list is for training
            and the second list is for testing.
        """

        kfold = KFold(shuffle=True, random_state=random_state, n_splits=self.n_splits)
        chunks = range(self.n_chunks)
        kfold_gen = kfold.split(chunks)

        training_chunk_idxs_for_each_split = []
        testing_chunk_idxs_for_each_split = []

        for train_chunk_idxs, test_chunks_idx in kfold_gen:
            training_chunk_idxs_for_each_split.append(train_chunk_idxs)
            testing_chunk_idxs_for_each_split.append(test_chunks_idx)

        return training_chunk_idxs_for_each_split, testing_chunk_idxs_for_each_split

    def _parse_stead_metadata(self, metadata_csv):
        """
        Parses the metadata of the STEAD dataset and transforms certain columns in a
        way to enable generic handling.

        Parameters
        ----------
        metadata_csv : str
            The path of the csv file for the STEAD dataset that contains the metadata of the events.

        Returns
        -------
        metadata : pandas.DataFrame
            The dataframe that contains standardized metadata of the STEAD dataset.
        """
        metadata = pd.read_csv(metadata_csv)

        metadata["source_id"] = metadata["source_id"].astype(str)
        metadata.rename({"receiver_code": "station_name"})

        eq_metadata = metadata[metadata.trace_category == "earthquake_local"].copy()
        no_metadata = metadata[metadata.trace_category == "noise"].copy()

        # You need to merge eq and no samples to form chunks. Because of that you
        # need to put a column named source_id to noise waveforms. Since
        # trace_name is unique for each trace, you can use it as source_id.
        no_metadata["source_id"] = no_metadata["trace_name"]

        # Label samples.
        eq_metadata.loc[:, "label"] = "eq"
        no_metadata.loc[:, "label"] = "no"

        standardized_metadata = pd.concat([eq_metadata, no_metadata])

        return standardized_metadata

    def _parse_instance_metadata(self, eq_metadata_csv, no_metadata_csv):
        """
        Parses the metadata of the instance dataset and transforms certain columns in a
        way to enable generic handling.

        Parameters
        ----------
        eq_metadata_csv : str
            The path of the csv file for the instance dataset that contains the metadata of the events.
        no_metadata_csv : str
            The path of the csv file for the instance dataset that contains the metadata of the noise.

        Returns
        -------
        metadata : pandas.DataFrame
            The dataframe that contains standardized metadata of the instance dataset.
        """
        eq_metadata = pd.read_csv(eq_metadata_csv)
        no_metadata = pd.read_csv(no_metadata_csv)

        eq_metadata["source_id"] = eq_metadata["source_id"].astype(str)
        eq_metadata.rename(
            columns={
                "station_code": "station_name",
                "trace_P_arrival_sample": "p_arrival_sample",
                "trace_S_arrival_sample": "s_arrival_sample",
            },
            inplace=True,
        )
        no_metadata.rename(columns={"station_code": "station_name"}, inplace=True)

        # Source id is not unique for noise waveforms. Instead of using
        # source id, you can use trace_name as source_id.
        no_metadata["source_id"] = no_metadata["trace_name"]

        # Label samples.
        eq_metadata["label"] = "eq"
        no_metadata["label"] = "no"

        standardized_metadata = pd.concat([eq_metadata, no_metadata])
        return standardized_metadata
        
    def _create_raw_hdf5(self, output_dir):
        '''
        Checks for and creates an HDF5 file for raw data if it doesn't already exist.
        '''
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # This naming convention is temporary, and will be updated with a more robust way.
        output_file = os.path.join(output_dir, "raw_data.hdf5")
        
        # Might need to revisit this part of the code!
        #if not os.path.isfile(output_file):
        self._convert_mseed_to_hdf5(
            output_file=output_file,
            segment_length=self.raw_time_window
        )
        return output_file

    def _convert_mseed_to_hdf5(self, output_file, segment_length):
        '''
        Converts mseed files to HDF5
        '''
        with h5py.File(output_file,'w') as hdf:
            data_grp = hdf.create_group('data')
            mseed_files=self._gather_mseed_files(directory=RAW_WAVEFORMS_MSEED_DIR)
            print(f"Found {len(mseed_files)} mseed files to process.")
            for mseed in mseed_files: 
                st = obspy.read(mseed)
                for tr in st:
                    self._process_trace(tr, data_grp, segment_length)
    
    def _gather_mseed_files(self, directory):
        mseed_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mseed'):
                    mseed_files.append(os.path.join(root, file))
        return mseed_files

    def _process_trace(self, tr, data_grp, segment_length):
        start_time = tr.stats.starttime
        while start_time + segment_length <= tr.stats.endtime:
            end_time = start_time + segment_length
            segment = tr.slice(start_time,end_time)

            # More forgiving approach
            expected_points = int(segment_length * tr.stats.sampling_rate)
            actual_points = segment.stats.npts
            if actual_points >= 0.98 * expected_points and actual_points <= 1.02 * expected_points:
            # Unforgiving approach (causes problems)
            #if segment.stats.npts == int(segment_length * tr.stats.sampling_rate):
                seg_id = f"{tr.id}_{start_time.timestamp:.0f}"
                self._store_segment(segment, data_grp, seg_id)
                
            start_time += segment_length

    def _store_segment(self, segment, parent_group, seg_id):
        grp = parent_group.create_group(seg_id)
        grp.create_dataset('data', data=segment.data.astype('float32'))

        segment_length= segment.stats.npts / segment.stats.sampling_rate
        grp.attrs.update({
            'trace_name': seg_id,
            'source_id': seg_id,
            'station_name': segment.stats.station,
            'network': segment.stats.network,
            'channel': segment.stats.channel,
            'starttime': segment.stats.starttime.timestamp,
            'endtime': (segment.stats.starttime + segment.stats.npts/segment.stats.sampling_rate).timestamp,
            'sampling_rate': segment.stats.sampling_rate,
            'p_arrival_sample': -1,
            's_arrival_sample': -1,
            'segment_length': segment_length,
            'label': 'raw'
        })

    def _parse_raw_metadata(self,raw_hdf5_path):
        # To be updated
        """
        Parse metadata from raw file
            
        """
        metadata = []
        with h5py.File(raw_hdf5_path,'r') as hdf: 
            for seg_id in hdf['data']:
                seg_grp = hdf[f'data/{seg_id}']
                metadata.append({
                'trace_name': seg_id,
                'source_id':seg_id,
                'station_name': seg_grp.attrs.get('station_name', 'UNKNOWN'),
                'network': seg_grp.attrs.get('network', 'UNKNOWN'),
                'channel': seg_grp.attrs.get('channel', 'UNKNOWN'),
                'starttime': seg_grp.attrs.get('starttime', 0.0),
                'endtime': seg_grp.attrs.get('endtime', 0.0),
                'sampling_rate': seg_grp.attrs.get('sampling_rate', 0.0),
                'segment_length': seg_grp.attrs.get('segment_length', 0.0),
                'p_arrival_sample': seg_grp.attrs.get('p_arrival_sample', -1),
                's_arrival_sample': seg_grp.attrs.get('s_arrival_sample', -1),
                'label': seg_grp.attrs.get('label','raw')
            })

        if not metadata:
            raise ValueError(f"No metadata found in path: {raw_hdf5_path}")
        
        return pd.DataFrame(metadata)


    def _make_chunk_metadata_multiple_of_batch_size(self, chunk_metadata_list):
        """
        Makes the chunk metadata lengths multiple of batch size by cropping leap rows.

        Parameters
        ----------
        chunk_metadata_list : list
            A list of chunk dataframes.

        Returns
        -------
        cropped_chunk_metadata_list : list
            A list of cropped chunk dataframes.
        """

        cropped_chunk_metadata_list = []
        for chunk_metadata in chunk_metadata_list:
            cropped_chunk_metadata_list.append(
                chunk_metadata[
                    0 : ((len(chunk_metadata) // self.batch_size) * self.batch_size)
                ]
            )

        return cropped_chunk_metadata_list

    def _get_datagen(self, active_chunks=None):
        """
        Initializes the environment for training, testing and validation.

        Parameters
        ----------
        active_chunks : list
            A list of chunk indexes that will be used to create the datagenerator.

        Returns
        -------
        datagen : DataGenerator
            The datagenerator.
        """
        # Creates the directory for the preprocessed dataset if not exists.
        processed_hdf5_dir = join(
            PREPROCESSED_DATASET_DIRECTORY,
            self.dataset,
        )
        makedirs(processed_hdf5_dir, exist_ok=True)

        # Creates the path of the preprocessed dataset.
        processed_hdf5_path = join(
            processed_hdf5_dir,
            "subsampled_{}percent.hdf5".format(int(100 * self.subsampling_factor)),
        )

        common_params = {
            'processed_hdf5_path': processed_hdf5_path,
            'chunk_metadata_list': self.chunk_metadata_list,
            'batch_size': self.batch_size,
            'dataset_time_window': self.dataset_time_window,
            'model_time_window': self.model_time_window,
            'phase_ensured_crop_ratio': self.phase_ensured_crop_ratio,
            'last_axis': self.last_axis,
            'sampling_freq': self.sampling_freq,
            'active_chunks': active_chunks,
            'freqmin': self.freqmin,
            'freqmax': self.freqmax,
        }

        if self._dataset == "raw":
            common_params['raw_hdf5_path'] = self.raw_hdf5_path
        else:
            common_params['eq_hdf5_path'] = self.eq_hdf5_path
            common_params['no_hdf5_path'] = self.no_hdf5_path

        datagen = DataGenerator(**common_params)
        return datagen

    def _seperate_train_and_validation_chunks(self, chunk_metadata_list):
        """
        Seperates the chunk dataframes into train and validation dataframes.

        Parameters
        ----------
        chunk_metadata_list : list
            A list of chunk dataframes.

        Returns
        -------
        train_chunk_dfs : list
            A list of train chunk dataframes.
        validation_chunk_dfs : list
            A list of validation chunk dataframes.
        """
        train_chunk_dfs = []
        validation_chunk_dfs = []
        random.seed(0)

        for i in range(len(chunk_metadata_list)):
            n_train_chunks = int(len(chunk_metadata_list[i]) * self.train_val_ratio)

            random.shuffle(chunk_metadata_list[i])
            train_chunk_dfs.append(chunk_metadata_list[i][0:n_train_chunks])
            validation_chunk_dfs.append(chunk_metadata_list[i][n_train_chunks:])

        return train_chunk_dfs, validation_chunk_dfs

    def _subsample_chunk_metadata(self, chunk_metadata_list):
        """
        Subsamples the chunk dataframes.

        Parameters
        ----------
        chunk_metadata_list : list
            A list of chunk dataframes.

        Returns
        -------
        list
            A list of subsampled chunk dataframes.
        """
        subsampled_chunk_metadata_list = []

        for chunk_metadata in chunk_metadata_list:
            subsampled_chunk_metadata_list.append(
                chunk_metadata.sample(frac=self.subsampling_factor, random_state=0)
            )

        return subsampled_chunk_metadata_list

    def _assign_chunk_idx(self, metadata, chunk_idx):
        """
        Args:
            metadata (pd.DataFrame): Dataframe that contains the metadata of the waveforms.
            chunk_idx (int): The index of the chunk.

        Returns:
            pd.DataFrame: Dataframe that contains the metadata of the waveforms with chunk indexes
                assigned.
        """

        metadata["chunk_idx"] = chunk_idx

        return metadata

    def _assign_crop_offsets(self, metadata):
        # To add: RAW DATA METADATA IMPLEMENTATION 
        """
        Args:
            metadata (pd.DataFrame): Dataframe that contains the metadata of the waveforms.

        Returns:
            pd.DataFrame: Dataframe that contains the metadata of the waveforms with crop offsets
                assigned.
        """
        metadata.reset_index(drop=True, inplace=True)

        # Assign crop offset hard limits.
        metadata["crop_offset_low_limit"] = 0
        metadata["crop_offset_high_limit"] = self._get_ts(
            self.dataset_time_window
        ) - self._get_ts(self.model_time_window)

        # Assign crop offset min and max limits.
        metadata["crop_offset_min"] = metadata["crop_offset_low_limit"]
        metadata["crop_offset_max"] = metadata["crop_offset_high_limit"]

        # Assign random crop offsets to waveforms. Since eq and no waveforms
        # need to be seperately handled, we first seperate them.
        metadata_eq = metadata[metadata.label == "eq"]
        metadata_no = metadata[metadata.label == "no"]

        # Get number of pick ensured waveforms.
        n_pick_ensured_eq_waveforms = int(
            self.phase_ensured_crop_ratio * len(metadata_eq)
        )

        # Seperate eq waveforms to two groups. First group is ensured to include
        # the phase arrival times. Second group is not ensured to include the
        # phase arrival times.
        metadata_eq_pick_ensured_crop = metadata_eq[0:n_pick_ensured_eq_waveforms]
        metadata_eq_random_crop = metadata_eq[n_pick_ensured_eq_waveforms:]

        # Assign max limits to crop offsets for pick ensured waveforms.
        metadata_eq_pick_ensured_crop = self._assign_pick_ensured_crop_offset_ranges(
            metadata_eq_pick_ensured_crop
        )

        # Merge eq and no waveforms.
        metadata = pd.concat(
            [metadata_eq_pick_ensured_crop, metadata_eq_random_crop, metadata_no],
            axis=0,
        )

        # Assign random crop offsets.
        np.random.seed(0)
        metadata["crop_offset"] = metadata["crop_offset_min"] + (
            metadata["crop_offset_max"] - metadata["crop_offset_min"]
        ) * np.random.uniform(0, 1, len(metadata))

        metadata["crop_offset"] = metadata["crop_offset"].astype(int)

        # Drop auxilary columns.
        metadata.drop(
            [
                "crop_offset_low_limit",
                "crop_offset_high_limit",
                "crop_offset_min",
                "crop_offset_max",
            ],
            axis=1,
            inplace=True,
        )

        # Sort them by their idx to restore the original order.
        metadata.sort_index(inplace=True)

        return metadata

    def _assign_pick_ensured_crop_offset_ranges(self, eq_metadata):
        """
        Args:
            eq_metadata (pd.DataFrame): Dataframe that contains the metadata of the eq waveforms.

        Returns:
            pd.DataFrame: Dataframe that contains the metadata of the eq waveforms with crop offsets
                assigned.
        """

        _eq_metadata = eq_metadata.copy()

        # Assign min limits to crop offsets for pick ensured waveforms.
        _eq_metadata["crop_offset_min"] = (
            _eq_metadata[["p_arrival_sample", "s_arrival_sample"]].min(axis=1)
            + self._get_ts(self.phase_ensuring_margin)
            - self._get_ts(self.model_time_window)
        )

        _eq_metadata["crop_offset_min"] = _eq_metadata[
            ["crop_offset_min", "crop_offset_low_limit"]
        ].max(axis=1)

        # Assign max limits to crop offsets for pick ensured waveforms.
        _eq_metadata["crop_offset_max"] = _eq_metadata[
            ["p_arrival_sample", "s_arrival_sample"]
        ].max(axis=1) - self._get_ts(self.phase_ensuring_margin)

        _eq_metadata["crop_offset_max"] = _eq_metadata[
            ["crop_offset_max", "crop_offset_high_limit"]
        ].min(axis=1)

        return _eq_metadata

    def _get_ts(self, t):
        """
        Args:
            t (float): Time in seconds.

        Returns:
            int: Number of timesteps that corresponds to the given time.
        """
        return int(t * self.sampling_freq)
