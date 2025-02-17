from obspy import UTCDateTime
import h5py
import numpy as np
import pandas as pd
from os.path import exists
from tensorflow.keras.utils import Sequence
from config import BATCH_SIZE, SAMPLING_FREQ, FREQMIN, FREQMAX

# CURRENTLY, BATCH GENERATOR IN THIS FILE IS INCOMPLETE (FOR RAW DATA) AND WILL NEED TO BE REVISED LATER!
class GeneratorWrapper(Sequence):
    """
    GeneratorWrapper is a wrapper class for DataGenerator class. It is used to
    generate batches of data for training.

    Args:
        data_generator (DataGenerator): DataGenerator instance that is used to
            generate batches of data.
    """

    def __init__(self, data_generator):
        self.data_generator = data_generator

    def __len__(self):
        return self.data_generator.length()

    def __getitem__(self, idx):
        return self.data_generator.getitem(idx)

    def on_epoch_end(self):
        self.data_generator.on_epoch_end()


class PredictGenerator(Sequence):
    """
    PredictGenerator is a wrapper class for DataGenerator class. It is used to
    generate batches of data for prediction.

    Args:
        data_generator (DataGenerator): DataGenerator instance that is used to
            generate batches of data.
    """

    def __init__(self, data_generator):
        self.data_generator = data_generator

    def __len__(self):
        return self.data_generator.length()

    def __getitem__(self, idx):
        x, y = self.data_generator.getitem(idx)
        return x

    def on_epoch_end(self):
        self.data_generator.on_epoch_end()


class BatchGenerator:
    """
    BatchGenerator is a class that generates batches of data from a given
    dataframe. It is used to generate batches of data for training, validation
    and testing. It is used by DataGenerator class.

    Args:
        batch_size (int): Batch size.
        batch_metadata (pd.DataFrame): Dataframe that contains the metadata of the waveforms.
        eq_hdf5_path (str): Path of the hdf5 file that contains the waveforms of the eq events.
        no_hdf5_path (str): Path of the hdf5 file that contains the waveforms of the no events.
        raw_hdf5_path (str): Path of the hdf5 file that contains the raw waveforms.
        dataset_time_window (float): The time window of the dataset.
        model_time_window (float): The time window of the model.
        sampling_freq (int): Sampling frequency.
        freqmin (float): The lower frequency of the bandpass filter.
        freqmax (float): The upper frequency of the bandpass filter.
        last_axis (str): The last axis of the data. Can be either "channels" or "timesteps".
    """

    def __init__(
        self,
        batch_size=BATCH_SIZE,
        batch_metadata=pd.DataFrame(),
        eq_hdf5_path="",
        no_hdf5_path="",
        raw_hdf5_path="",
        dataset_time_window=120.0,
        model_time_window=30.0,
        sampling_freq=SAMPLING_FREQ,
        freqmin=FREQMIN,
        freqmax=FREQMAX,
        last_axis="channels",
        dataset_type="stead" 
    ):
        self.batch_size = batch_size
        self.dataset_time_window = dataset_time_window
        self.model_time_window = model_time_window
        self.sampling_freq = sampling_freq
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.last_axis = last_axis
        self.dataset_type = dataset_type
        
        self.eq_hdf5_path = eq_hdf5_path
        self.no_hdf5_path = no_hdf5_path
        self.raw_hdf5_path = raw_hdf5_path
        

        self.f = np.fft.fftfreq(
            self._get_ts(self.dataset_time_window), 1.0 / sampling_freq
        )

        # Parse metadata.
        self.waveforms = self._get_waveforms(batch_metadata)

        f_eq = h5py.File(self.eq_hdf5_path, "r")
        self.data_pick = f_eq.get("data/")

        f_no = h5py.File(self.no_hdf5_path, "r")
        self.data_noise = f_no.get("data/")

        f_raw = h5py.File(self.raw_hdf5_path,'r')
        self.data_raw = f_raw.get("data/")

        self.x_batch = None
        self.y_batch = None

    def num_batches(self):
        """
        Returns:
            int: Number of batches that can be generated from the dataframe.
        """
        return len(self.waveforms) // self.batch_size

    def get_batch(self, idx):
        """
        Args:
            idx (int): Index of the batch.

        Returns:
            tuple: A tuple of (x_batch, y_batch). x_batch is the batch of data
                and y_batch is the batch of labels.
        """
        batch_waveforms = self.waveforms[
            (idx * self.batch_size) : ((idx + 1) * self.batch_size)
        ]

        self.x_batch = self._get_batchx(batch_waveforms)
        self.y_batch = self._get_batchy(batch_waveforms)

        return self.x_batch, self.y_batch

    def _get_waveforms(self, df):
        """
        Get a list of waveforms from a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing metadata.

        Returns
        -------
        list
            List of waveforms.
        """
        waveforms = []

        for __, row in df.iterrows():

            # Common for all labels
            waveform = {
                'trace_name': row['trace_name'],
                'station_name': row['station_name'],
                'trace_start_time': self._utc_datetime_with_nan(row['trace_start_time']),
                'crop_offset': row['crop_offset'],
                'label': row['label']
            }

            if row['label'] == "eq":
                waveform.update({
                    'p_arrival_sample': row['p_arrival_sample'],
                    's_arrival_sample': row['s_arrival_sample'],
                })

            else: # Covers for both 'raw' and 'no'
                waveform.update({
                    'p_arrival_sample': pd.NA,
                    's_arrival_sample': pd.NA,
                })

            waveforms.append(waveform)

        return waveforms

    def _get_batchx(self, batch_waveforms):
        """
        Args:
            batch_waveforms (list): List of waveforms in the batch.

        Returns:
            np.ndarray: Batch of x data.
        """
        x_batch = []
        crop_offsets = []

        for waveform in batch_waveforms:
            if waveform["label"] == 'eq':
                loaded_data_eq=self._load_labeled_waveform(waveform)
                x_batch.append(loaded_data_eq)

            elif waveform["label"] == 'no':
                loaded_data_no= self._load_labeled_waveform(waveform)
                x_batch.append(loaded_data_no)
            
            elif waveform["label"] == 'raw':
                loaded_data_raw=self._load_raw_waveform(waveform)
                x_batch.append(loaded_data_raw)
            else:
                raise ValueError('Unknown label for the waveform: ', waveform['label'])

            crop_offset = waveform['crop_offset']
            crop_offsets.append(crop_offset)

        # Create x tensor. It's shape is (BATCH_SIZE, N_TIMESTEPS, N_CHANNELS)
        x_batch = np.array(x_batch).astype(np.float32)
        crop_offsets = np.array(crop_offsets).astype(np.float32)

        return self._preprocess(x_batch,crop_offsets)


    def _load_labeled_waveform(self,waveform):
        '''
        Loads waveform from labeled datasets such as STEAD or INSTANCE.
        '''
        try: 
            data_source = self.data_pick if waveform['label']=='eq' else self.data_noise

            if self.last_axis == 'channels':
                return data_source[waveform['trace_name'][:]]
            elif self.last_axis == 'timesteps':
                return data_source[waveform['trace_name'][:]].T # "If last_axis was given as timesteps axis in the dataset, make channels last."
            else: 
                raise ValueError(f"Invalid last_axis: {self.last_axis}. Valid values for last_axis: [channels, timesteps]")

        except KeyError:
            raise ValueError(f"Waveform {waveform['trace_name']} not found in {waveform['label']} data")



    def _load_raw_waveform(self,waveform):
        '''
        Loads raw waveforms data from HDF5.

        Args:
            waveform
        Returns: 

        '''
        try: 
            trace_group = self.data_raw[waveform['trace_name']]
            data = trace_group['data'][:]

            if self.last_axis == 'timesteps':
                data = data.T
            
            return data

        except KeyError:
            raise ValueError(f"Waveform {waveform['trace_name']} not found in the raw data HDF5.")
        


    def _preprocess(self,x_batch,crop_offsets):
        '''
        Applies preprocessing pipeline to a batch of waveforms. 
        
        On the original RECOVAR paper (Efe O., Ozakin A. (2024)), 
        is cropping the input to 30 seconds, applying a bandpass filter (1-20 Hz) and normalization.

        Args: 
            x_batch (np.ndarray): Batch of raw waveforms
            crop_offsets (np.ndarray)

        Returns: 
            np.ndarray: Preprocessed batch ready to be input. 

        '''

        # If last_axis was given as timesteps axis in the dataset, make channels last.
        if self.last_axis == "timesteps":
            x_batch = np.transpose(x_batch, axes=[0, 2, 1]) #(batch, timesteps, channels)
        
        # Convert to Fourier domain.
        xw = np.fft.fft(x_batch, axis=1)

        # Roll waveforms towards the left by crop_offsets.
        xw = xw * np.exp(1j * 2 * np.pi * f * crop_offsets / self.sampling_freq)

        # Apply bandpass filtering at Fourier domain.
        mask = (np.abs(self.f) < self.freqmin) | (np.abs(self.f) > self.freqmax)
        xw[:, mask, :] = 0

        # Convert to time domain and retrieve numeric type.
        x_batch = np.fft.ifft(xw, axis=1)
        x_batch = np.real(x_batch).astype(np.float32)

        # Slice the x in timesteps direction.
        x_batch = x_batch[:, 0 : self._get_ts(self.model_time_window), :]

        # Demean timesteps axis. And then normalize each channel.
        x_batch -= np.mean(x_batch, axis=1, keepdims=True)
        return self._normalize(x_batch, axis=1)


    def _get_batchy(self, batch_waveforms):
        """
        Args:
            batch_waveforms (list): List of waveforms in the batch.

        Returns:
            np.ndarray: Batch of y data.
        """
        y = np.array(
            [waveform["label"] == "eq" for waveform in batch_waveforms],
            dtype=np.int32,
        )
        return y

    def _get_ts(self, t):
        """
        Args:
            t (float): Time in seconds.

        Returns:
            int: Number of timesteps that corresponds to the given time.
        """
        return int(t * self.sampling_freq)

    @staticmethod
    def _utc_datetime_with_nan(s):
        """
        Convert a string to UTCDateTime, or pd.NA if the string is NaN.

        Parameters
        ----------
        s : str
            String to convert.

        Returns
        -------
        UTCDateTime or pd.NA
        """
        if pd.isna(s):
            return pd.NA
        else:
            return UTCDateTime(s)

    @staticmethod
    def _normalize(x, axis):
        """
        Args:
            x (np.ndarray): Array to be normalized.
            axis (int): Axis to be normalized.

        Returns:
            np.ndarray: Normalized array.
        """

        norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
        return x / (1e-37 + norm)


class DataGenerator(Sequence):
    """
    DataGenerator is a class that generates batches of data for training, validation
    and testing. It is used by TrainingGenerator, ValidationGenerator and TestGenerator
    classes.

    Args:
        processed_hdf5_path (str): Path of the hdf5 file that contains the preprocessed
            data.
        chunk_metadata_list (list): List of dataframes that contains the metadata of the waveforms
            in the chunks.
        batch_size (int): Batch size.
        phase_ensured_crop_ratio (float): The ratio of the eq waveforms that are ensured to include
            the phase arrival times.
        eq_hdf5_path (str): Path of the hdf5 file that contains the waveforms of the eq events.
        no_hdf5_path (str): Path of the hdf5 file that contains the waveforms of the no events.
        raw_hdf5_path (str): Path of the hdf5 file that contains the raw waveforms.
        meta_parser (MetaParser): MetaParser instance that is used to parse the metadata.
        dataset_time_window (float): The time window of the dataset.
        model_time_window (float): The time window of the model.
        phase_ensured_crop_ratio (float): The ratio of the eq waveforms that are ensured to include
            the phase arrival times.
        sampling_freq (int): Sampling frequency.
        freqmin (float): The lower frequency of the bandpass filter.
        freqmax (float): The upper frequency of the bandpass filter.
        last_axis (str): The last axis of the data. Can be either "channels" or "timesteps".
    """

    def __init__(
        self,
        processed_hdf5_path,
        chunk_metadata_list,
        batch_size,
        phase_ensured_crop_ratio,
        dataset_time_window=120.0,
        model_time_window=30.0,
        sampling_freq=SAMPLING_FREQ,
        active_chunks=[],
        *args,
        **kwargs
    ):
        self.processed_hdf5_path = processed_hdf5_path
        self.chunk_metadata_list = chunk_metadata_list
        self.batch_size = batch_size
        self.phase_ensured_crop_ratio = phase_ensured_crop_ratio
        self.dataset_time_window = dataset_time_window
        self.model_time_window = model_time_window
        self.sampling_freq = sampling_freq
        self.active_chunks = active_chunks
        self.bg_args = args
        self.bg_kwargs = kwargs
        self.chunk_batch_counts = self.get_chunk_batch_counts()

        if not exists(self.processed_hdf5_path):
            self._render_dataset()
        
        self.processed_hdf5 = h5py.File(self.processed_hdf5_path, "r", locking=True)
        
    def getitem(self, idx):
        """
        Args:
            idx (int): Index of the batch.
        Returns:
            tuple: A tuple of (x_batch, y_batch). x_batch is the batch of data
                and y_batch is the batch of labels.
        """
        chunk_idx, batch_offset = self.get_chunk_idx_and_batch_offset(idx)

        x_batch = self.processed_hdf5.get(
            "data/x/chunk{}/{}".format(chunk_idx, batch_offset)
        )
        y_batch = self.processed_hdf5.get(
            "data/y/chunk{}/{}".format(chunk_idx, batch_offset)
        )

        return x_batch, y_batch

    def length(self):
        """
        Returns:
            int: Number of batches that can be generated from the given chunks.
        """
        n_batches = 0
        for chunk in self.active_chunks:
            n_batches = n_batches + self.chunk_batch_counts[chunk]

        return n_batches

    def on_epoch_end(self):
        """
        This method is called at the end of each epoch.
        """
        pass

    def get_chunk_idx_and_batch_offset(self, batch_idx):
        """
        Args:
            batch_idx (int): Index of the batch.

        Returns:
            tuple: A tuple of (chunk, batch_offset). chunk is the chunk that the batch belongs to
                and batch_offset is the offset of the batch in the chunk.
        """
        batch_offset = batch_idx
        for chunk in self.active_chunks:
            if batch_offset < self.chunk_batch_counts[chunk]:
                return chunk, batch_offset

            batch_offset -= self.chunk_batch_counts[chunk]

        return None, None

    def get_chunk_batch_counts(self):
        """
        Returns:
            list: List of batch counts for each chunk.
        """
        chunk_batch_counts = {}
        for chunk_idx, chunk_metadata in enumerate(self.chunk_metadata_list):
            chunk_batch_counts[chunk_idx] = len(chunk_metadata) // self.batch_size

        return chunk_batch_counts

    def __del__(self):
        """
        This method is called when the class instance is deleted closing the hdf5 file.
        """
        self.processed_hdf5.close()

    def _render_dataset(self):
        """
        Renders the dataset into a specific format and saves it to a hdf5 file. The dataset is
        split into chunks. Seperation is must be given by the list of dataframes for each chunk.
        Then each chunk is split into batches and batches are stored in the hdf5 file.
        """
        with h5py.File(self.processed_hdf5_path, "w") as processed_hdf5:
            for chunk_idx in range(len(self.chunk_metadata_list)):
                bg = BatchGenerator(
                    batch_size=self.batch_size,
                    batch_metadata=self.chunk_metadata_list[chunk_idx],
                    eq_hdf5_path=self.bg_kwargs["eq_hdf5_path"],
                    no_hdf5_path=self.bg_kwargs["no_hdf5_path"],
                    meta_parser=self.bg_kwargs["meta_parser"],
                    dataset_time_window=self.dataset_time_window,
                    model_time_window=self.model_time_window,
                    sampling_freq=self.sampling_freq,
                    freqmin=self.bg_kwargs["freqmin"],
                    freqmax=self.bg_kwargs["freqmax"],
                    last_axis=self.bg_kwargs["last_axis"],
                )

                n_chunk_batches = bg.num_batches()

                for chunk_batch_offset in range(n_chunk_batches):
                    x, y = bg.get_batch(chunk_batch_offset)

                    processed_hdf5.create_dataset(
                        "data/x/chunk{}/{}".format(chunk_idx, chunk_batch_offset),
                        data=x,
                        compression=None,
                    )

                    processed_hdf5.create_dataset(
                        "data/y/chunk{}/{}".format(chunk_idx, chunk_batch_offset),
                        data=y,
                        compression=None,
                    )

                processed_hdf5.create_dataset(
                    "metadata/chunk{}/{}".format(chunk_idx, "num_batches"),
                    data=n_chunk_batches,
                )
