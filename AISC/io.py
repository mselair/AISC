
import re
import os
import shutil
import time
import scipy.signal as signal

from tqdm import tqdm
from scipy.io import savemat, loadmat
from scipy.stats import multivariate_normal

from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, cohen_kappa_score

from AISC.Sleep.Analysis import *
from AISC.FeatureExtractor import SleepSpectralFeatureExtractor
from AISC.utils.feature_util import zscore, augment_features, find_category_outliers, remove_samples, remove_features, balance_classes
from AISC.utils.signal import decimate, get_datarate, unify_sampling_frequency

class SleepDataManager:
    __version__ = '0.0.1'
    def __init__(self, df, channel, sleep_states=['AWAKE', 'REM', 'N2', 'N3'], datarate_threshold=0.95,
                 segm_size=30,
                 f_samp=200, fbands=[[1, 5],
                                     [4, 7],
                                     [5, 10],
                                     [9, 13],
                                     [10, 16],
                                     [16, 20],
                                     [20, 30]],
                 bands_to_erase = [
                     [6, 8],
                     [13, 15],
                     [20, 22]
                 ],
                 perwelchseg=5,
                 soverlapwelchseg=3.5
                 ):
        self._df = df
        self.datarate_threshold = datarate_threshold
        self.sleep_states = sleep_states
        self.channel = channel

        # Feature Extraction
        self.FeatureExtractor = SleepSpectralFeatureExtractor()
        self.sampling_frequency = f_samp # Hz - signals will be downsampled to this
        self.segm_size = segm_size # seconds
        self.fbands = fbands # Hz
        self.bands_to_erase = bands_to_erase
        self.perwelchseg = perwelchseg
        self.soverlapwelchseg = soverlapwelchseg

        # Data
        self._datarate = [] # array for each valid
        self._X_raw = None
        self._X = None
        self.feature_names = None

        if not isinstance(df, type(None)):
            self.load()

    def load(self):
        df, X_raw, fs, datarate = self.load_raw_data(self._df, self.channel)
        X_raw, _ = unify_sampling_frequency(X_raw, sampling_frequency=fs, fs_new=self.sampling_frequency)

        self._df = df
        self._X_raw = X_raw
        self._datarate = datarate

        self.extract_features()

    def extract_features(self):
        X, feature_names = self._extract_features(self._X_raw, self.sampling_frequency, self.segm_size, self.fbands, self.bands_to_erase, self.perwelchseg, self.soverlapwelchseg, datarate=False)
        self.feature_names = feature_names
        self._X = X


    def _extract_features(self, X, fs, segm_size, fbands, bands_to_erase, perwelchseg=None, soverlapwelchseg=0, datarate=False):
        X = self.FeatureExtractor(
            x = list(X),
            fs=fs,
            segm_size=segm_size,
            sperwelchseg = perwelchseg,
            soverlapwelchseg = soverlapwelchseg,
            fbands=fbands,
            bands_to_erase=bands_to_erase,
            datarate=datarate
        )

        #x: (np.ndarray, list) = None,
        #fs: float = None,
        #segm_size: float = None,
        #sperwelchseg: float = None,
        #soverlapwelchseg: float = 0,
        #fbands: list = None,
        #bands_to_erase: list = [],
        #datarate: bool = True,
        #n_processes: int = 1



        X = np.array(X)
        feature_names = X[0, 1, :].squeeze()
        X = X[:, 0, :].squeeze().astype(np.float32)
        return X, feature_names


    @property
    def index(self):
        return remove_samples(np.arange(self._df.shape[0]), to_del=self._datarate<self.datarate_threshold)

    @property
    def X(self):
        return remove_samples(self._X, to_del=self._datarate<self.datarate_threshold)

    @property
    def Y(self):
        return np.array(self.df.annotation)

    @property
    def X_raw(self):
        return remove_samples(self._X_raw, to_del=self._datarate<self.datarate_threshold)

    @property
    def df(self):
        return remove_samples(self._df, to_del=self._datarate<self.datarate_threshold)

    @property
    def datarate(self):
        return remove_samples(self._datarate, to_del=self._datarate<self.datarate_threshold)


    def get_X(self, day=None, stim_freq=None, sleep_state=None, datarate=None):
        ret_true = self.get_bool_indexes(day, stim_freq, sleep_state, datarate)
        return self._X[ret_true]

    def get_Y(self, day=None, stim_freq=None, sleep_state=None, datarate=None):
        ret_true = self.get_bool_indexes(day, stim_freq, sleep_state, datarate)
        return np.array(self._df.loc[ret_true].reset_index(drop=True).annotation)

    def get_X_raw(self, day=None, stim_freq=None, sleep_state=None, datarate=None):
        ret_true = self.get_bool_indexes(day, stim_freq, sleep_state, datarate)
        return np.concatenate(self._X_raw)[ret_true]

    def get_datarate(self, day=None, stim_freq=None, sleep_state=None, datarate=None):
        ret_true = self.get_bool_indexes(day, stim_freq, sleep_state, datarate)
        return self._datarate[ret_true]

    def get_df(self, day=None, stim_freq=None, sleep_state=None, datarate=None):
        ret_true = self.get_bool_indexes(day, stim_freq, sleep_state, datarate)
        return self._df.loc[ret_true].reset_index(drop=True)

    def get_bool_indexes(self, day, stim_freq, sleep_state, datarate):
        log_array = np.ones_like(self._datarate, dtype=bool)
        if not isinstance(datarate, type(None)):
            log_array = self._datarate >= datarate

        if not isinstance(day, type(None)):
            if not isinstance(day, list):
                day = [day]
            temp_log_array = np.zeros_like(log_array, dtype=bool)
            for d_idx in day:
                temp_log_array = (temp_log_array) | (self._df.day == d_idx)
            log_array = (log_array) & (temp_log_array)

        if not isinstance(stim_freq, type(None)):
            if not isinstance(stim_freq, list):
                stim_freq = [stim_freq]
            temp_log_array = np.zeros_like(log_array, dtype=bool)
            for freq in stim_freq:
                temp_log_array = (temp_log_array) | (self._df.stimulation_frequency == freq)
            log_array = (log_array) & (temp_log_array)

        if not isinstance(sleep_state, type(None)):
            if not isinstance(sleep_state, list):
                sleep_state = [sleep_state]
            temp_log_array = np.zeros_like(log_array, dtype=bool)
            for state in sleep_state:
                temp_log_array = (temp_log_array) | (self._df.annotation == state)
            log_array = (log_array) & (temp_log_array)

        return log_array

    @staticmethod
    def load_raw_data(df, channel_name=None, datarate_threshold = None):
        X_raw, fs = LoadDataFromMEFDClient(df, channels=channel_name)
        for k in range(X_raw.__len__()):
            X_raw[k] = X_raw[k] / 1000
        datarate = np.array(get_datarate(X_raw))
        if datarate_threshold:
            df = remove_samples(df, to_del=datarate<datarate_threshold)
            X_raw, fs = remove_samples(X_raw, fs, to_del=datarate<datarate_threshold)
            datarate = remove_samples(datarate, to_del=datarate<datarate_threshold)
        return df, X_raw, fs, datarate

    def load_inference_data(self, start_stamp=None, end_stamp=None, channel=None):
        df = pd.DataFrame()
        df['start'] = [datetime.datetime.utcfromtimestamp(start_stamp)]
        df['end'] = [datetime.datetime.utcfromtimestamp(end_stamp)]
        df['duration'] = [(df['end'][0] - df['start'][0]).seconds]
        df['day'] = 0
        #df = load_info_stimulation(df)
        df = find_session_iEEG_db(df)

        X_raw, fs = LoadDataFromMEFDClient(df, channels=channel)
        X_raw = X_raw[0].squeeze()
        fs = fs[0]
        X_raw = X_raw / 1000
        X_raw = SleepSpectralFeatureExtractor.buffer(X_raw, fs, int(self.segm_size))
        datarate = np.array(get_datarate(X_raw))
        return X_raw, datarate
