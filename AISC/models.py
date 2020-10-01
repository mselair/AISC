import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dateutil import tz
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal, gaussian_kde
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn import preprocessing

from hypnogram.io import load_CyberPSG
from hypnogram.utils import create_day_indexes, time_to_timezone, time_to_timestamp, tile_annotations, create_duration, filter_by_duration

from AISC.utils.feature_util import augment_features, print_classification_scores
from AISC.utils.signal import unify_sampling_frequency, get_datarate, buffer
from AISC.modules.feature import ZScoreModule, LogModule, FeatureAugmentorModule, Log10Module
from AISC.FeatureExtractor.FeatureExtractor import SleepSpectralFeatureExtractor


class KDEBayesianModel:
    __name__ = "KDEBayesianModel"
    def __init__(self, fbands=[[0.5, 5], # delta
                               [4, 9], # theta
                               [8, 14], # alpha
                               [11, 16], # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=None):
        self.fbands = fbands
        self.segm_size = segm_size
        self.fs = fs
        self.bands_to_erase = bands_to_erase

        self.STATES = []
        self.KDE = []
        self.PipelineClustering = None
        self.FeatureSelector = None

        self.FeatureExtractor_MeanBand = SleepSpectralFeatureExtractor()
        self.FeatureExtractor_MeanBand._extraction_functions = \
            [
                self.FeatureExtractor_MeanBand.mean_bands,
            ]



        self.FeatureExtractor = SleepSpectralFeatureExtractor()
        self.FeatureExtractor._extraction_functions = \
            [
                self.FeatureExtractor.MeanFreq,
                self.FeatureExtractor.MedFreq,
                self.FeatureExtractor.rel_bands,
                self.FeatureExtractor.normalized_entropy,
                self.FeatureExtractor.normalized_entropy_bands
            ]

    def extract_features(self, signal, return_names=False):
        if signal.ndim > 1:
            raise AssertionError('[INPUT ERROR]: Input data has to be of a dimension size 1 - raw signal')
        if signal.shape[0] != self.fs * self.segm_size:
            print('[INPUT WARNING]: input data is not a defined size fs*segm_size ' + str(self.fs*self.segm_size) + '; Signal of a size ' + str(signal.shape[0]) + ' found instead. Extracted features might be inaccurate.')


        ## Mean band-derived features - delta/beta ratio etc
        mean_bands, feature_names = self.FeatureExtractor_MeanBand(signal, fbands=self.fbands, fs=self.fs, segm_size=signal.shape[0]/self.fs, datarate=False)
        mean_bands = np.concatenate(mean_bands)

        functions = [np.divide]
        symbols = ['/']
        mean_band_derived_features, mean_band_derived_names = mean_bands, feature_names
        for idx in range(functions.__len__()):
            mean_band_derived_features, mean_band_derived_names = augment_features(
                mean_band_derived_features.reshape(1, -1), operation=functions[idx], mutual=True,  operation_str=symbols[idx], feature_names=mean_band_derived_names
            )

        mean_band_derived_names = mean_band_derived_names[feature_names.__len__():]
        mean_band_derived_features = mean_band_derived_features[0, feature_names.__len__():]


        ## other features
        other_features, feature_names = self.FeatureExtractor(signal, fbands=self.fbands, fs=self.fs, segm_size=signal.shape[0]/self.fs, datarate=False)
        other_features = np.concatenate(other_features)

        features = np.log10(np.append(other_features, mean_band_derived_features))
        feature_names = feature_names + mean_band_derived_names
        if return_names:
            return features, feature_names
        return features

    def extract_features_bulk(self, list_of_signals, fsamp_list, return_names=False):
        data = list_of_signals
        data, fs = unify_sampling_frequency(data, sampling_frequency=fsamp_list, fs_new=200)
        x = []
        for k in tqdm(range(data.__len__())):
            x += [self.extract_features(data[k])]
        if return_names:
            _, feature_names = self.extract_features(data[k], return_names=True)
            return np.array(x), feature_names

        return np.array(x), fs

    def predict_signal(self, signal, fs, datarate_threshold=0.85):
        data = list(buffer(signal, fs, self.segm_size))
        start_time = np.array([k*self.segm_size for k in range(data.__len__())])
        datarate = np.array(get_datarate(data))
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        x = x[datarate >= datarate_threshold]
        start_time = start_time[datarate >= datarate_threshold]
        df = {'annotation': self.predict(x), 'start': start_time, 'end': start_time+30, 'duration':30}
        return df

    def fit(self, X, y):
        estimator = SVR(kernel="linear")
        self.SELECTOR = RFECV(estimator, step=5, verbose=True, min_features_to_select=1, n_jobs=10)
        self.UMAP = UMAP(n_neighbors=30, min_dist=1,
                         n_components=2,
                         )
        self.ZScore = ZScoreModule(trainable=True, continuous_learning=False, multi_class=False)
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_ = le.transform(y)
        X = self.SELECTOR.fit_transform(X, y_)
        X = self.UMAP.fit_transform(X)
        X = self.ZScore.fit_transform(X, y)


        self.STATES = np.unique(y)
        self.KDE = []
        for state in self.STATES:
            x = X[y==state, :]
            kernel = gaussian_kde(x.T)
            self.KDE.append(kernel)

    def scores(self, X):
        X = self.transform(X)
        scores = {}
        for idx, kde in enumerate(self.KDE):
            scores[self.STATES[idx]] = kde.pdf(X.T)
        return pd.DataFrame(scores)

    def transform(self, X):
        X = self.SELECTOR.transform(X)
        X = self.UMAP.transform(X)
        return self.ZScore.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return np.array(self.scores(X).idxmax(axis=1))