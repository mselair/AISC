import numpy as np
from tqdm import tqdm

from copy import deepcopy, copy

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
from hypnogram.utils import create_day_indexes, time_to_utc, time_to_timezone, time_to_timestamp, tile_annotations, create_duration, filter_by_duration, merge_annotations

from AISC.utils.feature_util import augment_features, print_classification_scores
from AISC.utils.signal import unify_sampling_frequency, get_datarate, buffer
from AISC.modules.feature import ZScoreModule, LogModule, FeatureAugmentorModule, Log10Module, PCAModule
from AISC.FeatureExtractor.FeatureExtractor import SleepSpectralFeatureExtractor
from scipy.signal import filtfilt, lfilter
from scipy.signal.windows import gaussian

class SleepStageProbabilityMarkovChainFilter:
    def __init__(self, stability=0.8):
        self.STATES = np.array(['AWAKE', 'N1', 'N2', 'N3', 'REM'])
        self.tmat = np.array(
            [[0.961, 0.038, 0.001, 0.000, 0.000],
             [0.097, 0.215, 0.634, 0.000, 0.054],
             [0.020, 0.001, 0.846, 0.060, 0.073],
             [0.005, 0.001, 0.105, 0.880, 0.009],
             [0.017, 0.003, 0.061, 0.000, 0.918]]
        )

        np.fill_diagonal(self.tmat, self.tmat.diagonal()*stability)
        diag = self.tmat.diagonal()
        self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        #diag = self.tmat.diagonal() * stability
        #self.tmat = self.tmat * (1-stability)
        #np.fill_diagonal(self.tmat, diag)
        self._get_vars()

    def fit(self, X, y):
        classes = np.unique(y)
        class_certainty = dict([[state, X[state][y==state].median()] for state in classes])

        for cl in self.STATES:
            if cl not in classes: self.remove_class(cl)
        self.correct_certainty(class_certainty)



    def _get_vars(self):
        self.tmat = copy(self.tmat / self.tmat.sum(axis=1).reshape(-1, 1))
        self.change_prob = copy(1 - self.tmat.diagonal())
        self.prob_change_to = copy(self.tmat / (self.tmat.sum(axis=1) - self.tmat.diagonal()).reshape(-1, 1))
        self.prob_change_to[range(self.prob_change_to.shape[0]), range(self.prob_change_to.shape[1])] = 0

    def get_state_idx(self, state):
        return np.where(self.STATES == state)[0][0]

    def get_state_priors(self, state):
        idx = self.get_state_idx(state)
        priors = self.tmat[idx]
        return priors

    def get_state_change_prior(self, state):
        state_idx = self.get_state_idx(state)
        priors = self.get_state_priors(state)
        change_prior = 1 - priors[state_idx]
        return change_prior

    def get_prob_to_change(self, state, x_prob):
        idx = self.get_state_idx(state)
        prob_to_stay = x_prob[idx]
        return 1 - prob_to_stay

    def get_state_change_posterior(self, state, x_prob):
        prior = self.get_state_change_prior(state)
        prob = self.get_prob_to_change(state, x_prob)
        return prior * prob / (prior*prob + (1-prior)*(1-prob))


    def get_changing_state_priors(self, state):
        state_idx = copy(self.get_state_idx(state))
        priors = copy(self.get_state_priors(state))
        priors[state_idx] = 0
        priors = priors / priors.sum()
        return priors

    def get_changing_state_probabilities(self, state, x_prob):
        idx = self.get_state_idx(state)
        x_prob[idx] = 0
        return x_prob / x_prob.sum()

    def get_changing_state_posteriors(self, state, x_prob):
        priors = self.get_changing_state_priors(state)
        probs = self.get_changing_state_probabilities(state, x_prob)
        return probs*priors / np.sum(priors*probs)

    def correct_certainty(self, certainty: dict):
        for state in certainty.keys():
            idx = self.get_state_idx(state)
            cert = certainty[state]
            self.tmat[idx, idx] = cert * self.tmat[idx, idx]

        #diag = self.tmat.diagonal()
        #self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        self._get_vars()

    def remove_class(self, class_name):
        idx = self.get_state_idx(class_name)
        self.tmat = np.delete(np.delete(self.tmat, idx, axis=0), idx, axis=1)

        diag = self.tmat.diagonal()
        self.tmat = self.tmat / (self.tmat.sum(axis=1)-diag).reshape(-1, 1) * (1-diag).reshape(-1, 1)
        np.fill_diagonal(self.tmat, diag)

        self.STATES = self.STATES[self.STATES != class_name]
        self._get_vars()



class KDEBayesianModel:
    __name__ = "KDEBayesianModel"
    def __init__(self, fbands=[[0.5, 5], # delta
                               [4, 9], # theta
                               [8, 14], # alpha
                               [11, 16], # spindle
                               [14, 20],
                               [20, 30]], segm_size=30, fs=200, bands_to_erase=None,
                 window_smooth_n=3, window_std=1, cat_bias={'N2': 1, 'REM': 1}
                 ):

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


        self.WINDOW = gaussian(window_smooth_n, window_std)
        self.WINDOW = self.WINDOW / self.WINDOW.sum()
        self.CAT_BIAS = cat_bias


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

                mean_band_derived_features.reshape(1, -1), feature_indexes=np.arange(mean_band_derived_features.shape[0]), operation=functions[idx], mutual=True,  operation_str=symbols[idx], feature_names=mean_band_derived_names

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

    def fit(self, X, y):
        estimator = SVR(kernel="linear")
        self.SELECTOR = RFECV(estimator, step=5, verbose=True, min_features_to_select=1, n_jobs=10)
        self.PCA = PCAModule(var_threshold=0.98)
        self.ZScore = ZScoreModule(trainable=True, continuous_learning=False, multi_class=False)
        #self.UMAP = UMAP(n_neighbors=30, min_dist=1,
        #                 n_components=2)

        le = preprocessing.LabelEncoder()
        le.fit(y)
        y_ = le.transform(y)


        X = self.SELECTOR.fit_transform(X, y_)
        X = self.PCA.fit_transform(X)
        X = self.ZScore.fit_transform(X, y)
        #X = self.UMAP.fit_transform(X)



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

        scores = pd.DataFrame(scores)
        scores = scores.div(scores.sum(axis=1), axis=0)
        for key in scores.keys():
            scores[key] = filtfilt(self.WINDOW, 1, scores[key])

        for cat in self.CAT_BIAS.keys():
            if cat in scores.keys(): scores[cat] = scores[cat]*self.CAT_BIAS[cat]

        scores = scores.div(scores.sum(axis=1), axis=0)
        return scores

    def transform(self, X):
        X = self.SELECTOR.transform(X)
        X = self.PCA.transform(X)
        X = self.ZScore.transform(X)
        #X = self.UMAP.transform(X)
        return X


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):

        return np.array(self.scores(X).idxmax(axis=1))

    def preprocess_signal(self, signal, fs, datarate_threshold=0.85):
        data = buffer(signal, fs, self.segm_size)
        start_time = np.array([k*self.segm_size for k in range(data.__len__())])
        end_time = start_time + self.segm_size
        datarate = np.array(get_datarate(data))

        data = data[datarate >= datarate_threshold]
        start_time = start_time[datarate >= datarate_threshold]
        end_time = end_time[datarate >= datarate_threshold]
        return list(data), start_time, end_time

    def predict_signal(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        scores = self.scores(x)
        df = pd.DataFrame({'annotation': scores.idxmax(axis=1), 'start': start_time, 'end': start_time+30, 'duration':30})
        df = time_to_utc(df)
        df = merge_annotations(df)
        df = time_to_timestamp(df)
        return df

    def predict_signal_scores(self, signal, fs, datarate_threshold=0.85):
        data, start_time, end_time = self.preprocess_signal(signal, fs, datarate_threshold)
        x, fs = self.extract_features_bulk(data, [fs]*data.__len__())
        scores = self.scores(x)
        return scores



class KDEBayesianCausalModel(KDEBayesianModel):
    __name__ = "KDEBayesianModel"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MarkovFilter = None

    def fit(self, X, y):
        super().fit(X, y)

        scores = super().scores(X)
        self.MarkovFilter = SleepStageProbabilityMarkovChainFilter()
        self.MarkovFilter.fit(scores, y)


    def scores(self, X):
        scores = super().scores(X)
        state = 'AWAKE'

        ch_posts = []
        for k in range(scores.__len__()):
            p_likelihood_change = scores.iloc[k][self.MarkovFilter.STATES[self.MarkovFilter.STATES != state]].sum()
            p_prior_change = self.MarkovFilter.get_state_change_prior(state)
            p_post_change = (p_likelihood_change * p_prior_change) / ((p_likelihood_change * p_prior_change) + ((1-p_likelihood_change) * (1-p_prior_change)))
            ch_posts += [[p_post_change, state]]

            p_likelihood = scores.iloc[k][self.MarkovFilter.STATES[self.MarkovFilter.STATES != state]]
            p_prior = self.MarkovFilter.get_changing_state_priors(state)[self.MarkovFilter.STATES != state]
            p_post = p_likelihood*p_prior / sum(p_likelihood*p_prior)

            if p_post_change > 0.5:
                state = p_post.idxmax()
                scores.loc[k, state] = p_post_change
                scores.loc[k, self.MarkovFilter.STATES[self.MarkovFilter.STATES!=state]] = p_post*(1-p_post_change)
            else:
                scores.loc[k, state] = 1-p_post_change
                scores.loc[k, self.MarkovFilter.STATES[self.MarkovFilter.STATES!=state]] = p_post*p_post_change
            #yy[k] = state
        return scores

    def _scores(self, X):
        return super().scores(X)

