# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from copy import deepcopy
from AISC.utils.signal import LowFrequencyFilterLP, fft_filter

def get_first_extreme(X, X_signum, idx=0, tag=''):
    if tag == 'min':
        func_value = -2
    elif tag == 'max':
        func_value = 2

    pos = np.where(X_signum[idx:] == func_value)[0]
    if pos.__len__() == 0:
        return None, None
    pos = pos[0] + idx
    val = X[pos]
    return pos, val

def get_next_extrem_by_zero(X, X_signum, curr_pos=0, tag=''):
    if tag == 'min':
        func_value = -2
        zero_comp_func = np.less
    elif tag == 'max':
        func_value = 2
        zero_comp_func = np.greater

    pos = np.where(X_signum[curr_pos:] == func_value)[0]
    if pos.__len__() == 0:
        return None, None
    pos = pos + 1 + curr_pos
    vals = X[pos]

    pos = pos[zero_comp_func(vals, 0)]

    pos = pos[0]
    val = X[pos]
    return pos, val

def check_next_extreme(X, X_signum, curr_pos=0, tag=''):
    if tag == 'min':
        argfunc = np.argmin
        func_value = -2
        comp_func = np.less
        rev_comp_func = np.greater
        rev_tag = 'max'
    elif tag == 'max':
        argfunc = np.argmax
        func_value = 2
        comp_func = np.greater
        rev_comp_func = np.less
        rev_tag = 'min'

    tag_pos = curr_pos
    tag_val = X[curr_pos]

    curr_pos = curr_pos
    curr_val = X[curr_pos]

    next_rtag_pos = curr_pos + 1
    next_rtag_val = 0

    next_tag_pos = curr_pos + 1
    next_tag_val = 0
    # MAX
    while not rev_comp_func(next_rtag_val, 0):
        next_tag_pos, next_tag_val = get_first_extreme(X, X_signum, next_tag_pos + 1, tag)
        next_rtag_pos, next_rtag_val = get_first_extreme(X, X_signum, next_rtag_pos + 1, rev_tag)

        if isinstance(next_rtag_pos, type(None)) or isinstance(next_tag_pos, type(None)):
            tag_pos, tag_val = None, None
            break

        if not rev_comp_func(next_rtag_val, 0):
            if comp_func(next_tag_val, tag_val):
                tag_pos = next_tag_pos
                tag_val = next_tag_val

    return tag_pos, tag_val

def correct_extreme_positions(X, extreme_positions, n_samples_interval=10, tag=''):
    if tag == 'min':
        argfunc = np.argmin
    elif tag == 'max':
        argfunc = np.argmax

    extreme_positions = deepcopy(extreme_positions)
    for idx, pos in enumerate(extreme_positions):
        idx_from = pos - n_samples_interval
        idx_to = pos + n_samples_interval

        if idx_from < 0: idx_from = 0
        if idx_to >= X.shape[0]: idx_to = X.shape[0] - 1

        new_pos = argfunc(X[idx_from:idx_to]) + idx_from
        extreme_positions[idx] = new_pos
    return extreme_positions

def find_wave_extremes(X, fs, cutoff_low=0.5, cutoff_high=4):
    X_ = X.copy()
    X = fft_filter(X, fs, cutoff_low, 'hp')
    X = fft_filter(X, fs, cutoff_high, 'lp')

    X_sign = np.sign(np.diff(X))
    X_sign = X_sign[:-1] - X_sign[1:]


    mins = []
    maxes = []
    pos_max = 0
    pos_min = 0
    while pos_max != -1:
        # find 1st min lower than zero
        pos_min, val_min = get_next_extrem_by_zero(X, X_sign, pos_max, tag='min')
        if pos_min == None:
            pos_max = -1
            break
        pos_min, val_min = check_next_extreme(X, X_sign, pos_min, tag='min')
        if pos_min == None:
            pos_max = -1
            break
        pos_max, val_max = get_next_extrem_by_zero(X, X_sign, pos_min, tag='max')
        if pos_max == None:
            pos_max = -1
            break
        pos_max, val_max = check_next_extreme(X, X_sign, pos_max, tag='max')
        if pos_max == None:
            pos_max = -1
            break

        if pos_max == None:
            pos_max = -1
            break
        else:
            mins += [pos_min]
            maxes += [pos_max]
            #pos_min = pos_max + n_plato

    mins = np.array(mins)
    maxes = np.array(maxes)

    mins = correct_extreme_positions(X_, extreme_positions=mins, n_samples_interval=round(int((1 / cutoff_high) * fs / 2)), tag='min')
    maxes = correct_extreme_positions(X_, extreme_positions=maxes, n_samples_interval=round(int((1 / cutoff_high) * fs / 2)), tag='max')

    dif = maxes - mins
    maxes = maxes[dif > ((1/cutoff_high) * fs / 2)]
    mins = mins[dif > ((1/cutoff_high) * fs / 2)]

    dif = maxes - mins
    maxes = maxes[dif < ((1/cutoff_low) * fs / 2)]
    mins = mins[dif < ((1/cutoff_low) * fs / 2)]
    return mins, maxes

class WaveDetector:
    """
    WaveDetector
    Designed for Delta wave detection. Can be deployed on any frequency range.
    For example see GitHub README File
    """

    def __init__(self, fs, cutoff_low=0.5, cutoff_high=4):
        self.fs = fs
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.LFFilter = LowFrequencyFilterLP(fs=fs, cutoff=cutoff_low, n_decimate=2, n_order=101, dec_cutoff=0.3)

        self.statistic_functions = [self.min_stats, self.max_stats, self.pk2pk_stats, self.slope_stats, self.delta_t_stats]


    def __call__(self, X, stats=True):
        if not isinstance(X, (list, np.ndarray)):
            raise AssertionError('')

        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = [X]
            else:
                X = list(X)


        min_pos = []
        min_val = []
        max_pos = []
        max_val = []

        for X_ in X:
            min_pos_temp, min_val_temp, max_pos_temp, max_val_temp = self.detect_waves(X_)
            min_pos += list(min_pos_temp)
            min_val += list(min_val_temp)
            max_pos += list(max_pos_temp)
            max_val += list(max_val_temp)


        min_pos = np.array(min_pos)
        min_val = np.array(min_val)
        max_pos = np.array(max_pos)
        max_val = np.array(max_val)

        outp_data = {
            'min_pos': min_pos,
            'min_val': min_val,
            'max_pos': max_pos,
            'max_val': max_val
        }


        if stats is True:
            outp_stats = {}
            for func in self.statistic_functions:
                name = func.__name__
                dict_ = func(min_pos, min_val, max_pos, max_val, self.fs)
                outp_stats[name] = dict_
            return outp_stats, outp_data

        return outp_data


    def detect_waves(self, X):
        X = X - X.mean()
        X = self.LFFilter(X)
        min_pos, max_pos = find_wave_extremes(X, fs=self.fs, cutoff_low=self.cutoff_low, cutoff_high=self.cutoff_high)
        min_val = X[min_pos]
        max_val = X[max_pos]
        return min_pos, min_val, max_pos, max_val

    @classmethod
    def min_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        return cls.stat(min_vals)

    @classmethod
    def max_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        return cls.stat(max_vals)

    @classmethod
    def pk2pk_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        pk2pk = max_vals - min_vals
        return cls.stat(pk2pk)

    @classmethod
    def slope_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        slope = (max_vals - min_vals) / ((max_pos - min_pos) / fs)
        return cls.stat(slope)

    @classmethod
    def delta_t_stats(cls, min_pos=None, min_vals=None, max_pos=None, max_vals=None, fs=None):
        delta_t = (max_pos - min_pos) / fs
        return cls.stat(delta_t)

    @staticmethod
    def stat(X):
        return {
            'min': X.min(),
            'max': X.max(),
            'mean': X.mean(),
            'std': X.std(),
            'median': np.median(X),
        }


