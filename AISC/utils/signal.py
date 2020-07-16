# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.signal as signal
import scipy.fft as fft


def get_datarate(x):
    """
    Calculates datarate based on NaN values

    Parameters
    ----------
    x : numpy ndarray / list

    Returns
    -------
    numpy ndarray

    flaot values 0-1 with relative NaN count per signal.
    """
    if isinstance(x, np.ndarray):
        return 1 - (np.isnan(x).sum(axis=1)  / (x.shape[1]))
    else:
        return [1-(np.isnan(x_).sum()/x_.squeeze().shape[0]) for x_ in x]

def decimate(x, fs, fs_new, cutoff=None, datarate=False):
    """
    Downsample signal with anti-aliasing filter (Butterworth - 16th order).
    Works for signals with NaN values - replaced by zero.
    Can return also data-rate. Can take matrix where signals are stacked in 0th dim.

    Parameters
    ----------
    x : np ndarray
        shape[n_signal, n_sample], shape[n_sample] - can process multiple signals.
    fs : float
        Signals fs
    fs_new : float
        Downsample to fs_new
    cutoff : float
        Elective cutoff freq. of anti-alias. filter. By default 2/3 of 0.5*fs_new
    datarate : bool
        If return datarate of signals

    Returns
    -------
    numpy ndarray / tuple

    """

    x = x.copy()
    if datarate is True:
        datarate = get_datarate(x)

    if isinstance(cutoff, type(None)):
        cutoff = fs_new / 3 # two 3rds of half-sampling

    b_multiple_signals = True
    if x.ndim == 1:
        b_multiple_signals = False
        x = x.reshape(1, -1)

    for idx in range(x.shape[0]):
        x[idx, np.isnan(x[idx, :])] = np.nanmean(x[idx, :])

    b, a = signal.butter(16, cutoff/(0.5*fs), 'lp', analog=False)
    x = signal.filtfilt(b, a, x, axis=1)

    n_resampled = int(np.round((fs_new / fs) * x.shape[1]))
    x = signal.resample(x, n_resampled, axis=1)

    if b_multiple_signals is False:
        x = x.squeeze()

    if datarate:
        return x, datarate
    return x
    # PLOT AMPLITUDE CHAR
    #w, h = signal.freqs(b, a)
    #w = 0.5 * fs * w / w.max()
    #plt.semilogx(w, 20 * np.log10(abs(h) / (abs(h)).max()))
    #plt.title('Butterworth filter frequency response')
    #plt.xlabel('Frequency [radians / second]')
    #plt.ylabel('Amplitude [dB]')
    #plt.margins(0, 0.1)
    #plt.grid(which='both', axis='both')
    #plt.axvline(125, color='green') # cutoff frequency
    #plt.show()

def unify_sampling_frequency(x : list, sampling_frequency: list, fs_new=None) -> tuple:
    """
    Takes list of signals and list of frequencies and downsamples to the same sampling frequency.
    If all frequencies are same and fs_new is not specified, no operation performed. If not all frequencies are the same
    and fs_new is not specified, downsamples all signals on the lowest fs present in the list. If fs_new is specified,
    signals will be processed and downsampled on that frequency. If all sampling frequencies == fs_new, nothing is performed.

    Parameters
    ----------
    x : list
        list of numpy signals for downsampling
    sampling_frequency : list
        for each signal
    fs_new : float
        new sampling frequency

    Returns
    -------
    tuple - (numpy ndarray, new_freq)
    """

    b_process = False

    if not isinstance(x, list):
        raise TypeError('First variable must be list of numpy arrays')

    if not isinstance(sampling_frequency, (list, np.ndarray)):
        raise TypeError('Second parameter must be list or array of floats/integers')
    sampling_frequency = np.array(sampling_frequency)

    if x.__len__() != sampling_frequency.__len__():
        raise AssertionError('Length of a signal list must be same as length of sampling_frequency list')

    fs_in_set = np.unique(sampling_frequency)
    if isinstance(fs_new, type(None)):
        if fs_in_set.__len__() > 1:
            b_process = True
            fs_new = fs_in_set.min()
    else:
        if (sampling_frequency != fs_new).sum() > 0:
            b_process = True
        else:
            fs_new = fs_in_set.min()

    if b_process is True:
        for idx in range(x.__len__()):
            fs = sampling_frequency[idx]
            sig = x[idx]
            sig = decimate(sig, fs, fs_new)
            x[idx] = sig
            sampling_frequency[idx] = fs_new

    return x, fs_new

def fft_filter(X, fs, cutoff, type=''):
    Xs = fft.fft(X)
    freq = np.linspace(0, fs, Xs.shape[0])
    pos = np.where(freq > cutoff)[0][0]

    if type == 'lp':
        X_new = Xs
        X_new[pos:-pos] = 0
    elif type == 'hp':
        X_new = np.zeros_like(Xs)
        X_new[pos:-pos] = Xs[pos:-pos]
    X = np.real(fft.ifft(X_new))
    return X



class LowFrequencyFilter:
    __version__ = '0.0.1'
    def __init__(self, fs=None, cutoff=None, n_decimate=1, n_order=101, dec_cutoff=0.3, filter_type='lp'):
        self.fs = fs
        self.cutoff = cutoff
        self.n_decimate = n_decimate
        self.n_order = n_order
        self.dec_cutoff =dec_cutoff
        self.filter_type = filter_type

        self.n_append = None

        self.design_filters()

    def design_filters(self):
        self.n_append = (2 * self.n_order) * (2**self.n_decimate)

        self.a_dec = [1]
        self.b_dec = signal.firwin(self.n_order, self.dec_cutoff, pass_zero=True)
        self.b_dec /= self.b_dec.sum()

        self.a_filt = [1]
        self.b_filt = signal.firwin(self.n_order, 2 * self.cutoff / (self.fs/2**self.n_decimate), pass_zero=True)
        self.b_filt /= self.b_filt.sum()


    def decimate(self, X):
        X = signal.filtfilt(self.b_dec, self.a_dec, X)
        return X[::2]

    def upsample(self, X):
        #X_up = np.zeros(X.shape[0] * (2**self.n_decimate))
        #X_up[::self.n_decimate] = X
        #X_up = signal.filtfilt(self.b_dec, self.a_dec, X_up) * self.n_decimate

        X_up = np.zeros(X.shape[0] * 2)
        X_up[::2] = X
        X_up = signal.filtfilt(self.b_dec, self.a_dec, X_up) * 2
        return X_up

    def filter_signal(self, X):
        # append for filter
        X = np.concatenate((np.zeros(self.n_append), X, np.zeros(self.n_append)), axis=0)

        # append to divisible by 2
        C = int(2**np.ceil(np.log2(X.shape[0] + 2*self.n_append))) - X.shape[0]
        X = np.append(np.zeros(C), X)

        for k in range(self.n_decimate):
            X = self.decimate(X)

        X = signal.filtfilt(self.b_filt, self.a_filt, X)

        for k in range(self.n_decimate):
            X = self.upsample(X)
        #X = self.upsample(X)

        X = X[self.n_append + C : -self.n_append]
        return X


    def __call__(self, X):
        # append for filter
        X_orig = X.copy()
        X = self.filter_signal(X)
        if self.filter_type == 'lp': return X
        if self.filter_type == 'hp': return X_orig - X











