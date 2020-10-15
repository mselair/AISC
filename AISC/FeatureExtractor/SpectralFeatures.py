# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import multiprocessing
from functools import partial
import scipy as sp
import scipy.fft as fft
import scipy.stats as stats
import scipy.signal as signal

from AISC.utils.types import ObjDict
from AISC.utils.signal import buffer, LowFrequencyFilter, PSD


def normalized_entropy(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']

    subpsdx = Pxx[:, (freq >= bands.min()) & (freq <= bands.max())]
    return [
               stats.entropy(subpsdx ** 2, axis=1)
           ], [
               'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
           ]


def non_normalized_entropy(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']

    subpsdx = Pxx[:, (freq >= bands.min()) & (freq <= bands.max())]
    return [
               - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
           ], [
               'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
           ]


def mean_frequency(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']

    f = args.freq

    min_position = np.nanargmin(np.abs(f - bands.min()))
    max_position = np.nanargmin(np.abs(f - bands.max()))

    P = Pxx[:, min_position: max_position + 1]
    f = f[min_position: max_position + 1]

    f = np.reshape(f, (1, -1))
    pwr = np.sum(P, axis=1)
    mnfreq = np.dot(P, f.T).squeeze() / pwr
    return [mnfreq], ['MEAN_DOMINANT_FREQUENCY']


def median_frequency(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']


    pwr = np.sum(Pxx, axis=1)
    #f = 0.5 * fs * np.arange(1, Pxx.shape[1]) / Pxx.shape[1]
    f = args.freq
    min_position = np.nanargmin(np.abs(f - bands.min()))
    max_position = np.nanargmin(np.abs(f - bands.max()))

    P = Pxx[:, min_position: max_position + 1]
    f = f[min_position: max_position + 1]

    pwr05 = np.repeat(pwr / 2, P.shape[1]).reshape(P.shape)
    P = np.cumsum(np.abs(P), axis=1)

    medfreq_pos = np.argmax(np.diff(P > pwr05, axis=1), axis=1) + 1
    medfreq = f.squeeze()[medfreq_pos]
    return [medfreq], ['SPECTRAL_MEDIAN_FREQUENCY']


def mean_bands(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']


    outp_params = []
    outp_msg = []
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            np.nanmean(subpsdx, axis=1)
        )
        outp_msg.append('MEAN_PSD' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg


def relative_bands(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']


    outp_params = []
    outp_msg = []

    fullpsdx = np.nansum(Pxx[:, (freq >= bands.min()) & (freq <= bands.max())], axis=1)
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            np.nansum(subpsdx, axis=1) / fullpsdx
        )
        outp_msg.append('REL_PSD_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg


def normalized_entropy_bands(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']


    outp_params = []
    outp_msg = []
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            stats.entropy(subpsdx ** 2, axis=1)
        )
        outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg


def non_normalized_entropy_bands(args):
    Pxx = args['psd']
    bands = args['fbands']
    fs = args['fs']
    segm_size = args['segm_size']
    freq = args['freq']


    outp_params = []
    outp_msg = []
    for band in bands:
        subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
        outp_params.append(
            - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
        )
        outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
    return outp_params, outp_msg



