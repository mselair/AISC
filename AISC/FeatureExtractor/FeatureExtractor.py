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

from AISC.utils.types import ObjDict

class SleepSpectralFeatureExtractor:
    __version__ = "0.2.0"
    """
    Spectral feature extractor designed for sleep classification
    
    ...
    
    Attributes
    ----------
    extraction_functions : list
        List of functions for parameter extraction from np.ndarray where shape is (n_spectrums, spectrum_samples).
        Each function has to return tuple (features : list, name_of_features : list). By default set to extract all features within this class.
    
    ...
    
    Methods
    --------
            
    ...
    
    Notes
    ------
    v0.1.0 Updates
    - communication between functions (Pxx, fs, ...) changed to ObjDict - see AISC.types.ObjDict
    - float value frequency bands enabled
    v0.1.1 Updates
    - bands_to_erase as an input into __call__ - erases defined bands of psd
    v0.1.2 Updates
    - self._extraction_functions init moved to __init__    
    v0.2.0
        - methods buffer, PSD transferred into AISC.utils.signal
        - PSD replaced by welch's method implementation by scipy.signal.welch
        - implementation of spectral feature extraction from shorter segments and average them
        - removed filtering due switch to welch method of periodogram - constant detrending
            - can be replaced with LowFrequencyFilter later on
        - automatic windown implemented in welch method - hann window
    """
    #TODO: move feature extraction functions into FeatureExtractor.SpectralFeatures and call them from the methods

    def __init__(self):
        self._extraction_functions = [self.normalized_entropy, self.MeanFreq, self.MedFreq, self.mean_bands,
                                      self.rel_bands, self.normalized_entropy_bands]

    @staticmethod
    def _verify_input_fs(item):
        if not isinstance(item, (int, float)):
            raise TypeError('[INPUT TYPE ERROR] Sampling frequency \"fs\" has to be an integer or float!')
        if not item > 0:
            raise ValueError(
                '[INPUT VALUE ERROR] Sampling frequency is required to be higher than 0! Pasted value: ' + str(item))
        return item

    @staticmethod
    def _verify_input_segm_size(item):
        if not isinstance(item, (int, float)):
            raise TypeError(
                '[INPUT TYPE ERROR] A segment size \"segm_size\" is required to be an integer or float. Parsed data type is ' + str(
                    type(item)))
        if not item > 0:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" is required to be  higher than 0!')
        if item == np.inf:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" cannot be Inf')
        return item

    @staticmethod
    def _verify_input_fbands(item):
        if not isinstance(item, (list, np.ndarray)):
            raise TypeError(
                '[INPUT TYPE ERROR] fbands variable has to be of a list or numpy.array type. Pasted value: ' + str(
                    type(item)))
        if not item.__len__() > 0:
            raise ValueError(
                '[INPUT SIZE ERROR] Length of fbands has to be > 0. Current length: ' + str(item.__len__()))
        for idx, subitem in enumerate(item):
            if not subitem.__len__() == 2:
                raise TypeError(
                    '[INPUT SIZE ERROR] Length of each frequency band in fband variable has to contain exactly 2 elements min and max frequency for a given bandwidth. Current size: ' + str(
                        subitem.__len__()))
            if not subitem[0] < subitem[1]:
                raise ValueError('[INPUT VALUE ERROR] For a bandwidth in variable fbands with index ' + str(
                    idx) + ' an error has been found. The first value has to be lower than the second one! Current input: ' + str(
                    subitem))
        return np.array(item)

    @staticmethod
    def _verify_input_x(item):
        if not isinstance(item, (np.ndarray, list)):
            raise TypeError(
                '[INPUT TYPE ERROR] An input signal has to be a type of list or numpy.ndarray. Pasted ' + str(
                    type(item)) + ' instead.')

        if isinstance(item, np.ndarray):
            if not (item.shape.__len__() == 1 or item.shape.__len__() == 2):
                raise TypeError(
                    '[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, np.ndarray):
                    raise TypeError(
                        '[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        return item

    @staticmethod
    def _verify_input_n_processes(item):
        if not isinstance(item, int):
            raise TypeError('[INPUT TYPE ERROR] Input n_processes has to be of a type int. Type ' + str(
                type(input)) + ' has found instead.')
        if item < 1:
            raise ValueError(
                '[INPUT VALUE ERROR] Number of processes dedicated to feature extraction should be > than 0.')
        if item > multiprocessing.cpu_count() / 2:
            raise PendingDeprecationWarning(
                '[INPUT VALUE ERROR] Number of processes dedicated to feature extraction shouldn\'t be higher than half of the number of processors. This can significantly slow down the processing time and decrease performance. Value is decreased to a number ' + str(
                    multiprocessing.cpu_count() / 2))
            return int(multiprocessing.cpu_count() / 2)
        return item

    def _verify_extractor_functions(self):
        if self._extraction_functions.__len__() < 1:
            raise TypeError('')

        for idx, func in enumerate(self._extraction_functions):
            if not callable(func):
                raise TypeError('[FUNCTION ERROR] A feature extraction function ' + str(func) + ' with an index ' + str(
                    idx) + ' is not callable')

    def __call__(
            self,
            x: (np.ndarray, list) = None,
            fs: float = None,
            segm_size: float = None,
            sperwelchseg: float = None,
            soverlapwelchseg: float = 0,
            fbands: list = None,
            bands_to_erase: list = [],
            datarate: bool = True,
            n_processes: int = 1
    ):

        """

        Parameters
        ----------
        x : np.ndarray
        fs
        segm_size
        sub_segment_size
        sub_segment_overlap
        fbands
        bands_to_erase
        datarate
        n_processes

        Returns
        -------

        """
        # Standard parameters
        x = self._verify_input_x(x)
        fs = self._verify_input_fs(fs)
        segm_size = self._verify_input_segm_size(segm_size)
        fbands = self._verify_input_fbands(fbands)
        n_processes = self._verify_input_n_processes(n_processes)

        if isinstance(x, np.ndarray):
            return self.process_signal(x=x, fs=fs, segm_size=segm_size, sperwelchseg=sperwelchseg, soverlapwelchseg=soverlapwelchseg, fbands=fbands, datarate=datarate,
                                       bands_to_erase=bands_to_erase)

        if isinstance(x, list) and x.__len__() == 1:
            return self.process_signal(x=x[0], fs=fs, segm_size=segm_size, sperwelchseg=sperwelchseg, soverlapwelchseg=soverlapwelchseg, fbands=fbands, datarate=datarate,
                                       bands_to_erase=bands_to_erase)

        else:
            if n_processes == 1:
                output = []
                for signal in x:
                    out_tuple = self.process_signal(x=signal, fs=fs, segm_size=segm_size, sperwelchseg=sperwelchseg, soverlapwelchseg=soverlapwelchseg, fbands=fbands,
                                                    datarate=datarate, bands_to_erase=bands_to_erase)
                    output.append(out_tuple)
                return output
            else:
                with multiprocessing.Pool(n_processes) as p:
                    pfunc = partial(self.process_signal, fs=fs, segm_size=segm_size, sperwelchseg=sperwelchseg, soverlapwelchseg=soverlapwelchseg, fbands=fbands, datarate=datarate,
                                    bands_to_erase=bands_to_erase)
                    output = p.map(pfunc, x)
                return output

    def process_signal(self, x=None, fs=None, segm_size=None, sperwelchseg=None, soverlapwelchseg=0,
                       fbands=None, bands_to_erase=[], datarate=False):
        """

        Parameters
        ----------
        x
        fs
        segm_size
        sub_segment_size
        sub_segment_overlap
        fbands
        bands_to_erase
        datarate

        Returns
        -------

        """
        x = x.copy().squeeze()
        features = []
        msg = []
        #cutoff = np.array(fbands).max() + 15
        #b, a = signal.butter(4, cutoff / (0.5 * fs), 'lp', analog=False)

        xbuffered = buffer(x, fs, segm_size)
        if datarate is True:
            features = features + [1 - (np.isnan(xbuffered).sum(axis=1) / (segm_size * fs))]
            msg = msg + ['DATA_RATE']
        xbuffered = xbuffered - np.nanmean(xbuffered, axis=1).reshape((-1, 1))
        xbuffered[np.isnan(xbuffered)] = 0
        #xbuffered = signal.filtfilt(b, a, xbuffered, axis=1)


        if isinstance(sperwelchseg, type(None)):
            soverlapwelchseg = 0
        else:
            sperwelchseg = int(np.round(sperwelchseg * fs))
            soverlapwelchseg = int(np.round(soverlapwelchseg * fs))
        freq, psd = PSD(xbuffered, fs, nperseg=sperwelchseg, noverlap=soverlapwelchseg)
        freq = freq[1:] # remove 0Hz sample
        psd = psd[:, 1:]

        if bands_to_erase.__len__() > 0:
            for eband in bands_to_erase:
                psd[:, (freq > eband[0]) & (freq < eband[1])] = 0

        inp_params = ObjDict({
            'psd': psd,
            'fs': fs,
            'fbands': fbands,
            'segm_size': segm_size,
            'freq': freq
        })

        for func in self._extraction_functions:
            feature, ftr_name = func(inp_params)
            features = features + feature
            msg = msg + ftr_name
        return features, msg

    @property
    def extraction_functions(self):
        return self._extraction_functions

    @extraction_functions.setter
    def extraction_functions(self, item:list):
        self._extraction_functions = item
        self._verify_extractor_functions()

    @staticmethod
    def normalized_entropy(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size
        freq = args.freq

        #subpsdx = Pxx[:, int(round(bands.min() * segm_size)): int(round(bands.max() * segm_size)) + 1]
        subpsdx = Pxx[:, (freq >= bands.min()) & (freq <= bands.max())]
        return [
                   stats.entropy(subpsdx ** 2, axis=1)
               ], [
                   'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
               ]

    @staticmethod
    def non_normalized_entropy(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size
        freq = args.freq

        #subpsdx = Pxx[:, int(round(bands.min() * segm_size)): int(round(bands.max() * segm_size)) + 1]
        subpsdx = Pxx[:, (freq >= bands.min()) & (freq <= bands.max())]
        return [
                   - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
               ], [
                   'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
               ]

    @staticmethod
    def MeanFreq(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size

        #f = 0.5 * fs * np.arange(1, Pxx.shape[1]) / Pxx.shape[1]
        f = args.freq

        min_position = np.nanargmin(np.abs(f - bands.min()))
        max_position = np.nanargmin(np.abs(f - bands.max()))

        P = Pxx[:, min_position: max_position + 1]
        f = f[min_position: max_position + 1]

        f = np.reshape(f, (1, -1))
        pwr = np.sum(P, axis=1)
        mnfreq = np.dot(P, f.T).squeeze() / pwr
        return [mnfreq], ['MEAN_DOMINANT_FREQUENCY']

    @staticmethod
    def MedFreq(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size

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

    @staticmethod
    def mean_bands(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size
        freq = args.freq

        outp_params = []
        outp_msg = []
        for band in bands:
            #subpsdx = Pxx[:, int(round(band[0] * segm_size)):int(round(band[1] * segm_size)) + 1]
            subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
            outp_params.append(
                np.nanmean(subpsdx, axis=1)
            )
            outp_msg.append('MEAN_PSD' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

    @staticmethod
    def rel_bands(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size
        freq = args.freq

        outp_params = []
        outp_msg = []
        #fullpsdx = np.nansum(Pxx[:, int(round(bands.min() * segm_size)): int(round(bands.max() * segm_size)) + 1],
        #                     axis=1)

        fullpsdx = np.nansum(Pxx[:, (freq >= bands.min()) & (freq <= bands.max())], axis=1)
        for band in bands:
            #subpsdx = Pxx[:, int(round(band[0] * segm_size)):int(round(band[1] * segm_size)) + 1]
            subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
            outp_params.append(
                np.nansum(subpsdx, axis=1) / fullpsdx
            )
            outp_msg.append('REL_PSD_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

    @staticmethod
    def normalized_entropy_bands(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size
        freq = args.freq

        outp_params = []
        outp_msg = []
        for band in bands:
            #subpsdx = Pxx[:, int(round(band[0] * segm_size)):int(round(band[1] * segm_size)) + 1]
            subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
            outp_params.append(
                stats.entropy(subpsdx ** 2, axis=1)
            )
            outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

    @staticmethod
    def non_normalized_entropy_bands(args):
        Pxx = args.psd
        bands = args.fbands
        fs = args.fs
        segm_size = args.segm_size
        freq = args.freq

        outp_params = []
        outp_msg = []
        for band in bands:
            #subpsdx = Pxx[:, int(round(band[0] * segm_size)):int(round(band[1] * segm_size)) + 1]
            subpsdx = Pxx[:, (freq >= band[0]) & (freq <= band[1])]
            outp_params.append(
                - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
            )
            outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg








