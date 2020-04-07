# Copyright 2020-present, Mayo Clinic Department of Neurology
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

class SleepSpectralFeatureExtractor:
    """

    """
    @property
    def fs(self):
        """
        Sampling frequency getter

        :return: Int value of sampling frequency
        """
        return self._fs

    @fs.setter
    def fs(self, value: int):
        """
        Sampling frequency setter

        :param value: Sampling frequency of processed data.
        :return: None
        """
        self._verify_input_fs(value)
        self._fs = value

    @property
    def segm_size(self):
        """
        Segmentation time window size getter

        :return: Int/Float value of the size of a segmentation window in seconds.
        """
        return self._segm_size

    @segm_size.setter
    def segm_size(self, t):
        """
        Segmentation time window size setter

        :param value: (float or int) Size of the segmentation window for parameter extraction in seconds.
        :return:  None
        """
        self._verify_input_segm_size(t)
        self._segm_size = t

    @property
    def fbands(self):
        return self._fbands

    @fbands.setter
    def fbands(self, bands):
        self._verify_input_fbands(bands)
        self._fbands = np.array(bands)

    @property
    def n_processes(self):
        return self._n_processes

    @n_processes.setter
    def n_processes(self, item):
        self._verify_input_n_processes(item)
        self._n_processes = item

    @staticmethod
    def _verify_input_fs(item):
        if not isinstance(item, int):
            raise TypeError('[INPUT TYPE ERROR] Sampling frequency \"fs\" has to be an integer!')
        if not item > 0:
            raise ValueError('[INPUT VALUE ERROR] Sampling frequency is required to be higher than 0! Pasted value: ' + str(item))

        return item

    @staticmethod
    def _verify_input_segm_size(item):
        if not isinstance(item, (int, float)):
            raise TypeError('[INPUT TYPE ERROR] A segment size \"segm_size\" is required to be an integer or float. Parsed data type is ' + str(type(item)))
        if not item > 0:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" is required to be  higher than 0!')
        if item == np.inf:
            raise ValueError('[INPUT VALUE ERROR] A segment size \"segm_size\" cannot be Inf')
        return item

    @staticmethod
    def _verify_input_fbands(item):
        if not isinstance(item, (list, np.ndarray)):
            raise TypeError('[INPUT TYPE ERROR] fbands variable has to be of a list or numpy.array type. Pasted value: ' + str(type(item)))
        if not item.__len__() > 0:
            raise ValueError('[INPUT SIZE ERROR] Length of fbands has to be > 0. Current length: ' + str(item.__len__()))
        for idx, subitem in enumerate(item):
            if not subitem.__len__() == 2:
                raise TypeError('[INPUT SIZE ERROR] Length of each frequency band in fband variable has to contain exactly 2 elements min and max frequency for a given bandwidth. Current size: ' + str(subitem.__len__()))
            if not subitem[0] < subitem[1]:
                raise ValueError('[INPUT VALUE ERROR] For a bandwidth in variable fbands with index ' + str(idx) + ' an error has been found. The first value has to be lower than the second one! Current input: ' + str(subitem))
        return np.array(item)

    @staticmethod
    def _verify_input_x(item):
        if not isinstance(item, (np.ndarray, list)):
            raise TypeError('[INPUT TYPE ERROR] An input signal has to be a type of list or numpy.ndarray. Pasted ' + str(type(item)) + ' instead.')

        if isinstance(item, np.ndarray):
            if not (item.shape.__len__() == 1 or item.shape.__len__() == 2):
                raise TypeError('[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, np.ndarray):
                    raise TypeError('[INPUT SIZE ERROR] An input signal has to consist of an input of a single dimension for a single signal, 2D numpy.ndarray field for multiple signals (n_signal, signal_length), or list containing multiple fields with a single signal in each of these cells.')

        return item


    @staticmethod
    def _verify_input_n_processes(item):
        if not isinstance(item, int):
            raise TypeError('[INPUT TYPE ERROR] Input n_processes has to be of a type int. Type ' + str(type(input)) + ' has found instead.')
        if item < 1:
            raise ValueError('[INPUT VALUE ERROR] Number of processes dedicated to feature extraction should be > than 0.')
        if item > multiprocessing.cpu_count() / 2:
            raise PendingDeprecationWarning('[INPUT VALUE ERROR] Number of processes dedicated to feature extraction shouldn\'t be higher than half of the number of processors. This can significantly slow down the processing time and decrease performance. Value is decreased to a number ' + str(multiprocessing.cpu_count() / 2))
            return int(multiprocessing.cpu_count() / 2)
        return item

    def _verify_standard_parameters(self):
        self._fs = self._verify_input_fs(self._fs)
        self. _segm_size = self._verify_input_segm_size(self._segm_size)
        self._fbands = self._verify_input_fbands(self._fbands)
        self._n_processes = self._verify_input_n_processes(self._n_processes)

    def _verify_extractor_functions(self):
        if self._extraction_functions.__len__() < 1:
            raise TypeError('')

        for idx, func in enumerate(self._extraction_functions):
            if not callable(func):
                raise TypeError('[FUNCTION ERROR] A feature extraction function ' + str(func) + ' with an index ' + str(idx) + ' is not callable')


    def __call__(self, x=None, fs=None, segm_size=None, fbands=None, n_processes=1):
        self._extraction_functions = [self.normalized_entropy, self.MeanFreq, self.MedFreq, self.mean_bands, self.rel_bands, self.normalized_entropy_bands]


        # Standard parameters

        x = self._verify_input_x(x)
        fs = self._verify_input_fs(fs)
        segm_size = self._verify_input_segm_size(segm_size)
        fbands = self._verify_input_fbands(fbands)
        n_processes = self._verify_input_n_processes(n_processes)

        # Input
        #self._verify_standard_parameters()

        if isinstance(x, np.ndarray):
            return self.process_signal(x=x, fs=fs, segm_size=segm_size, fbands=fbands)

        if isinstance(x, list) and x.__len__() == 1:
            return self.process_signal(x=x[0], fs=fs, segm_size=segm_size, fbands=fbands)

        else:
            if n_processes == 1:
                output = []
                for signal in x:
                    out_tuple = self.process_signal(x=signal, fs=fs, segm_size=segm_size, fbands=fbands)
                    output.append(out_tuple)
                return output
            else:
                with multiprocessing.Pool(n_processes) as p:
                    pfunc = partial(self.process_signal, fs=fs, segm_size=segm_size, fbands=fbands)
                    output = p.map(pfunc, x)
                return output


    def process_signal(self, x=None, fs=None, segm_size=None, fbands=None):
        features = []
        msg = []
        cutoff = np.array(fbands).max()
        b, a = signal.butter(4, cutoff/(0.5*fs), 'lp', analog=False)


        xbuffered = self.buffer(x, fs, segm_size)
        #print(xbuffered.shape)
        features = features + [1 - (np.isnan(xbuffered).sum(axis=1)  / (segm_size * fs))]
        msg = msg + ['DATA_RATE']
        xbuffered[np.isnan(xbuffered)] = 0
        xbuffered = xbuffered - xbuffered.mean(axis=1).reshape((-1, 1))
        xbuffered = signal.filtfilt(b, a, xbuffered, axis=1)


        psd = self.PSD(xbuffered, fs)

        for func in self._extraction_functions:
            feature, ftr_name = func(psd, fbands, fs, segm_size)
            features = features + feature
            msg = msg + ftr_name

        return features, msg

    @staticmethod
    def buffer(x, fs, segm_size):
        n_segm_size = fs * segm_size
        residuum = x.shape[0] % n_segm_size
        if residuum > 0:
            x = np.append(x, np.zeros(n_segm_size - residuum))

        return x.reshape((-1, n_segm_size))

    @staticmethod
    def PSD(xbuffered, fs):
        N = xbuffered.shape[1]
        psdx = fft.fft(xbuffered, axis=1)
        psdx = psdx[:, 1:int(np.round(N / 2)) + 1]

        psdx = (1 / (fs * N)) * np.abs(psdx) ** 2
        psdx[np.isinf(psdx)] = np.nan
        return psdx

    @staticmethod
    def normalized_entropy(Pxx, bands, fs, segm_size):
        subpsdx = Pxx[:, bands.min()*segm_size : bands.max()*segm_size + 1]
        return [
                   stats.entropy(subpsdx ** 2, axis=1)
               ], [
            'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
        ]

    @staticmethod
    def non_normalized_entropy(Pxx, bands, fs, segm_size):
        subpsdx = Pxx[:, bands.min()*segm_size : bands.max()*segm_size + 1]
        return [
                   - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
               ], [
            'SPECTRAL_ENTROPY_' + str(bands.min()) + '-' + str(bands.max()) + 'Hz'
        ]


    @staticmethod
    def MeanFreq(Pxx, bands, fs, segm_size):
        f = 0.5 * fs * np.arange(1, Pxx.shape[1]) / Pxx.shape[1]

        min_position = np.nanargmin(np.abs(f - bands.min()))
        max_position = np.nanargmin(np.abs(f - bands.max()))

        P = Pxx[:, min_position : max_position + 1]
        f = f[min_position : max_position + 1]

        f = np.reshape(f, (1, -1))
        pwr = np.sum(P, axis=1)
        mnfreq = np.dot(P, f.T).squeeze() / pwr
        return [mnfreq], ['MEAN_DOMINANT_FREQUENCY']

    @staticmethod
    def MedFreq(Pxx, bands, fs, segm_size):
        pwr = np.sum(Pxx, axis=1)

        f = 0.5 * fs * np.arange(1, Pxx.shape[1]) / Pxx.shape[1]
        min_position = np.nanargmin(np.abs(f - bands.min()))
        max_position = np.nanargmin(np.abs(f - bands.max()))

        P = Pxx[:, min_position : max_position + 1]
        f = f[min_position : max_position + 1]

        pwr05 = np.repeat(pwr / 2, P.shape[1]).reshape(P.shape)
        P = np.cumsum(np.abs(P), axis=1)

        medfreq_pos = np.argmax(np.diff(P > pwr05, axis=1), axis=1) + 1
        medfreq = f.squeeze()[medfreq_pos]
        return [medfreq], ['SPECTRAL_MEDIAN_FREQUENCY']

    @staticmethod
    def mean_bands(P, bands, fs, segm_size):
        outp_params = []
        outp_msg = []
        for band in bands:
            subpsdx = P[:, band[0]*segm_size:band[1]*segm_size + 1]
            outp_params.append(
                np.nanmean(subpsdx, axis=1)
            )
            outp_msg.append('MEAN_PSD' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

    @staticmethod
    def rel_bands(P, bands, fs, segm_size):
        outp_params = []
        outp_msg = []
        fullpsdx = np.nansum(P[:, bands.min()*segm_size:bands.max()*segm_size + 1], axis=1)
        for band in bands:
            subpsdx = P[:, band[0]*segm_size:band[1]*segm_size + 1]
            outp_params.append(
                np.nansum(subpsdx, axis=1) / fullpsdx
            )
            outp_msg.append('REL_PSD_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

    @staticmethod
    def normalized_entropy_bands(P, bands, fs, segm_size):
        outp_params = []
        outp_msg = []
        for band in bands:
            subpsdx = P[:, band[0]*segm_size:band[1]*segm_size + 1]
            outp_params.append(
                stats.entropy(subpsdx ** 2, axis=1)
            )
            outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

    @staticmethod
    def non_normalized_entropy_bands(P, bands, fs, segm_size):
        outp_params = []
        outp_msg = []
        for band in bands:
            subpsdx = P[:, band[0]*segm_size:band[1]*segm_size + 1]
            outp_params.append(
                - np.sum(subpsdx ** 2 * np.log(subpsdx ** 2), axis=1)
            )
            outp_msg.append('SPECTRAL_ENTROPY_' + str(band[0]) + '-' + str(band[1]) + 'Hz')
        return outp_params, outp_msg

