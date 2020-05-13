# Feature Extractor
Feture extractor contains a class SleepSpectralFeatureExtractor for extracting spectral  features for sleep classification from a raw signal.
The extractor extracts by default following features:
* Mean dominant frequency
* Spectral median frequency
* Spectral entropy
* Mean power spectral density
* Relative spectral density

The extractor also reports **Data rate** which gives a relative ratio of valid values in a pasted signal based on a number os NaN values.
The extractor requires input information about bands at which parameters will be calculated. Please see an example with a synthetic signal bellow.
Mean dominant frequency, spectral median frequency and spectral entropy are estimated from the frequency range specified by a minimum and maximum value within the all bands.
Spectral entropy is also estimated for each of the specified bands separately. Mean power spectral density is returned for each band.
Relative spectral density is normalized using an absolute mean power spectral density value on an interval defined by a minimum and maximum value within the all bands.


### Example 
```python
import sys
sys.path.append('D:\\MayoWork\\MSEL\\lib') # Set the source path to the lib folder of this python package.

import numpy as np
from FeatureExtractor import SleepSpectralFeatureExtractor

# Example synthetic signal generator
fs = 500 # sampling frequency
f = 10 # sin frequyency
a = 1 # amplitude
b = 0 # bias
t = np.arange(0, 1000, 1/fs)
x = a * np.sin(2*np.pi*f*t) + b


# Spectral Feature  Extraction
fs = 500 # sampling frequency of an analysed signal
segm_size = 30 # time length of a segment which is used for extraction of each feature
fbands = [[1, 4], # has to be int values
 [4, 8],
 [8, 12],
 [12, 14],
 [14, 20],
 [20, 30]] # frequency bands at which you want to extract features



Extractor = SleepSpectralFeatureExtractor() # init
feature_values, feature_names = Extractor(x=[x], fs=fs, segm_size=segm_size, fbands=fbands, n_processes=2)
```

If any features should be added or removed, see the information and the example bellow. The extractor instance estimates all features using a static methods of a SleepSpectralFeatureExtractor clas.
Therefore, any features can be added, or removed. Each method takes a unified set of input variables. *Pxx, bands, fs, segm_size*.
* Pxx - numpy.ndarray with dimensions [number_of_segments_to_classify, n_samples_spectrum] of spectral power densities (PSD) on an interval (0 Hz - fs/2>.
    * To get this PSDs you can use a built-in method *buffer* to cut your signal into specified segments and a method *PSD* to obtain PSD - returns already 1 sided PSD without a bias value.
* bands - list of bands, see the example above
* fs, segment_size - see above


```python
Extractor._extraction_functions = [
                                    SleepSpectralFeatureExtractor.normalized_entropy,
                                    SleepSpectralFeatureExtractor.MeanFreq,
                                    SleepSpectralFeatureExtractor.MedFreq,
                                    SleepSpectralFeatureExtractor.mean_bands,
                                    SleepSpectralFeatureExtractor.rel_bands,
                                    SleepSpectralFeatureExtractor.normalized_entropy_bands
                                  ]
```

