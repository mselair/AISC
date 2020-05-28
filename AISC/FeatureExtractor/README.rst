Sleep Feature Extractor
^^^^^^^^^^^^^^^^^^^^^^^

The Feature Extractor package contains SleepSpectralFeatureExtractor for spectral feature extraction from designed for sleep classification from a raw EEG signal.

Features implemented
^^^^^^^^^^^^^^^^^^^^^^^^^
* Mean dominant frequency
    - Calculates mean dominant frequency on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/signal/ref/meanfreq.html

* Spectral median frequency
    - Calculates median dominant frequency on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/signal/ref/medfreq.html

* Spectral entropy (Shannon Entropy)
    - Estimates Shannon Entropy of a spectrum on a frequency range defined as min-to-max of frequency bands at the input.
    - Source: https://www.mathworks.com/help/wavelet/ref/wentropy.html

* Mean power spectral density
    - Mean spectral power for each frequency band.

* Relative spectral density
    - Mean spectral power for each frequency band relative to the power of a whole spectrum defined on a frequency range defined as min-to-max of frequency bands at the input.



The extractor can return **Data rate** which gives a relative ratio of valid values in the input signal based on a number of NaN values. The extractor requires information about frequency bands at which parameters will be calculated. Please see an example bellow.

Mean dominant frequency, spectral median frequency and spectral entropy are estimated from the frequency range specified by a minimum and maximum value within the all bands.
Spectral entropy is also estimated for each of the specified bands separately. Mean power spectral density is returned for each band.
Relative spectral density is normalized using an absolute mean power spectral density value on an interval defined by a minimum and maximum value within the all bands.

Sources
^^^^^^^^^^^

This Feature Extractor implementation is based on the following papers (when use whole, parts, or are inspired by, we appreciate you acknowledge and refer these journal papers)


| Kremen, V., Duque, J. J., Brinkmann, B. H., Berry, B. M., Kucewicz, M. T., Khadjevand, F., G.A. Worrell, G. A. (2017). Behavioral state classification in epileptic brain using intracranial electrophysiology. Journal of Neural Engineering, 14(2), 026001. https://doi.org/10.1088/1741-2552/aa5688


| Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab


| Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61?70. https://doi.org/10.1016/j.jneumeth.2019.01.013


and on repository `Semi Automated Sleep Classifier <https://github.com/vkremen/Semi_Automated_Sleep_Classifier_iEEG>`_, see details in the original repository.

Example
^^^^^^^^^


.. code-block:: python

    import sys
    import numpy as np
    from AISC.FeatureExtractor import SleepSpectralFeatureExtractor

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
    fbands = [[1, 4],
     [4, 8],
     [8, 12],
     [12, 14],
     [14, 20],
     [20, 30]] # frequency bands at which you want to extract features



    Extractor = SleepSpectralFeatureExtractor() # init
    feature_values, feature_names = Extractor(x=[x], fs=fs, segm_size=segm_size, fbands=fbands, n_processes=2)


If any features should be added or removed, see the information and the example bellow. The extractor instance estimates all features using a static methods of a SleepSpectralFeatureExtractor clas. Therefore, any features can be added, or removed. Each method takes a unified set of input variables. *Pxx, bands, fs, segm_size*.
* Pxx - numpy.ndarray with dimensions [number_of_segments_to_classify, n_samples_spectrum] of spectral power densities (PSD) on an interval (0 Hz - fs/2>.

* To get this PSDs you can use a built-in method *buffer* to cut your signal into specified segments and a method *PSD* to obtain PSD - returns already 1 sided PSD without a bias value.
* bands - list of bands, see the example above
* fs, segment_size - see above

.. code-block:: python

    Extractor._extraction_functions = [
                                        SleepSpectralFeatureExtractor.normalized_entropy,
                                        SleepSpectralFeatureExtractor.MeanFreq,
                                        SleepSpectralFeatureExtractor.MedFreq,
                                        SleepSpectralFeatureExtractor.mean_bands,
                                        SleepSpectralFeatureExtractor.rel_bands,
                                        SleepSpectralFeatureExtractor.normalized_entropy_bands
                                      ]
