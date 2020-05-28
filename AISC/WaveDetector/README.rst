Wave Detector
^^^^^^^^^^^^^^^^^^^^^^

WaveDetector object detects minimums and maximums of waves in a given frequency bandpass. Designed mainly to detect slow (delta - 0.5 - 4 Hz) waves. WaveDetector can also return statistical report of detected waves such as Δt, slope, peak2peak and min and max values.

Example
^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from AISC.WaveDetector import WaveDetector

    fs = 200
    signal_length = 5 # s
    wave_freq1 = 2
    wave_freq2 = 1
    noise_level = 0.5

    t = np.arange(0, signal_length * fs) / fs
    noise = np.random.randn(t.shape[0]) * noise_level
    x1 = np.sin(2*np.pi * wave_freq1 * t)
    x2 = np.sin(2*np.pi * wave_freq2 * t) * 0.5
    x_noise = x1 + x2 + noise

    WDet = WaveDetector(fs=fs, cutoff_low=0.5, cutoff_high=4)

    stats, det = WDet(x_noise)
    min_pos = det['min_pos']
    min_val = det['min_val']
    max_pos = det['max_pos']
    max_val = det['max_val']

    #plt.plot(t, x_noise)
    #plt.xlabel('t [s]')
    #plt.title('Original')
    #plt.show()

    plt.plot(t, x_noise)
    plt.stem(t[min_pos], min_val, 'r')
    plt.stem(t[max_pos], max_val, 'k')
    plt.title('Detections')
    plt.show()

    print('Slope Stats')
    print(stats['slope_stats'])

    print('Peak2Peak Stats')
    print(stats['pk2pk_stats'])

    print('delta_t Stats')
    print(stats['delta_t_stats'])

