utils
^^^^^^^^

A Python package of tools and utilities which are nice to have working not only on this project.

feature_util
^^^^^^^^^^^^^^^^

Tools utilized in machine learning pipelines for feature augmentation, outlier detection, z-score normalization, balancing classes etc.


signal
^^^^^^^^^^^^^^^^

Tools for digital signal processing: fft filtering, low-frequency filtering utilizing downsampling etc.


types
^^^^^^^^^^^^^^^^

* ObjDict
    - Combines Matlab struct and python dict. Mirroring keys and attributes and its values. Nested initialization implemented as well.


.. code-block:: python

    from AISC.utils.types import ObjDict

    x = ObjDict()
    x['a'] = 1
    x.b = 2
    print(x['a'], x.a)
    print(x['b'], x.b)

    del x.b
    print(x)
    del x['a']

    x.a.b.c = 10 # creates nested ObjDicts
    x['b']['b']['c'] = 20
