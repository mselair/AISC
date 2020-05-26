![Build Status](https://travis-ci.com/mselair/AISC.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/mselair-aisc/badge/?version=latest)](https://mselair-aisc.readthedocs.io/en/latest/?badge=latest)

# Automated iEEG Sleep Classifier (AISC)
version: 0.0.1 - alpha

This is a pre-release of tools for our collaborators. Any reports, suggestions, or bug reports are welcomed.

See readme files for each subpackage in the AISC folder - AISC.FeatureExtractor - README

## Contains
- Simplified python wrapper for [Multiscale Electrophysiology Format (MEF)](https://github.com/msel-source/meflib) using [pymef](https://github.com/msel-source/pymef) - see [pymef documentation](https://pymef.readthedocs.io/en/latest/).

- EEG Sleep Feature Extractor - Based on [Semi_Automated_Sleep_Classifier](https://github.com/vkremen/Semi_Automated_Sleep_Classifier_iEEG)

- WaveDetector - Wave detector designed for Delta Wave detection


## Installation:

```bash
pip install mselair-aisc
```

## Documentation
Documentation will be added in next release.


## Acknowledgment
When use whole, parts, or are inspired by, we appreciate you acknowledge and refer these journal papers: 

Kremen, V., Duque, J. J., Brinkmann, B. H., Berry, B. M., Kucewicz, M. T., Khadjevand, F., G.A. Worrell, G. A. (2017). Behavioral state classification in epileptic brain using intracranial electrophysiology. Journal of Neural Engineering, 14(2), 026001. https://doi.org/10.1088/1741-2552/aa5688

Kremen, V., Brinkmann, B. H., Van Gompel, J. J., Stead, S. (Matt) M., St Louis, E. K., & Worrell, G. A. (2018). Automated Unsupervised Behavioral State Classification using Intracranial Electrophysiology. Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/aae5ab

Gerla, V., Kremen, V., Macas, M., Dudysova, D., Mladek, A., Sos, P., & Lhotska, L. (2019). Iterative expert-in-the-loop classification of sleep PSG recordings using a hierarchical clustering. Journal of Neuroscience Methods, 317(February), 61?70. https://doi.org/10.1016/j.jneumeth.2019.01.013






