import setuptools

from setuptools import Command, Extension
import shlex
import subprocess
import os
import re


## get version from file
VERSIONFILE="./AISC/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))



setuptools.setup(
    name="mselair-aisc",
    version=verstr,
    license='MFMER',
    url="https://github.com/mselair/AISC",

    author="Filip Mivalt",
    author_email="mivalt.filip@mayo.edu",


    description="Python package for EEG sleep classification and analysis.",
    long_description="Python package for EEG sleep classification and analysis. Developed by laboratory of Bioelectronics Neurophysiology and Engineering - Mayo Clinic",
    long_description_content_type="",

    packages=setuptools.find_packages(),




    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='>=3.6',
    install_requires =[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'pytz',
        'tqdm',
        'pymef',
        'pyedflib',
        'mef_tools'
    ]
)






