import setuptools

from setuptools import Command, Extension
import shlex
import subprocess
import os


setuptools.setup(
    name="mselair-aisc",
    version="0.0.1",
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
        "Development Status :: Alpha"
    ],
    python_requires='>=3.6',
    install_requires =[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'pymef',
        'scikit-learn',
    ]
)






