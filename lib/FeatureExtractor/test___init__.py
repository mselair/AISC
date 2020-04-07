# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from unittest import TestCase
from FeatureExtractor import SleepSpectralFeatureExtractor
import numpy as np

class TestFeatureExtractor(TestCase):
    #def test__init(self):
        #self.assertIsNone()


    def test_fs(self):
        testObj = SleepSpectralFeatureExtractor(unittest=True)
        valid_inputs = [1, 10, 1000]
        error_inputs = [0, 0.1, -10, '  ', None, np.nan, np.inf]

        # Test Error inputs
        for inp in error_inputs:
            with self.assertRaises(AssertionError, msg='Assertion error not raised for setter FeatureExtractor.segm_size and input: ' + str(inp)):
                testObj.fs = inp

        # Test Valid inputs
        for inp in valid_inputs:
            try:
                testObj.fs = inp
            except:
                raise self.fail("[UNEXPECTED ERROR] FeatureExtractor.segm_size has raised an error for value " + str(inp) + " of data type " + str(type(inp)))

        # Test matching input / output of setter
        for inp in valid_inputs:
            testObj.fs = inp
            self.assertEqual(testObj.fs, inp)



    def test_segm_size(self):
        self.longMessage = True
        testObj = SleepSpectralFeatureExtractor(unittest=True)
        valid_inputs = [1, 10, 1.5, 0.1]
        error_inputs = [0, -0.1, -10, 'RandomString', None, np.nan, np.inf]

        # Test Error inputs
        for inp in error_inputs:
            with self.assertRaises(AssertionError, msg='Assertion error not raised for setter FeatureExtractor.segm_size and input: ' + str(inp)):
                testObj.segm_size = inp

        # Test Valid inputs
        for inp in valid_inputs:
            try:
                testObj.segm_size = inp
            except:
                raise self.fail("[UNEXPECTED ERROR] FeatureExtractor.segm_size has raised an error for value " + str(inp) + " of data type " + str(type(inp)))

        # Test matching input / output of setter
        for inp in valid_inputs:
            testObj.segm_size = inp
            self.assertEqual(testObj.segm_size, inp)



