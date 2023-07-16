import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

import numpy as np

from graph_rnn.utils import get_training_in_out

class UtilsTestCases(unittest.TestCase):
    def test_get_training_in_out_one(self):
        # INITALIZATION
        # -------------
        data = [1,2,3]
        input_size = 2
        step_size = 1

        # ACTION
        # ------
        inputs, outputs = get_training_in_out(data=data, input_size=input_size, step_size=step_size)
        
        # ASSERT
        # ------
        # should always get the number inserted
        expected_inputs = np.array([[1,2]])
        expected_outputs = np.array([3])

        self.assertTrue( np.array_equal(expected_inputs,inputs) )
        self.assertTrue( np.array_equal(expected_outputs, outputs) )
    
    def test_get_training_in_out_two(self):
        # INITALIZATION
        # -------------
        data = [1,2,3,4]
        input_size = 2
        step_size = 1

        # ACTION
        # ------
        inputs, outputs = get_training_in_out(data=data, input_size=input_size, step_size=step_size)
        
        # ASSERT
        # ------
        # should always get the number inserted
        expected_inputs = [[1,2],[2,3]]
        expected_outputs = [3,4]
        
        self.assertTrue( np.array_equal(expected_inputs,inputs) )
        self.assertTrue( np.array_equal(expected_outputs, outputs) )

if __name__ == "__main__":
    unittest.main()