import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from graph_rnn.rnn_layer import RNN_Layer, forward_propagation_rnn, backwards_propagation_rnn, train_rnn
from graph_rnn.activation_function import ActivationFunction, leaky_relu, leaky_relu_derivative

class RNNTestCase(unittest.TestCase):

    def test_forward_propagation_rnn_one(self):
        # INITALIZATION
        # -------------
        rnn = RNN_Layer(
            size=1,
            activation_function=ActivationFunction(
            function=leaky_relu,
            function_derivative=leaky_relu_derivative
            ),
            seed=1
        )
        rnn.bias = np.array([0.])
        rnn.weights = np.array([[1.]])

        inputs = np.array([[1.]])
        step_size = 1

        # ACTION
        # ------
        predictions = forward_propagation_rnn(rnn=rnn, inputs=inputs, step_size=step_size)

        # ASSERT
        # ------
        # should always get the number inserted
        expected = [1.]
        self.assertTrue( predictions == expected)
    
    def test_forward_propagation_rnn_two(self):
        # INITALIZATION
        # -------------
        rnn = RNN_Layer(
            size=2,
            activation_function=ActivationFunction(
            function=leaky_relu,
            function_derivative=leaky_relu_derivative
            ),
            seed=1
        )
        rnn.bias = np.array([0.])
        rnn.weights = np.array([[1.,1.]])

        inputs = np.array([[2.,3.]])
        step_size = 1

        # ACTION
        # ------
        predictions = forward_propagation_rnn(rnn=rnn, inputs=inputs, step_size=step_size)

        # ASSERT
        # ------
        # because 2 + 3 = 5
        expected = [5.]
        self.assertTrue( predictions == expected)
    
    def test_backwards_propagation_rnn_one_no_change(self):
        # INITALIZATION
        # -------------
        rnn = RNN_Layer(
            size=1,
            activation_function=ActivationFunction(
            function=leaky_relu,
            function_derivative=leaky_relu_derivative
            ),
            seed=1
        )
        rnn.bias = np.array([[.0]])
        rnn.weights = np.array([[1.0]])

        inputs = np.array([[1.]])
        output = np.array([1.])
        step_size = 1

        # ACTION
        # ------
        _ = forward_propagation_rnn(rnn=rnn, inputs=inputs, step_size=step_size)
        backwards_propagation_rnn(rnn=rnn, expected_output=output,learning_rate=1)

        # ASSERT
        # ------
        self.assertTrue( rnn.weights ==  np.array([[1.0]]) )
        self.assertTrue( rnn.bias ==  np.array([[.0]]) )
    
    def test_backwards_propagation_rnn_one_change(self):
        # INITALIZATION
        # -------------
        rnn = RNN_Layer(
            size=1,
            activation_function=ActivationFunction(
            function=leaky_relu,
            function_derivative=leaky_relu_derivative
            ),
            seed=1
        )
        rnn.bias = np.array([[.0]])
        rnn.weights = np.array([[1.0]])

        inputs = np.array([[1.]])
        output = np.array([.5])
        step_size = 1

        # ACTION
        # ------
        _ = forward_propagation_rnn(rnn=rnn, inputs=inputs, step_size=step_size)
        backwards_propagation_rnn(rnn=rnn, expected_output=output,learning_rate=.5)

        # ASSERT
        # ------
        self.assertTrue( rnn.weights ==  np.array([[.5]]) )
        self.assertTrue( rnn.bias ==  np.array([[-.5]]) )
    
    def test_train_rnn(self):
        # INITALIZATION
        # -------------
        rnn = RNN_Layer(
            size=8,
            activation_function=ActivationFunction(
            function=leaky_relu,
            function_derivative=leaky_relu_derivative
            ),
            seed=1
        )

        inputs = np.array([[1.,0.,1.,0.,1.,0.,1.,0.],[0.,1.,0.,1.,0.,1.,0.,1.0]])
        outputs = np.array([1.,0.])

        epoches = 20
        learning_rate = .1
        step_size = 1

        # ACTION
        # ------
        train_rnn(rnn=rnn,inputs=inputs,outputs=outputs,epoches=epoches,learning_rate=learning_rate,step_size=step_size)

        # ASSERT
        # ------
        one_prediction = forward_propagation_rnn(rnn=rnn,inputs=np.array([[1.,0.,1.,0.,1.,0.,1.,0.]]),step_size=step_size)
        zero_prediction = forward_propagation_rnn(rnn=rnn,inputs=np.array([[0.,1.,0.,1.,0.,1.,0.,1.]]),step_size=step_size)

        self.assertTrue( round(one_prediction[0],5) == 1 )
        self.assertTrue( round(zero_prediction[0],5) == 0 )


if __name__ == "__main__":
    unittest.main()