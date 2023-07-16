import numpy as np

from graph_rnn.activation_function import ActivationFunction

class RNN_Layer:
    def __init__(self, size:int, activation_function:ActivationFunction, seed:int = 0):
        self.size = size
        self.activation_function = activation_function
        self.seed = seed

        if seed:
            np.random.seed(seed)

        # Output x Input
        self.weights = np.random.rand(1, size)
        # Because size: ( A, B ) dot ( B, C ) = ( A, C )
        # ex
        # ( 1, WEIGHT ) dot ( INPUT, ) = ( 1, )
        # When we are done and want to add, we need to add a ( 1, 1 ) with another ( 1, )
        self.bias = np.random.rand(1,1)

        # Propagation Values
        self.predictions = np.array([])
        self.inputs = np.array([])

def forward_propagation_rnn(rnn:RNN_Layer, inputs:np.array, step_size:int, return_rnn_size:bool = True)->np.array:
    """A function to preform forward propagation over a period of time

    Args:
        rnn (RNN_Layer): A recurrent neural network layer
        inputs (np.array): inputs into the layer
        step_size (int): how far in advance to predict
        return_rnn_size (bool, optional): Should the output be the same size of the input? Defaults to True.

    Raises:
        ValueError: input and weight are not matching sizes
        ValueError: stepsize is too small to preform forward propagation

    Returns:
        (np.array): The next layer or a list of predictions
    """

    # VALIDATION
    # ----------
    if np.shape(inputs)[-1] != np.shape(rnn.weights)[-1]:
        raise ValueError(f"[ERROR] Mismatching Size:\nInput {np.shape(inputs)}\nExpected {np.shape(rnn.weights)}")
    if step_size < 1:
        raise ValueError("[ERROR] step_size must be greater than 0")
    # INITALIZATION
    # -------------
    rnn.predictions = np.array([])
    rnn.inputs = np.copy(inputs)
    # ACTION
    # ------
    for _ in range(step_size):
        # size: ( A, B ) dot ( B, C ) = ( A, C )
        # ex.
        # ( 1, WEIGHT ) dot ( INPUT, 1) = ( 1, 1 )
        # 
        # bias is added to the 1 number 
        # 
        # Finally activation function is called returning 1 number for our new prediction        
        prediction = rnn.activation_function.function(np.dot(rnn.weights, inputs.T) + rnn.bias)
        # Rolling the inputs
        # ex.
        # [ 1, 2, 3 ] -> [ 2, 3, 1 ]
        inputs = np.array([np.roll(inputs[0], -1)])
        # replacing the rolled input so, our new input is continous
        # [ 2, 3, 1 ] (4) -> [ 2, 3, 4 ]
        inputs[0][-1] = prediction[0][0]

        rnn.predictions = np.append(rnn.predictions, prediction)
    # RETURN
    # ------
    if return_rnn_size:
        return np.copy(rnn.predictions[-return_rnn_size:])
    return np.copy(rnn.predictions)

def backwards_propagation_rnn( rnn:RNN_Layer, expected_output: float, learning_rate: float ) -> None:
    """A function to preform backwards propagation through time

    Args:
        rnn (RNN_Layer): A recurrent neural network layer to update
        expected_outputs (np.array): The outputs we were intented to produce

    Raises:
        ValueError: No predictions made
        ValueError: No inputs given
        ValueError: Mismatch in shape: expected vs predicted

    Returns:
        _type_: _description_
    """

    # VALIDATION
    # ----------
    if not rnn.predictions.size:
        raise ValueError("[ERROR] RNN has not made any predictions")
    if not rnn.inputs.size:
        raise ValueError("[ERROR] RNN has not had any inputs")
    
    # INITALIZATION
    # -------------
    weight_delta = np.zeros((1,rnn.size))
    bias_delta = np.zeros((1,1))

    # ACTION
    # ------
    # loss = (prediction - expected)^2
    error_derivatives = ( 2 * (rnn.predictions[-1] - expected_output) * rnn.activation_function.function_derivative(rnn.predictions[-1]) )
    error_derivatives = np.array([error_derivatives])

    for index in range(len(rnn.predictions)-1):
        # The weights and bias are adjusted each slightly each itteration
        # it's important to remember that even though an RNN can have many predictions, 
        # the network is still reusing the same bias
        epoch_input = np.append(rnn.inputs, rnn.predictions[ : len(rnn.predictions) - index ])[-rnn.size:]

        # size: ( SIZE, 1 ) dot ( 1, ) = ( SIZE, )
        weight_delta += ( np.dot(epoch_input, error_derivatives) * learning_rate / len(rnn.predictions) )
        # we just want the only value ( 1, )
        bias_delta += (error_derivatives[0] * learning_rate / len(rnn.predictions))[0]

        # size: ( 1, SIZE ).T = (SIZE, 1)
        # (SIZE, 1) dot ( 1, ) = ( SIZE, )
        # ( SIZE, ) * ( SIZE, ) = ( SIZE, )
        # pulling just the last elm, reshapes the error to
        # ( 1, )
        error_derivatives = np.dot( rnn.weights.T, error_derivatives ) * rnn.activation_function.function_derivative(epoch_input)
    
    # size: ( SIZE, 1 ) dot ( 1, ) = ( SIZE, )
    weight_delta += ( np.dot(rnn.inputs.T, error_derivatives).T * learning_rate / len(rnn.predictions) )
    # we just want the only value ( 1, )
    bias_delta += (error_derivatives * learning_rate / len(rnn.predictions))[0]

    # UPDATES
    # -------
    rnn.weights -= weight_delta
    rnn.bias -= bias_delta

    return None

def train_rnn(rnn:RNN_Layer, inputs:np.array, outputs:np.array, epoches:int = 0, learning_rate: float=.01, step_size:int = 1) -> None:

    for _ in range(epoches):
        for training_input, training_output in zip(inputs, outputs):
            _ = forward_propagation_rnn(rnn=rnn,inputs=np.array([training_input]),step_size=step_size)
            backwards_propagation_rnn(rnn=rnn, expected_output=np.array([training_output]), learning_rate=learning_rate )

    return None
