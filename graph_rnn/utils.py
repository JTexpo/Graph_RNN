from typing import Tuple, List

import numpy as np

def get_training_in_out(
    data: List[float], input_size: int, step_size: int
) -> Tuple[np.array, np.array]:
    """A function to turn 1 list into the input outputs needed for training

    Args:
        data (List[float]): the list that we have of Y points over X time
        input_size (int): how large the rnn input is
        step_size (int): how far into the future to make the prediction per model

    Returns:
        Tuple[List[float], List[float]]: training inputs and outputs
    """
    inputs = []
    outputs = []
    for index in range(len(data) - (input_size + step_size - 1) ):
        inputs.append(data[index : index + input_size])
        outputs.append(data[index + input_size + step_size - 1 ])

    return np.array(inputs), np.array(outputs)