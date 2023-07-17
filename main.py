from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pyscript import Element, display

from graph_rnn.utils import get_training_in_out
from graph_rnn.rnn_layer import RNN_Layer, forward_propagation_rnn, train_rnn
from graph_rnn.activation_function import (
    ActivationFunction,
    linear,
    linear_derivative,
)

# INITALIZING ELEMENTS
# --------------------
RNN_GRAPH_ID = "rnn-graph"
RNN_SIZE = 6
RNN_TRAIN_MIN_SIZE = 9
RNN_SEED = 1
RNN_EPOCHES = 100
RNN_FORWARD_PREDICT = 10
new_coordinate = Element("new-coordinate")
graph_rnn_element = Element(RNN_GRAPH_ID)

new_coordinate.element.value = 4
graph_coordinates: List[float] = [1,2,3,2,3,4,3,4,5]

# CREATING GRAPH
# --------------
figure, axis = plt.subplots()
axis.set_title("RNN Graph")
axis.set_xlabel("X Axis")
axis.set_ylabel("Y Axis")

rnn_axis, = axis.plot([], label="Predicted")
given_axis, = axis.plot(graph_coordinates, label="Given")

axis.legend()

display(figure, target=RNN_GRAPH_ID)

# RNN
# ---
def predict_rnn(normalized_data:List[float]) -> np.array:
    """A function to predict the next graph data using a rnn

    Args:
        normalized_data (List[float]): a list of the data that the user entered, normalized (between 0 to 1)

    Returns:
        np.array: the predicted data from the rnn
    """

    # INITALIZATION
    # -------------
    rnn = RNN_Layer(
        size=RNN_SIZE,
        activation_function=ActivationFunction(
            function=linear, function_derivative=linear_derivative
        ),
        seed=RNN_SEED,
    )

    # ACTION
    # ------
    # getting testing data
    inputs, outputs = get_training_in_out(data=normalized_data, input_size=RNN_SIZE,step_size=1)
    # training the RNN
    train_rnn(rnn=rnn,inputs=inputs,outputs=outputs,epoches=RNN_EPOCHES,learning_rate=.1,step_size=1)
    # predicting with the rnn
    return forward_propagation_rnn(rnn=rnn, inputs=np.array([normalized_data[-RNN_SIZE:]]), step_size=RNN_FORWARD_PREDICT,return_rnn_size=False)


def plot_graph():
    """A function to plot / update the matplotlib graph
    """

    # INITALIZATION
    # -------------
    global RNN_GRAPH_ID, RNN_SIZE, graph_coordinates, figure, axis, given_axis, rnn_axis
    
    # if we have enough data, then we want to predict with the RNN
    if len(graph_coordinates) > RNN_TRAIN_MIN_SIZE :
        # grabbing the min and max for normalization of data
        max_y = max(graph_coordinates)
        min_y = min(graph_coordinates)
        
        # adjusting the graph_coordinates to be within the range of 0 and 1
        normalized_data = list((np.array(graph_coordinates) - min_y)/(max_y - min_y))
        # getting the rnn prediction
        predictions = predict_rnn(normalized_data=normalized_data)
        # converting the normalized rnn prediction to be within the same range of the graph_coordinates
        rnn_output = np.append(graph_coordinates, predictions * (max_y - min_y) + min_y )

        # refreshing the axis to have the rnn prediction
        rnn_axis.set_data(range(len(rnn_output)),rnn_output)
    else:
        # clearing the rnn prediction
        rnn_axis.set_data([],[])

    # refreshing the given data
    given_axis.set_data(range(len(graph_coordinates)),graph_coordinates)

    # refreshing the matplotlib subplot
    axis.relim()
    axis.autoscale_view()
    figure.canvas.draw()

    # clearing the HTML element and re-populating it with the new graph
    graph_rnn_element.clear()
    display(figure, target=RNN_GRAPH_ID)


# HTML BUTTON FUNCTIONS
# ---------------------
def add_coordinate():
    """A function to add a y coordinate to the grpah
    """
    global graph_coordinates, new_coordinate

    # getting the elm value
    y_position = new_coordinate.element.value

    # if the value can be a float, then we add it to the graph_coordinates and plot
    # else, we continue on
    try:
        graph_coordinates.append(float(y_position))
        plot_graph()
    except Exception as error:
        pass

    # clearing the input field for another value to be added
    new_coordinate.clear()

    return


def reset_graph():
    """A function to reset the graph_coordinates
    """
    global graph_coordinates
    graph_coordinates = []
    plot_graph()
    return

# KEY FUNCTIONS
# -------------
def add_task_event(event):
    """A key bind for the enter, in the event that someone doesn't want to use the buttons I setup

    Args:
        event (_type_): pyscript key event
    """
    if event.key == "Enter":
        add_coordinate()

new_coordinate.element.onkeypress = add_task_event