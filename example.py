#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from mlpy import *

# XOR Port example.
input_data = numpy.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
])

desired_output = numpy.array([
    [-1],
    [1],
    [1],
    [-1],
])

# Run the training algorithm.
(hidden_weights, output_weights) = trainning_algorithm(
    neurons_hidden_layer = 3,
    break_error = 0.0001,
    break_iterations = 10000,
    eta = 0.1,
    alpha = 0.3,
    input_data = input_data,
    desired_output = desired_output
)

test_data = numpy.matrix([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

result = validating_algorithm(hidden_weights, output_weights, test_data)
print numpy.transpose(result)
