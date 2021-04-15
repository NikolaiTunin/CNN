import numpy as np


def dictionary_to_vector(dict):
    #Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    count = 0
    for key in dict.keys():
        # flatten parameter
        new_vector = np.reshape(dict[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count += 1

    return theta


def vector_to_dictionary(theta, example_parameters):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    L = len(example_parameters) // 2
    parameters = {}
    right_index = 0

    for l in range(1, L + 1):
        left_index = right_index

        W_exmpl = example_parameters['W' + str(l)]
        b_exmpl = example_parameters['b' + str(l)]
        shape = W_exmpl.shape

        if l == 1:
            left_index += W_exmpl.shape[0] * W_exmpl.shape[1]

            parameters['W' + str(l)] = theta[:left_index].reshape(shape)
        else:
            right_index = left_index + W_exmpl.shape[0] * W_exmpl.shape[1]

            parameters['W' + str(l)] = theta[left_index:right_index].reshape(shape)

            left_index = right_index

        shape = b_exmpl.shape

        right_index = left_index + b_exmpl.shape[0] * b_exmpl.shape[1]

        parameters['b' + str(l)] = theta[left_index:right_index].reshape(shape)

    return parameters


def reverse_dict(dict):
    L = len(dict) // 2
    new_dict = {}

    for l in range(1, L + 1):
        key = 'dW' + str(l)
        new_dict[key] = dict[key]

        key = 'db' + str(l)
        new_dict[key] = dict[key]

    return new_dict
