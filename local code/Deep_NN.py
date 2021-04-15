import numpy as np
import math
import matplotlib.pyplot as plt
from utilities import *


def plot_cost(costs, learn_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learn_rate))
    plt.show()


def act_func(Z, act_f, deriv=False):
    if deriv:
        if act_f == "sigmoid":
            return act_func(Z, "sigmoid") * (1 - act_func(Z, "sigmoid"))
        elif act_f == "tanh":
            return 1 - np.power(act_func(Z, "tanh"), 2)
        elif act_f == "relu":
            T = Z
            T[T > 0] = 1
            return T
        elif act_f == "lrelu":
            T = Z
            T[T > 0] = 1
            T[T < 0] = 0.01
            return T
    else:
        if act_f == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif act_f == 'softmax':
            return np.exp(Z - np.max(Z))/np.sum(np.exp(Z - np.max(Z)), axis= 0)
        elif act_f == "tanh":
            return np.tanh(Z)
        elif act_f == "relu":
            T = Z
            T[T < 0] = 0
            return T
        elif act_f == "lrelu":
            return np.maximum(0.01 * Z, Z)


def Deep_gradient_check(parameters, gradients, X, Y, act_f, keep_probs, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # Set-up variables
    parameters_values = dictionary_to_vector(parameters)
    grad = dictionary_to_vector(reverse_dict(gradients))
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        thetaplus_dict = vector_to_dictionary(thetaplus, parameters)
        AL, _ = Deep_forward_prop(X, thetaplus_dict, act_f, keep_probs)
        J_plus[i] = compute_cost(AL, Y, thetaplus_dict, lambd=0)

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] -= epsilon
        thetaminus_dict = vector_to_dictionary(thetaminus, parameters)
        AL, _ = Deep_forward_prop(X, thetaminus_dict, act_f, keep_probs)
        J_minus[i] = compute_cost(AL, Y, thetaminus_dict, lambd=0)

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    difference = np.linalg.norm(grad - gradapprox) / (np.linalg.norm(grad) + np.linalg.norm(gradapprox))

    if difference > 2e-7:
        print("There is a mistake in the backward propagation! difference = " + str(difference))
    else:
        print("Your backward propagation works perfectly fine! difference = " + str(difference))

    return difference


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return v


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return v, s


def random_mini_batches(X, Y, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def init_parameters_deep(layers_dims):
    """
    Argument:
        layer_dimensions -- python array (list) containing the dimensions of each layer in our network

    Returns:
        parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
            Wl -- weight matrix of shape (size of previous layer, size of layer)
            bl -- bias vector of shape (1, size of layer)
    """
    parameters = {}
    L = len(layers_dims)  # number of layers in the nn, including 0 layer

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def Deep_forward_prop(X, parameters, act_f, keep_probs):
    """
    Arguments:
        X --  input dataset of shape (number of examples, number of features)
        parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
            Wl -- weight matrix of shape (size of previous layer, size of layer)
            bl -- bias vector of shape (1, size of layer)

    Returns:
        cache -- a dictionary containing "Z1", "A1", ... , ZL, AL (there are L-1 of them, indexed from 0 to L-1)
    """
    cache = {}
    cache['A' + str(0)] = X
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L + 1):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        Z = np.dot(W, A_prev) + b

        if l == L:
            A = act_func(Z, "sigmoid")
        else:
            A = act_func(Z, act_f)

        if keep_probs[l] == 1:
            D = A
        else:
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D <= keep_probs[l])
            A *= D
            A /= keep_probs[l]

        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
        cache['D' + str(l)] = D

    return A, cache


def compute_cost(AL, Y, parameters, lambd):
    """
    Arguments:
        AL -- the last activation
        Y -- labels vector of shape (number of examples, 1)

    Returns:
        cost -- accuracy of our prediction
    """
    m = Y.shape[1]
    L = len(parameters) // 2
    summa = 0

    cross_entropy_cost = -np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))/m

    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        summa += np.sum(np.square(W))

    L2_regularization_cost = (lambd / (2 * m)) * summa

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


def Deep_backward_prop(parameters, AL, Y, cache, act_f, lambd, keep_probs):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    gradients = {}
    L = len(cache) // 3  # the number of layers
    m = Y.shape[1]

    dZ_next = AL - Y
    gradients["dW" + str(L)] = np.dot(dZ_next, cache["A" + str(L - 1)].T)/m + lambd * parameters['W' + str(L)] / m
    gradients["db" + str(L)] = np.sum(dZ_next, axis=1, keepdims=True)/m

    # Loop from l=L-1 to l=1
    for l in reversed(range(1, L)):
        W_next = parameters['W' + str(l + 1)]
        W = parameters['W' + str(l)]
        Z = cache['Z' + str(l)]
        A_prev = cache['A' + str(l - 1)]
        D = cache['D' + str(l)]

        dA = np.dot(W_next.T, dZ_next)

        if keep_probs[l] != 1:
            dA *= D
            dA /= keep_probs[l]

        dZ = dA * act_func(Z, act_f, True)

        dW = np.dot(dZ, A_prev.T)/m + lambd * W / m
        db = np.sum(dZ, axis=1, keepdims=True)/m

        gradients["dW" + str(l)] = dW
        gradients["db" + str(l)] = db

        dZ_next = dZ

    return gradients


def Deep_update_parameters(parameters, gradients, learn_rate):
    """
    Arguments:
        parameters -- python dictionary containing parameters:
            W1 -- weight matrix of shape (n_x, n_h)
            b1 -- bias vector of shape (1, n_h)
            W2 -- weight matrix of shape (n_h, n_y)
            b2 -- bias vector of shape (1, n_y)
        gradients -- python dictionary containing your gradients with respect to different parameters
        learning_rate -- the length of every step in gradient descent

    Returns:
        parameters -- python dictionary containing parameters:
            W1 -- weight matrix of shape (n_x, n_h)
            b1 -- bias vector of shape (1, n_h)
            W2 -- weight matrix of shape (n_h, n_y)
            b2 -- bias vector of shape (1, n_y)
    """
    L = len(gradients) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learn_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] -= learn_rate * gradients['db' + str(l)]

    return parameters


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(1, L + 1):
        # compute velocities
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads['db' + str(l)]
        # update parameters
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]

    return parameters, v


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads['db' + str(l)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - math.pow(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - math.pow(beta1, t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.power(grads['dW' + str(l)], 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.power(grads['db' + str(l)], 2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - math.pow(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - math.pow(beta2, t))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (
                    np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (
                    np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s


def Deep_NN(X, Y, layers_dims, keep_probs, num_epochs=1000, learn_rate=1, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,  epsilon=1e-8,  act_f="relu", full_batch=True, optimizer='gd', lambd=0, print_cost=False, print_plot=False):
    """
    Arguments:
        X -- input dataset of shape (number of examples, number of features)
        Y -- labels of shape (number of examples, number of features)
        n_h -- size of the hidden layer
        num_iterations -- number of iterations in gradient descent loop
        learning_rate -- the length of every step in gradient descent
        print_cost -- if True, print the cost every 100 iterations

    Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
    np.random.seed(1)

    t = 0  # initializing the counter required for Adam update

    costs = []

    parameters = init_parameters_deep(layers_dims)

    # Initialize the optimizer
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    if full_batch:
        for i in range(1, num_epochs + 1):
            AL, cache = Deep_forward_prop(X, parameters, act_f, keep_probs)
            gradients = Deep_backward_prop(parameters, AL, Y, cache, act_f, lambd, keep_probs)

            if optimizer == 'gd':
                parameters = Deep_update_parameters(parameters, gradients, learn_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, gradients, v, beta, learn_rate)
            elif optimizer == "adam":
                t += 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, gradients, v, s,
                                                               t, learn_rate, beta1, beta2, epsilon)
            if print_cost and (i % (num_epochs / 10) == 0 or i == 1):
                cost = compute_cost(AL, Y, parameters, lambd)
                print("Cost after epoch " + str(i) + ": " + str(cost))
                costs.append(cost)
    else:
        seed = 10
        # Optimization loop
        for i in range(1, num_epochs + 1):
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed += 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # Forward propagation
                AL, cache = Deep_forward_prop(minibatch_X, parameters, act_f, keep_probs)
                # Compute cost
                cost = compute_cost(AL, minibatch_Y, parameters, lambd)
                # Backward propagation
                grads = Deep_backward_prop(parameters, AL, minibatch_Y, cache, act_f, lambd, keep_probs)
                # Update parameters
                if optimizer == "gd":
                    parameters = Deep_update_parameters(parameters, grads, learn_rate)
                elif optimizer == "momentum":
                    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learn_rate)
                elif optimizer == "adam":
                    t += 1  # Adam counter
                    parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                                   t, learn_rate, beta1, beta2, epsilon)

            # Print the cost every 1000 epoch
            if print_cost and (i % (num_epochs / 10) == 0 or i == 1):
                print("Cost after epoch " + str(i) + ": " + str(cost))
            if print_cost and (i % (num_epochs / 100) == 0 or i == 1):
                costs.append(cost)

    # plot the cost
    if print_plot:
        plot_cost(costs, learn_rate)

    return parameters


def predict(X, Y, parameters, layers_dims, act_f="relu", print_predictions=False, print_labels=False):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]

    # Forward propagation
    P, _ = Deep_forward_prop(X, parameters, act_f, keep_probs=np.ones_like(layers_dims))

    # convert probas to 0/1 predictions
    P[P >= 0.5] = 1
    P[P < 0.5] = 0

    # print results
    print("Accuracy: " + str(round(np.sum((P == Y) / (m * Y.shape[0])), 3)))

    if print_predictions:
        print('Predictions:\n', P)
    if print_labels:
        print('Labels:\n', Y)

    return P

