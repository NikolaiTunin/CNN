import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def input_size(X, Y):
    """
    Arguments:
        X -- input dataset of shape (number of examples, number of features)
        Y -- labels of shape (number of examples, number of features)

    Returns:
        n_x -- the number of features in every training example
        n_y -- the number of features in every label
        m -- the number of examples
    """
    n_x = X.shape[1]
    n_y = Y.shape[1]

    return n_x, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
        n_x -- the number of features in every training example
        n_x -- the number of nodes in hidden layer (hidden layer size)
        n_y -- the number of features in every label

    Returns:
        parameters -- python dictionary containing parameters:
            W1 -- weight matrix of shape (n_x, n_h)
            b1 -- bias vector of shape (m, n_h)
            W2 -- weight matrix of shape (n_h, n_y)
            b2 -- bias vector of shape (m, n_y)
    """
    W1 = np.random.randn(n_x, n_h) * 0.01
    b1 = 0
    W2 = np.random.randn(n_h, n_y) * 0.01
    b2 = 0

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Arguments:
        X --  input dataset of shape (number of examples, number of features)
        parameters -- python dictionary containing parameters:
            W1 -- weight matrix of shape (n_x, n_h)
            b1 -- bias vector of shape (1, n_h)
            W2 -- weight matrix of shape (n_h, n_y)
            b2 -- bias vector of shape (1, n_y)

    Returns:
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return cache


def compute_cost(cache, Y):
    """
    Arguments:
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        Y -- labels vector of shape (number of examples, 1)

    Returns:
        cost -- accuracy of our prediction
    """
    m = Y.shape[0]
    A2 = cache["A2"]

    loss = -(Y * np.log(A2) + (1 - Y) * np.log(1-A2))
    cost = np.sum(loss)/m

    return cost


def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def backward_propagation(parameters, cache, X, Y):
    """
    Arguments:
        parameters -- python dictionary containing parameters:
            W1 -- weight matrix of shape (n_x, n_h)
            b1 -- bias vector of shape (1, n_h)
            W2 -- weight matrix of shape (n_h, n_y)
            b2 -- bias vector of shape (1, n_y)
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        X --  input dataset of shape (number of examples, number of features)

    Returns:
        gradients -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[0]

    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.dot(dZ2, W2.T)*(1 - np.power(A1, 2))
    dW1 = np.dot(X.T, dZ1)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

    return gradients


def update_parameters(parameters, gradients, learning_rate):
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
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def two_layers_nn(X, Y, n_h=4, num_iterations=100, learning_rate=1, print_cost=False):
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
    np.random.seed(0)

    costs = []

    n_x, n_y = input_size(X, Y)

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        cache = forward_propagation(X, parameters)

        cost = compute_cost(cache, Y)

        gradients = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, gradients, learning_rate)

        if print_cost and i % (num_iterations / 10) == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    plot_cost(costs, learning_rate)

    return parameters


def predict(P, parameters):
    """
    Arguments:
        P --  input dataset of shape (number of examples, number of features) for prediction
        parameters -- parameters learnt by the model

    Returns:
        predictions -- vector of predictions for input dataset P (True, False)
    """
    A = forward_propagation(P, parameters)["A2"]

    predictions = (A >= 0.5)

    return predictions


X0 = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y0 = np.array([[0],
              [1],
              [0],
              [0]])

Y1 = np.array([[0, 1],
              [1, 0],
              [1, 0],
              [0, 1]])

param = two_layers_nn(X0, Y1, n_h=4, num_iterations=10, print_cost=True)
predi = predict(X0, param)
print(predi)