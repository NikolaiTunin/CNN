import time
import h5py
import scipy
import imageio
import sklearn
import sklearn.datasets
from Deep_NN import *
from scipy import ndimage
from PIL import Image


def load_data(name):
    #name == catvnoncat/ signs
    train_dataset = h5py.File('datasets/train_' + name + '.h5', "r")
    test_dataset = h5py.File('datasets/test_' + name + '.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    train_dataset.close()
    test_dataset.close()

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def custom_image_prediction(im_name, im_label, parameters, classes, layers_dims, act_f, print_predictions, print_labels):
    fname = "images/" + im_name
    image = np.array(imageio.imread(fname))
    my_image = np.array(Image.fromarray(image).resize(size=(64, 64))).reshape((64 * 64 * 3, 1))
    my_image = my_image / 255.
    my_predicted_image = predict(my_image, np.array(im_label), parameters, layers_dims, act_f, print_predictions, print_labels)

    plt.imshow(image)
    plt.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
        plt.show()


def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data('catvnoncat')
    """
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    """
    # Example of a picture
    """
    index = 102
    plt.imshow(train_x_orig[index])
    print("y = " + str(train_y[0, index]) + ". It's a " + str(classes[train_y[0, index]]) + " picture.")
    plt.show()
    """
    # Explore your dataset
    """
    print("Number of training examples: " + str(train_x_orig.shape[0]))
    print("Number of testing examples: " + str(test_x_orig.shape[0]))
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))
    """
    # Reshape the training and test examples. The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    """
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    """
    # Convert training and test labels to one hot matrices. If labels are non-binary!
    #train_y = np.eye(max(train_y[0]) + 1)[train_y.reshape(-1)].T
    #test_y = np.eye(max(test_y[0]) + 1)[test_y.reshape(-1)].T
    """
    print("train_y's shape: " + str(train_y.shape))
    print("test_y's shape: " + str(test_y.shape))
    """
    act_f = "relu"

    layers_dims = (train_x.shape[0], 20, 7, 3, train_y.shape[0])
    keep_probs = (1, 1, 1, 1, 1)

    start = time.time()

    parameters = Deep_NN(train_x, train_y, layers_dims, keep_probs, 2500, 0.0075, act_f=act_f, full_batch=True, optimizer='gd', lambd=0, print_cost=True, print_plot=False)

    predictions = predict(test_x, test_y, parameters, layers_dims, act_f=act_f, print_predictions=False, print_labels=False)
    #Print mislabeled images
    #print_mislabeled_images(classes, test_x, test_y, predictions)
    #Custom image prediction
    custom_image_prediction('ncat3.jpeg', [0], parameters, classes, layers_dims, act_f=act_f, print_predictions=True, print_labels=True)
    #Gradient check
    """
    np.random.seed(0)
    
    x = np.random.randn(12288, 50)
    y = np.random.randn(1, 50)

    param = init_parameters_deep(layers_dims)
    AL, cache = Deep_forward_prop(x, param, act_f, keep_prob)
    gradients = Deep_backward_prop(param, AL, y, cache, act_f, lambd=0, keep_probs=keep_prob)
    difference = Deep_gradient_check(param, gradients, x, y, act_f, keep_prob)
    """

    end = time.time()
    print("Time elapsed (sec): ", end - start)

main()

#Cost after iteration 2500: 0.09897165877535206

#Time elapsed (sec):  21.528037548065186