import numpy as np
import pickle
import cv2

def load_data():

    """
    Load train, validation and test set.
    """

    training_file = 'data/train.p'
    validation_file= 'data/valid.p'
    testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train.shape[1:4]
    n_classes = len(set(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_signlist():
    """
    Read the csv file and get a list of all the sign names.
    """
    sign_list = []
    with open('signnames.csv') as csvfile:
        for row in csvfile:
            name = row.strip().split(',')
            if (name[1] != "SignName"):
                sign_list.append(name[1])
    return sign_list


def grayscale(input_array):
    """
    Applies the Grayscale transform to an array of colored images.

    :param input_array: an array of colored images
    :return: an array of grayscaled images

    """
    res = []
    for i in range(input_array.shape[0]):
        image = input_array[i]
        gray_image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.float32)
        res.append(np.expand_dims(gray_image, axis=2))
    return np.array(res)


def normalize(image_data):
    """
    Normaize the greyscale image before training our neural network.

    :param image_data: an array of greyscaled images
    :return: an array of normalized greyscaled images
    """
    a = 0.1
    b = 0.9
    X_min = 0
    X_max = 255
    return a + (image_data - X_min) * (b-a) / (X_max - X_min)
