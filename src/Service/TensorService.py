import json
import logging

# TensorFlow imports
import pandas as pd
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras import backend as K
import matplotlib.pyplot as plt

# the data, split between train and test sets
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import KFold
import tensorflow as tf
# from tensorflow.python.keras import Sequential
from tensorflow.python.client import device_lib

# from tensorflow.python.keras.optimizers import SGD

# SETUP
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

log.debug("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
log.debug(device_lib.list_local_devices())


# TensorFlow Functions
def initialize_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # # One hot Code
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # convert from integers to floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # normalize to range [0, 1]
    x_train = (x_train / 255.0)
    x_test = (x_test / 255.0)

    return (x_train, y_train), (x_test, y_test)


def create_model(is_high_performance: bool = False) -> Sequential:
    '''
    Creates neural network model
    :return: 5 Layer CNN Model
    :rtype: Sequential
    '''
    # Create model
    # Building CNN
    model = Sequential()
    # relu: rectified linear unit activation function
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    if is_high_performance:
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test(x_train, model):
    test_images = x_train[1:5]
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    for i, test_image in enumerate(test_images, start=1):
        org_image = test_image
        test_image = test_image.reshape(1, 28, 28, 1)
        # prediction = model.predict_classes(test_image, verbose=0)
        prediction = model(test_image)

        log.debug("Predicted digit: {}".format(prediction[0]))
        log.debug("Predicted digit: {}".format(np.argmax(prediction[0])))
        log.debug("Prediction Confidence: {}".format(prediction[0][np.argmax(prediction[0])]))
        plt.subplot(220 + i)
        plt.axis('off')
        plt.title("Predicted digit: {}".format(np.argmax(prediction[0])))
        plt.imshow(org_image, cmap=plt.get_cmap('gray'))

    plt.show()


def export(model: Sequential) -> None:
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    log.debug(json.dumps(model_json, indent=4))

    """
    TODO: Modify this method to export a TFLite file, and a C char array to a text file/c file
    Possible Resources
    1. Model Format Overview:   https://www.tensorflow.org/lite/models/convert
    2. Model Conversion:        https://www.tensorflow.org/lite/models/convert/convert_models
    3. C++ Model File:          https://www.tensorflow.org/lite/microcontrollers/build_convert
    4. xxd command explanation: https://www.tutorialspoint.com/unix_commands/xxd.htm
    5. Stack Overflow Solution: https://stackoverflow.com/questions/6624453/whats-the-correct-way-to-convert-bytes-to-a-hex-string-in-python-3
    """

    # serialize weights to HDF5
    model.save_weights("model.h5")
    log.info("Saved model to disk")
    return
