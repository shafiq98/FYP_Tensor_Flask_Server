import logging

# Web Server Imports
from flask import Flask, jsonify, request

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
from tensorflow.python.keras import Sequential
from tensorflow.python.client import device_lib
# from tensorflow.python.keras.optimizers import SGD

# SQL Imports

'''
https://github.com/Joy2469/Deep-Learning-MNIST---Handwritten-Digit-Recognition/blob/master/digit_Recognition_CNN.py
'''

app = Flask(__name__)
GET = "GET"
POST = "POST"

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
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

    # # One hot Code
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # convert from integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize to range [0, 1]
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    return (X_train, y_train), (X_test, y_test)

def generate_dataframe(X_train, y_train, X_test, y_test, shouldLog:bool = False):
    '''
    Returns incoming nested tuples as 2 different dataframes
            Parameters:
                    X_train (nested tuple): TODO
                    X_test (nested tuple): TODO
                    y_train (nested tuple): TODO
                    y_test (nested tuple): TODO
            Returns:
                    (train_df, test_df) (Pandas Dataframes): 2 Dataframes, training & testing
            TODO:
                Modify this to use SQL DB instead, and return an SQL Connection object
    '''
    log.debug("Creating train Dataframe")
    train_df = pd.DataFrame(columns=["PixelArray", "Result"])
    for image in range(0, len(X_train)):
        dict_row = {"PixelArray": [X_train[image]], "Result": [y_train[image]]}
        df_row = pd.DataFrame(dict_row)
        train_df = pd.concat([train_df, df_row], ignore_index=True)

    log.debug("Successfully created train Dataframe")
    log.debug("Creating test Dataframe")

    test_df = pd.DataFrame(columns=["PixelArray", "Result"])
    for image in range(0, len(X_test)):
        dict_row = {"PixelArray": [X_test[image]], "Result": [y_test[image]]}
        df_row = pd.DataFrame(dict_row)
        test_df = pd.concat([test_df, df_row], ignore_index=True)

    log.debug("Successfully created test Dataframe")

    if shouldLog:
        log.debug("Number of training rows = {}".format(len(train_df)))
        log.debug("Number of nested arrays = {}".format(len(train_df.loc[0]["PixelArray"])))
        log.debug("Length of inner array = {}".format(len(train_df.loc[0]["PixelArray"][0])))

        log.debug("Number of testing rows = {}".format(len(test_df)))
        log.debug("Number of nested arrays = {}".format(len(test_df.loc[0]["PixelArray"])))
        log.debug("Length of inner array = {}".format(len(test_df.loc[0]["PixelArray"][0])))

    return (train_df, test_df)

def generate_mnist_tuples(train_df: pd.DataFrame, test_df: pd.DataFrame):
    '''
    Returns incoming 2 different dataframes as nested tuples
            Parameters:
                    :param test_df (Pandas Dataframe):
                    :param test_df (Pandas Dataframe):
            Returns:
                    (reconstructed_x_train, reconstructed_y_train), (reconstructed_x_test, reconstructed_y_test): 4 nested tuples, in the same format as initialize data
    '''
    reconstructed_x_train = train_df["PixelArray"].to_numpy().tolist()
    reconstructed_x_test = test_df["PixelArray"].to_numpy().tolist()

    reconstructed_y_train = train_df["Result"].to_numpy().tolist()
    reconstructed_y_test = test_df["Result"].to_numpy().tolist()
    # log.info(reconstructed_x_train)
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    log.debug("Length of reconstructed_x_train = {}".format(len(reconstructed_x_train)))
    log.debug("Length of reconstructed_x_train[0] = {}".format(len(reconstructed_x_train[0])))
    log.debug("Length of reconstructed_x_train[0][0] = {}".format(len(reconstructed_x_train[0][0])))
    # tuple_comparator(reconstructed_x_train, X_train)

    ## Return everything as np.array due to type expectation of TensorFlow training method
    # https://stackoverflow.com/questions/65474081/valueerror-data-cardinality-is-ambiguous-make-sure-all-arrays-contain-the-same
    return (np.array(reconstructed_x_train), np.array(reconstructed_y_train)), (np.array(reconstructed_x_test), np.array(reconstructed_y_test))

def create_model():
    '''
    Creates neural network model
            Parameters:
            Returns:
                    model {TensorFlow Model): 5 layer CNN Model
    '''
    # Create model
    # Building CNN
    model = Sequential()
    # relu: rectified linear unit activation function
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
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

def test(X_train, model):
    test_images = X_train[1:5]
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    for i, test_image in enumerate(test_images, start=1):
        org_image = test_image
        test_image = test_image.reshape(1, 28, 28, 1)
        prediction = model.predict_classes(test_image, verbose=0)

        print("Predicted digit: {}".format(prediction[0]))
        plt.subplot(220 + i)
        plt.axis('off')
        plt.title("Predicted digit: {}".format(prediction[0]))
        plt.imshow(org_image, cmap=plt.get_cmap('gray'))

    plt.show()


@app.route("/", methods=[POST])
def receive_training_data():
    divider = "================================================================"
    log.debug(request.data.decode("utf-8"))

    return jsonify(
        {
            "endpoint": "",
            "data": request.data.decode("utf-8"),
            # "form": request.form,
            # "json": request.get_json(),
        }
    )


if (__name__ == "__main__"):
    # app.run(host='0.0.0.0', port=8080, debug=True)

    (X_train, y_train), (X_test, y_test) = initialize_data()
    train_df, test_df = generate_dataframe(X_train, y_train, X_test, y_test)
    (reconstructed_x_train, reconstructed_y_train), (
    reconstructed_x_test, reconstructed_y_test) = generate_mnist_tuples(train_df, test_df)
    model = create_model()
    model.fit(reconstructed_x_train, reconstructed_y_train,
              validation_data=(reconstructed_x_test, reconstructed_y_test), epochs=10,
              batch_size=200)
    test(X_train, model)


