import logging

import matplotlib.pyplot as plt
# TensorFlow imports
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.python.client import device_lib

# SETUP
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

log.debug("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
log.debug(device_lib.list_local_devices())


# TensorFlow Service Functions
def model_service(x_train, y_train, x_test, y_test) -> Sequential:
    model: Sequential = create_model(is_high_performance=True)
    log.debug("Running model training")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    log.debug("Training complete!")
    export(model, c_file=True)

    return model


# TensorFlow Helper Functions
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


def export(model: Sequential, c_file: bool = False) -> None:
    log.debug("Serialize model to {} format in progress...".format("JSON"))
    model_json = model.to_json()
    with open(r"..\resources\model.json", "w") as json_file_writable:
        json_file_writable.write(model_json)
        json_file_writable.close()
    # log.debug(json.dumps(model_json, indent=4))
    log.debug("Serialize model to {} successful!".format("JSON"))

    """
    TODO: Modify this method to export a TFLite file, and a C char array to a text file/c file
    Possible Resources
    1. Model Format Overview:   https://www.tensorflow.org/lite/models/convert
    2. Model Conversion:        https://www.tensorflow.org/lite/models/convert/convert_models
    3. C++ Model File:          https://www.tensorflow.org/lite/microcontrollers/build_convert
    4. xxd command explanation: https://www.tutorialspoint.com/unix_commands/xxd.htm
    5. Stack Overflow Solution: https://stackoverflow.com/questions/6624453/whats-the-correct-way-to-convert-bytes-to-a-hex-string-in-python-3
    """

    log.debug("Serialize model to {} format in progress...".format("HDF5"))
    # serialize weights to HDF5
    model.save_weights(r"..\resources\model.h5")
    log.info("Saved model to disk")
    log.debug("Serialize model to {} successful!".format("HDF5"))

    log.debug("Serialize model to {} format in progress...".format("tflite"))
    # serialize model to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open(r"..\resources\model.tflite", 'wb') as tf_lite_writable:
        tf_lite_writable.write(tflite_model)
        tf_lite_writable.close()
    log.debug("Serialize model to {} successful!".format("tflite"))

    target_file_name = None
    if c_file:
        log.debug("Serialize model to {} format in progress...".format("cpp"))
        target_file_name = "model.cpp"
    else:
        log.debug("Serialize model to {} format in progress...".format("txt"))
        target_file_name = "model.txt"

    with open(file=r'..\resources\model.tflite', mode='rb') as tf_lite_readable, open(
            r'..\resources\{}'.format(target_file_name), 'w+') as c_array_writable:
        # log.debug("TFLite Model: {}".format(tf_lite_readable))
        # log.debug(f.read())
        hex_array = [hex(i) for i in tf_lite_readable.read()]
        # hex_array = hex_array[-5:]
        # log.debug("Last {} characters of hex_array = {}".format(5, hex_array[-5:]))
        # log.debug("Length of hex_array = {}".format(len(hex_array)))

        hex_array_stringified = ", ".join(hex_array)

        declaration_str1 = "unsigned char model_tflite[] = {"
        c_array_writable.write(declaration_str1)

        c_array_writable.write(hex_array_stringified)

        closing_str = "};\n"
        c_array_writable.write(closing_str)

        model_length_str = "unsigned int model_tflite_len = {};".format(len(hex_array))
        c_array_writable.write(model_length_str)

        tf_lite_readable.close()
        c_array_writable.close()

    if c_file:
        log.debug("Serialize model to {} successful!".format("cpp"))
    else:
        log.debug("Serialize model to {} successful!".format("txt"))

    return
