import logging
import pandas as pd
import numpy as np
import random

# SETUP
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# service method
def receive_training_data_service(train_df: pd.DataFrame, test_df: pd.DataFrame, incoming_data: str):
    log.debug("Convert incoming string to dataframe row (dictionary type)")
    row: dict = string_to_dataframe_row(incoming_data)

    log.debug("Update dataframe")
    train_df = train_df.append(row, ignore_index=True)

    return (train_df, test_df)


# setup methods
def generate_dataframes(x_train: tuple, y_train: tuple, x_test: tuple, y_test: tuple) -> (pd.DataFrame, pd.DataFrame):
    """
    Returns incoming nested tuples as 2 different dataframes
        :param y_test:
        :type y_test: tuple
        :param x_test:
        :type x_test: tuple
        :param y_train:
        :type y_train: tuple
        :param x_train:
        :type x_train: tuple
        :returns (train_df, test_df)
        :rtype: (pd.DataFrame, pd.DataFrame)
    TODO:
        Modify this to use SQL DB instead, and return an SQL Connection object

    """

    log.debug("Creating train Dataframe")
    train_df = generate_dataframe(x_train, y_train)
    log.debug("Successfully created train Dataframe")

    log.debug("Creating test Dataframe")
    test_df = generate_dataframe(x_test, y_test)
    log.debug("Successfully created test Dataframe")

    return (train_df, test_df)


def generate_dataframe(x_data: tuple, y_data: tuple, should_log: bool = False) -> pd.DataFrame:
    """
    Returns incoming nested tuples as 2 different dataframes
        :param should_log:
        :type should_log: bool
        :param x_data:
        :type x_data: nested tuple of pixels
        :param y_data:
        :type y_data: nested tuple of prediction
        :returns: df
        :type df: pd.DataFrame
    """
    # TODO: Find a way to refactor this for loop to be quicker
    # TODO: Modify this to use SQL DB instead, and return an SQL Connection object
    df = pd.DataFrame(columns=["PixelArray", "Result"])
    for image in range(0, len(x_data)):
        # for image in range(0, 20):
        dict_row = {"PixelArray": [x_data[image]], "Result": [y_data[image]]}
        df_row = pd.DataFrame(dict_row)
        df = pd.concat([df, df_row], ignore_index=True)

    if should_log:
        log.debug("Number of training rows = {}".format(len(df)))
        log.debug("Number of nested arrays = {}".format(len(df.loc[0]["PixelArray"])))
        log.debug("Length of inner array = {}".format(len(df.loc[0]["PixelArray"][0])))
        log.debug("y data sample = {}".format(len(df.loc[0]["Result"])))
        # log.debug("y data sample = {}".format(len(df.loc[0]["Result"][0])))

    return df


def generate_mnist_tuples(train_df: pd.DataFrame, test_df: pd.DataFrame) -> (np.array, np.array, np.array, np.array):
    """
    Returns incoming 2 different dataframes as nested tuples
        :param train_df: Training Dataframe that needs to be converted to 2 training tuples
        :type train_df: pd.DataFrame
        :param test_df: Testing Dataframe that needs to be converted to 2 testing tuples
        :type test_df: pd.DataFrame
        :returns (reconstructed_x_train, reconstructed_y_train), (reconstructed_x_test, reconstructed_y_test)
        :rtype: np.array
    """
    log.debug("Length of incoming Training DF: {}".format(len(train_df)))
    log.debug("Length of incoming Testing DF: {}".format(len(test_df)))

    reconstructed_x_train = train_df["PixelArray"].to_numpy().tolist()
    reconstructed_x_test = test_df["PixelArray"].to_numpy().tolist()

    reconstructed_y_train = train_df["Result"].to_numpy().tolist()
    reconstructed_y_test = test_df["Result"].to_numpy().tolist()
    # log.info(reconstructed_x_train)

    log.debug("Length of reconstructed_x_train = {}".format(len(reconstructed_x_train)))
    log.debug("Length of reconstructed_x_train[0] = {}".format(len(reconstructed_x_train[0])))
    log.debug("Length of reconstructed_x_train[0][0] = {}".format(len(reconstructed_x_train[0][0])))
    # tuple_comparator(reconstructed_x_train, x_train)

    # Return everything as np.array due to type expectation of TensorFlow training method
    # https://stackoverflow.com/questions/65474081/valueerror-data-cardinality-is-ambiguous-make-sure-all-arrays-contain-the-same
    reconstructed_x_train = np.array(reconstructed_x_train)
    reconstructed_y_train = np.array(reconstructed_y_train)
    reconstructed_x_test = np.array(reconstructed_x_test)
    reconstructed_y_test = np.array(reconstructed_y_test)
    return reconstructed_x_train, reconstructed_y_train, reconstructed_x_test, reconstructed_y_test


def string_to_dataframe_row(incoming_data: str) -> dict:
    # strip unnecessary characters
    incoming_data.replace('\n', '')
    incoming_data.replace(' ', '')

    # change string to float array
    str_array = incoming_data.split(",")
    float_array = list(map(float, str_array))

    # remove digit expectation (0-9) from training data (28x28 array)
    training_data = float_array[:-1]
    expectation = float_array[-1]

    # one-hot encode expectation
    one_hot_array = manual_one_hot_encode(expectation, 10)

    nested_training_data = np.reshape(training_data, (28, 28, 1)).astype('float32')

    dict_row = {"PixelArray": nested_training_data, "Result": one_hot_array}

    return dict_row


def manual_one_hot_encode(result: float, length: int) -> list:
    one_hot_array = [0.0] * length
    one_hot_array[int(result)] = 1.0
    return one_hot_array
