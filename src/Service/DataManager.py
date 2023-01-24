import logging
import pandas as pd
import numpy as np

# SETUP
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


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
    TODO: Modify this to use SQL DB instead, and return an SQL Connection object
    """
    # TODO: Find a way to refactor this for loop to be quicker
    # TODO: Modify this to use SQL DB instead, and return an SQL Connection object
    df = pd.DataFrame(columns=["PixelArray", "Result"])
    for image in range(0, len(x_data)):
        dict_row = {"PixelArray": [x_data[image]], "Result": [y_data[image]]}
        df_row = pd.DataFrame(dict_row)
        df = pd.concat([df, df_row], ignore_index=True)

    if should_log:
        log.debug("Number of training rows = {}".format(len(df)))
        log.debug("Number of nested arrays = {}".format(len(df.loc[0]["PixelArray"])))
        log.debug("Length of inner array = {}".format(len(df.loc[0]["PixelArray"][0])))

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
    reconstructed_x_train = train_df["PixelArray"].to_numpy().tolist()
    reconstructed_x_test = test_df["PixelArray"].to_numpy().tolist()

    reconstructed_y_train = train_df["Result"].to_numpy().tolist()
    reconstructed_y_test = test_df["Result"].to_numpy().tolist()
    # log.info(reconstructed_x_train)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    log.debug("Length of reconstructed_x_train = {}".format(len(reconstructed_x_train)))
    log.debug("Length of reconstructed_x_train[0] = {}".format(len(reconstructed_x_train[0])))
    log.debug("Length of reconstructed_x_train[0][0] = {}".format(len(reconstructed_x_train[0][0])))
    # tuple_comparator(reconstructed_x_train, x_train)

    # Return everything as np.array due to type expectation of TensorFlow training method
    # https://stackoverflow.com/questions/65474081/valueerror-data-cardinality-is-ambiguous-make-sure-all-arrays-contain-the-same
    return np.array(reconstructed_x_train), np.array(reconstructed_y_train), np.array(reconstructed_x_test), np.array(reconstructed_y_test)