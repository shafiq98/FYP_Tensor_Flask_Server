import unittest
import logging
import os

import pandas as pd

from src.Service import DataService as dm, TensorService as custom_tf

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class DataFrameHandler(unittest.TestCase):
    trainingDf: pd.DataFrame = None
    testDf = None
    data = None
    X_train, y_train, X_test, y_test = None, None, None, None
    model = None

    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = custom_tf.initialize_data()
        self.trainingDf, self.test_df = dm.generate_dataframes(self.X_train, self.y_train, self.X_test, self.y_test)
        log.debug("Current Directory: {}".format(os.getcwd()))
        with open(r'test_binary_request.txt', 'r') as file:
            # self.data = file.read().replace('\n', '')
            self.data = file.read()

    def test_createModel(self):
        dict_row = dm.string_to_dataframe_row(self.data)

        self.trainingDf = self.trainingDf.append(dict_row, ignore_index=True)

        reconstructed_x_train, reconstructed_y_train, reconstructed_x_test, reconstructed_y_test = dm.generate_mnist_tuples(
            self.trainingDf, self.test_df)
        self.model = custom_tf.create_model()
        self.model.fit(reconstructed_x_train, reconstructed_y_train,
                       validation_data=(reconstructed_x_test, reconstructed_y_test), epochs=10,
                       batch_size=200)

    def test_textFileCreation(self):
        custom_tf.export(self.model)

    def test_cFileCreation(self):
        custom_tf.export(self.model, c_file=True)
