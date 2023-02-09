import os
from unittest import TestCase
import pandas as pd
import logging

from src.Service import DataService as dm, TensorService as custom_tf

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class Test(TestCase):
    trainDf: pd.DataFrame = None
    testDf: pd.DataFrame = None
    data = None
    x_train, y_train, x_test, y_test = None, None, None, None
    model = None

    def setUp(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = custom_tf.initialize_data()
        (self.trainDf, self.testDf) = dm.generate_dataframes(self.x_train, self.y_train, self.x_test, self.y_test)
        log.debug("Current Directory: {}".format(os.getcwd()))
        with open(r'binary_request.txt', 'r') as file:
            # self.data = file.read().replace('\n', '')
            self.data = file.read()

        log.debug("Length of Training DF: {}".format(len(self.trainDf)))
        log.debug("Length of Testing DF: {}".format(len(self.testDf)))

    def test_createDictionaryRow(self):
        log.debug("Starting base test")
        log.debug("Last 5 Characters of binary_request.txt = {}".format(self.data[-5:]))

        dict_row = dm.string_to_dataframe_row(self.data)

        x_data = dict_row.get("PixelArray")
        y_data = dict_row.get("Result")

        # x_data assertions
        self.assertTrue(type(x_data), list)
        self.assertEqual(len(x_data), 28)
        self.assertEqual(len(x_data[0]), 28)

        # y_data assertions
        self.assertTrue(type(y_data), list)
        self.assertTrue(type(y_data[0]), float)
        self.assertEqual(y_data[6], 1.0)

    def test_tuple_generation(self):
        self.trainDf, self.testDf = dm.receive_training_data_service(self.trainDf, self.testDf, self.data)
        reconstructed_x_train, reconstructed_y_train, reconstructed_x_test, reconstructed_y_test = dm.generate_mnist_tuples(
            self.trainDf, self.testDf)

        # verify that new tuples are longer than original tuples
        self.assertTrue(len(reconstructed_x_train) > len(self.x_train) or len(reconstructed_x_test) > len(self.x_test))
        self.assertTrue(len(reconstructed_y_train) > len(self.y_train) or len(reconstructed_y_test) > len(self.y_test))

        log.debug("len(reconstructed_x_train) = {}\tlen(self.x_train) = {}".format(len(reconstructed_x_train),
                                                                                   len(self.x_train)))
        log.debug("len(reconstructed_y_train) = {}\tlen(self.y_train) = {}".format(len(reconstructed_y_train),
                                                                                   len(self.y_train)))
