import unittest
import logging
import os

import pandas as pd

from src.Service import DataManager as dm, TensorService as custom_tf

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class TestStringHandler(unittest.TestCase):
    data = None

    def setUp(self) -> None:
        # https://stackoverflow.com/questions/8369219/how-to-read-a-text-file-into-a-string-variable-and-strip-newlines
        log.debug("Current Directory: {}".format(os.getcwd()))
        with open(r'test_request.txt', 'r') as file:
            self.data = file.read().replace('\n', '')

    def test_basecase(self):
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


class DataFrameHandler(unittest.TestCase):
    trainingDf: pd.DataFrame = None
    testDf = None
    data = None
    X_train, y_train, X_test, y_test = None, None, None, None

    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = custom_tf.initialize_data()
        self.trainingDf, self.test_df = dm.generate_dataframes(self.X_train, self.y_train, self.X_test, self.y_test)
        log.debug("Current Directory: {}".format(os.getcwd()))
        with open(r'test_request.txt', 'r') as file:
            self.data = file.read().replace('\n', '')

    def test_basecase(self):
        dict_row = dm.string_to_dataframe_row(self.data)
        # df_row = pd.DataFrame(dict_row)
        # self.trainingDf = pd.concat([self.trainingDf, df_row], ignore_index=True)

        # reconstructed_x_train, reconstructed_y_train, reconstructed_x_test, reconstructed_y_test = dm.generate_mnist_tuples(self.trainingDf, self.test_df)
        # model = custom_tf.create_model()
        # model.fit(reconstructed_x_train, reconstructed_y_train,
        #           validation_data=(reconstructed_x_test, reconstructed_y_test), epochs=10,
        #           batch_size=200)

        log.debug("trainingDf.loc[0]: {}".format(self.trainingDf.loc[0]["PixelArray"]))
        log.debug("trainingDf.loc[0] length: {}".format(len(self.trainingDf.loc[0]["PixelArray"])))
        log.debug("trainingDf.loc[0][0]: {}".format(self.trainingDf.loc[0]["PixelArray"][0]))
        log.debug("trainingDf.loc[0][0] type: {}".format(type(self.trainingDf.loc[0]["PixelArray"][0])))

        log.debug("df_row: {}".format(dict_row.get("PixelArray")))
        log.debug("df_row[PixelArray] length: {}".format(len(dict_row.get("PixelArray"))))
        log.debug("df_row[PixelArray][0]: {}".format(dict_row.get("PixelArray")[0]))
        log.debug("df_row[PixelArray][0] type: {}".format(type(dict_row.get("PixelArray")[0])))

        self.trainingDf = self.trainingDf.append(dict_row, ignore_index=True)

        reconstructed_x_train, reconstructed_y_train, reconstructed_x_test, reconstructed_y_test = dm.generate_mnist_tuples(
            self.trainingDf, self.test_df)
        model = custom_tf.create_model()
        model.fit(reconstructed_x_train, reconstructed_y_train,
                  validation_data=(reconstructed_x_test, reconstructed_y_test), epochs=10,
                  batch_size=200)
        custom_tf.test(self.X_train, model)

        custom_tf.export(model)
