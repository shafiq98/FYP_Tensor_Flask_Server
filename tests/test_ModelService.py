from unittest import TestCase
from src.Service.ModelService import load_model

class Test(TestCase):
    def test_load_model1(self):
        load_model(r"..\resources\original_model.txt", shouldLog=True)

    def test_load_model2(self):
        load_model(r"..\resources\cnn_model.txt", shouldLog=True)
