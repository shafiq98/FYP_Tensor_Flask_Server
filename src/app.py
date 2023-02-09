import logging

# Web Server Imports
import os

from flask import Flask, jsonify, request
import requests

# # SQL Imports
# # Local Packages Import
from Service import ModelService as ms
from Service import DataService as ds
from Service import TensorService as ts

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

# initialize dataframes for training purpose
(x_train, y_train), (x_test, y_test) = ts.initialize_data()
(train_df, test_df) = ds.generate_dataframes(x_train, y_train, x_test, y_test)

@app.route("/train", methods=[POST])
def receive_training_data():
    log.debug("================================================================")
    data = request.data.decode('utf-8')
    # log.debug("Decoded Data Type = {}".format(type(incoming_data_raw)))
    # log.debug("Decoded Data Length = {}".format(len(incoming_data_raw)))
    # log.debug(incoming_data_raw)

    ds.receive_training_data_service(train_df, test_df, data)
    log.debug("================================================================")

    # trigger training based on condition like loop counter or simply schedule daily training

    return jsonify(
        {
            "message": "Placeholder Text for proper Response",
            "status": 200
        }
    )

# temporary method to send model to zephyr server
def send_model():
    test_file = open(r"..\resources\model.cpp", "rb")
    BASE_URL = "http://localhost:8081/retrieve_model"
    test_response = requests.post(BASE_URL, files={"model": test_file})
    test_file.close()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
    # send_model()

