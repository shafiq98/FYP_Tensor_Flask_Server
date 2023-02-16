import logging

# Local Packages Import
from Service import DataService as DS
from Service import TensorService as TS
from Utilities.Constants import LEARNING_TRIGGER

# Web Server Imports
import requests
from flask import Flask, jsonify, request
'''
CNN from: https://github.com/Joy2469/Deep-Learning-MNIST---Handwritten-Digit-Recognition/blob/master/digit_Recognition_CNN.py
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
(x_train, y_train), (x_test, y_test) = TS.initialize_data()
(train_df, test_df) = DS.generate_dataframes(x_train, y_train, x_test, y_test)
totalLength = len(train_df) + len(test_df)
# train_df = pd.DataFrame()
# test_df = pd.DataFrame()

@app.route("/train", methods=[POST])
def receive_training_data():
    log.debug("================================================================")
    data = request.data.decode('utf-8')
    # log.debug("Decoded Data Type = {}".format(type(incoming_data_raw)))
    # log.debug("Decoded Data Length = {}".format(len(incoming_data_raw)))
    # log.debug(incoming_data_raw)

    global train_df, test_df, totalLength

    (train_df, test_df) = DS.receive_training_data_service(train_df, test_df, data)
    log.debug("Received Data Successfully")

    # trigger training based on condition like loop counter or simply schedule daily training
    updatedLength = len(train_df) + len(test_df)
    log.debug("Updated Length = {}".format(updatedLength))
    if (updatedLength - totalLength >= LEARNING_TRIGGER) :
        log.debug("We have collected an additional {} samples, starting training now".format(LEARNING_TRIGGER))
        x_train, y_train, x_test, y_test = DS.generate_mnist_tuples(train_df, test_df)
        TS.model_service(x_train, y_train, x_test, y_test)
        send_model()
        totalLength = updatedLength
    log.debug("================================================================")

    return jsonify(
        {
            "message": "Data received successfully!",
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

