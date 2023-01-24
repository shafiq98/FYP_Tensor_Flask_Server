import logging

# Web Server Imports
from flask import Flask, jsonify, request

# SQL Imports

# Local Packages Import
from neural_network_functions import base as base_tf

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


# log.debug("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# log.debug(device_lib.list_local_devices())

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


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8080, debug=True)

    (X_train, y_train), (X_test, y_test) = base_tf.initialize_data()
    train_df, test_df = base_tf.generate_dataframe(X_train, y_train, X_test, y_test)
    (reconstructed_x_train, reconstructed_y_train), (
        reconstructed_x_test, reconstructed_y_test) = base_tf.generate_mnist_tuples(train_df, test_df)
    model = base_tf.create_model()
    model.fit(reconstructed_x_train, reconstructed_y_train,
              validation_data=(reconstructed_x_test, reconstructed_y_test), epochs=10,
              batch_size=200)
    # base_tf.test(X_train, model)
