import logging

# Web Server Imports
from flask import Flask, jsonify, request, send_file

# # SQL Imports
# # Local Packages Import
from Service import ModelService as ms

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

@app.route("/train", methods=[POST])
def receive_training_data():
    divider = "================================================================"
    incoming_data_raw = request.data.decode("utf-8")
    log.debug("Decoded Data Type = {}".format(type(incoming_data_raw)))
    # log.debug(incoming_data_raw)

    return jsonify(
        {
            "message": "Placeholder Text for proper Response",
            "status": 200
        }
    )

@app.route("/model", methods=[GET])
def send_model():
    # response_array = ms.load_model(r"..\resources\original_model.txt")
    # return response_array
    return send_file(r"..\resources\model.cpp")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

    # (X_train, y_train), (X_test, y_test) = custom_tf.initialize_data()
    # train_df, test_df = dm.generate_dataframes(X_train, y_train, X_test, y_test)
    # reconstructed_x_train, reconstructed_y_train, reconstructed_x_test, reconstructed_y_test = dm.generate_mnist_tuples(train_df, test_df)
    # model = custom_tf.create_model(is_high_performance=True)
    # model.fit(reconstructed_x_train, reconstructed_y_train,
    #           validation_data=(reconstructed_x_test, reconstructed_y_test), epochs=10,
    #           batch_size=200)
    # custom_tf.test(X_train, model)

