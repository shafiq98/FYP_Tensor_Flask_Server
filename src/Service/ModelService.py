import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(funcName)20s() - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def load_model(path: str, shouldLog:bool = False) -> list:
    # load from resources
    hex_stringified_array = None
    log.debug("Current Working Directory: {}".format(os.getcwd()))
    with open(file=path, mode='r') as readable_array:
        hex_stringified_array = readable_array.read()
        readable_array.close()

    hex_stringified_array = hex_stringified_array.replace(" ","")
    hex_stringified_array = hex_stringified_array.replace("\n","")
    hex_stringified_array = hex_stringified_array.split(",")

    hex_array = [hex(int(element, 16)) for element in hex_stringified_array]

    if (shouldLog):
        log.debug("Last 5 elements of hex_array: {}".format(hex_array[-5:]))
        log.debug("Length of hex_array: {}".format(len(hex_array)))
        log.debug("hex_array element type: {}".format(type(hex_array[0])))

    return hex_array

def model_export():
    with open(file='model.tflite', mode='rb') as tf_lite_readable, open('model.txt', 'w+') as c_array_writable:
        log.debug("TFLite Model: {}".format(tf_lite_readable))
        # log.debug(f.read())
        hex_array = [hex(i) for i in tf_lite_readable.read()]
        # hex_array = hex_array[-5:]
        log.debug("Last {} characters of hex_array = {}".format(5, hex_array[-5:]))
        log.debug("Length of hex_array = {}".format(len(hex_array)))

        hex_array_stringified = ", ".join(hex_array)

        # declaration_str1 = "unsigned char model_tflite[] = {"
        # c_array_writable.write(declaration_str1)

        c_array_writable.write(hex_array_stringified)

        # closing_str = "};\n"
        # c_array_writable.write(closing_str)
        #
        # model_length_str = "unsigned int model_tflite_len = {};".format(len(hex_array))
        # c_array_writable.write(model_length_str)

        tf_lite_readable.close()
        c_array_writable.close()