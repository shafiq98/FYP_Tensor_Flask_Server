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
