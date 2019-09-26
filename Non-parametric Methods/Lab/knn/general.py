import pickle
import logging

# Define the logger to be used throughout the application run
LOG_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

LOG_FILE_HANDLER = logging.FileHandler("project_demo.log")
LOG_FILE_HANDLER.setLevel(logging.DEBUG)
LOG_FILE_HANDLER.setFormatter(LOG_FORMATTER)

STDOUT_HANDLER = logging.StreamHandler()
STDOUT_HANDLER.setLevel(logging.DEBUG)
STDOUT_HANDLER.setFormatter(LOG_FORMATTER)

log = logging.getLogger("project-demo")
log.setLevel(logging.DEBUG)
log.addHandler(LOG_FILE_HANDLER)
log.addHandler(STDOUT_HANDLER)


def pickle_object(object_to_pickle, object_filename):
    """Wrapper to pickle an object

    This simply pickles an object while logging the action.

    :param object_to_pickle: Any, the Python object to be pickled
    :param object_filename: String, the filename for the object to be pickled to
    """
    #log_info("Pickling the object at path {}".format(object_filename))
    final_filename = "results/" + object_filename
    with open(final_filename, "wb") as pickle_file:
        pickle.dump(object_to_pickle, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(object_filename):
    """Wrapper to load a pickled object

    This simply loads a pickled object while logging the action.

    :param object_filename: String, the filename for the object to be loaded
    :return: a Python object of an arbitrary type
    """
    #log_info("Load a pickle from {}".format(object_filename))
    final_filename = "results/" + object_filename
    return pickle.load(open(final_filename, "rb"))


def log_info(message):
    """Log a message at an information level

    :param message: String, the message to log
    """
    log.info(message)


