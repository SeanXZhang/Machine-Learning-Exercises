import logging

# Define the logger to be used throughout the application run
LOG_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

LOG_FILE_HANDLER = logging.FileHandler("../logs/svm_lab.log")
LOG_FILE_HANDLER.setLevel(logging.DEBUG)
LOG_FILE_HANDLER.setFormatter(LOG_FORMATTER)

STDOUT_HANDLER = logging.StreamHandler()
STDOUT_HANDLER.setLevel(logging.DEBUG)
STDOUT_HANDLER.setFormatter(LOG_FORMATTER)

log = logging.getLogger("svm_lab")
log.setLevel(logging.DEBUG)
log.addHandler(LOG_FILE_HANDLER)
log.addHandler(STDOUT_HANDLER)


def log_info(message):
    """Log a message at an information level

    :param message: String, the message to log
    """
    log.info(message)
