import logging
import os

def get_logger(name: str):
    # Create logs directory if not exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # create a logger or get an existing one
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )

    # Console handler
    # handler is the object that sends the log messages to their final destination
    # streamhandler sends the log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
