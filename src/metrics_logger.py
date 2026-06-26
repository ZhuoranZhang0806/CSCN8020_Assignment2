import logging
from pathlib import Path


def setup_logger(log_file="assignment2_execution.log"):
    """Create logger and save training information into a log file."""

    log_file = Path(log_file)

    # Create parent folder if it does not exist
    if log_file.parent != Path("."):
        log_file.parent.mkdir(exist_ok=True)

    logger = logging.getLogger("assignment2_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Each run has timestamped filename, so write mode is fine
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("New program run started")
    logger.info("=" * 80)

    return logger