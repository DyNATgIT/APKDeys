import logging
import sys
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO", log_path: str = "./logs"):
    """Configure structured JSON logging"""

    # Create logs directory
    Path(log_path).mkdir(parents=True, exist_ok=True)

    # Create formatters
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (JSON format)
    log_file = (
        Path(log_path) / f"reconciliation_{datetime.now().strftime('%Y%m%d')}.json"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    return root_logger
