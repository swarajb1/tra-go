"""
Centralized logging configuration for the tra_go project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from core.config import settings


def setup_logger(
    name: str = "tra_go",
    log_file: Optional[str] = None,
    level: Optional[int] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (defaults to config setting)

    Returns:
        Configured logger instance
    """
    if level is None:
        level = settings.LOGGING_LEVEL

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()


def log_model_training_start(ticker: str, model_type: str, datetime: str) -> None:
    """Log the start of model training."""
    logger.info(f"Starting model training for {ticker} - {model_type} at {datetime}")


def log_model_training_complete(ticker: str, model_type: str, duration: float) -> None:
    """Log the completion of model training."""
    logger.info(f"Completed model training for {ticker} - {model_type} in {duration:.2f} seconds")


def log_data_loading(ticker: str, interval: str, shape: tuple) -> None:
    """Log data loading information."""
    logger.info(f"Loaded data for {ticker} ({interval}) with shape: {shape}")


def log_error(operation: str, error: Exception) -> None:
    """Log error with context."""
    logger.error(f"Error in {operation}: {str(error)}", exc_info=True)


def log_warning(message: str) -> None:
    """Log warning message."""
    logger.warning(message)
