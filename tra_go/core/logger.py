"""
Centralized logging configuration for the tra_go project.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from core.config import settings
from decorators.time import format_time


def setup_logger(
    name: str = "tra_go",
    log_file: Optional[str] = None,
    level: Optional[int] = None,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output with rotation.

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (defaults to config setting)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

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

    # Create formatter with more structured output
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(module)s:%(lineno)d | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (optional)
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Use rotating file handler to manage log file size
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"File logging enabled: {log_file}")
        except OSError as e:
            logger.warning(f"Failed to set up file logging: {e}")

    return logger


# Global logger instance
logger = setup_logger()


def log_model_training_start(ticker: str, model_type: str, datetime: str) -> None:
    """Log the start of model training."""
    logger.info(f"Starting model training | Ticker: {ticker} | Type: {model_type} | Time: {datetime}")


def log_model_training_complete(ticker: str, model_type: str, duration_seconds: float) -> None:
    """Log the completion of model training."""
    duration = format_time(duration_seconds)

    logger.info(f"Completed model training | Ticker: {ticker} | Type: {model_type} | Duration: {duration}")


def log_data_loading(ticker: str, interval: str, shape: tuple) -> None:
    """Log data loading information."""
    logger.info(f"Data loaded | Ticker: {ticker} | Interval: {interval} | Shape: {shape}")


def log_error(operation: str, error: Exception, **kwargs) -> None:
    """Log error with context and additional metadata."""
    context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
    error_msg = f"Error in {operation}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    logger.error(error_msg, exc_info=True, stack_info=True)


def log_warning(message: str, **kwargs) -> None:
    """Log warning message with optional context."""
    context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
    warning_msg = message
    if context:
        warning_msg += f" | Context: {context}"
    logger.warning(warning_msg)


def log_performance_metric(metric_name: str, value: float, **kwargs) -> None:
    """Log performance metrics in a structured way."""
    context = " | ".join([f"{k}: {v:.2f}" for k, v in kwargs.items()])
    metric_msg = f"Performance Metric | {metric_name}: {value:.2f}"
    if context:
        metric_msg += f" | {context}"
    logger.info(metric_msg)
