"""
Centralized logging configuration for the tra_go project.
"""

import inspect
import logging
import sys
from collections.abc import Callable
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, TypeVar, cast

from core.config import settings


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

F = TypeVar("F", bound=Callable[..., Any])


def log_error(operation: str, error: Exception, **kwargs) -> None:
    """Log error with context and additional metadata."""
    caller_frame = inspect.stack()[1]
    caller_info = f"{Path(caller_frame.filename).name}:{caller_frame.lineno} in {caller_frame.function}"
    context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
    error_msg = f"[{caller_info}] Error in {operation}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    logger.error(error_msg, exc_info=True, stack_info=True)


def log_warning(message: str, **kwargs) -> None:
    """Log warning message with optional context."""
    caller_frame = inspect.stack()[1]
    caller_info = f"{Path(caller_frame.filename).name}:{caller_frame.lineno} in {caller_frame.function}"
    context = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
    warning_msg = f"[{caller_info}] {message}"
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


def log_exceptions(exit_on_exception: bool = True, exit_code: int = 1) -> Callable[[F], F]:
    """Decorator to log exceptions and optionally exit the application."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            try:
                return func(*args, **kwargs)
            except Exception as error:  # pylint: disable=broad-except
                log_error(operation=func.__name__, error=error)
                if exit_on_exception:
                    raise SystemExit(exit_code) from error
                raise

        return cast(F, wrapper)

    return decorator
