"""
Configuration helper for optimized data loader settings.

This module provides utilities to easily configure and control the use of
optimized tf.data pipelines vs traditional numpy-based data loading.
"""

from core.config import settings
from core.logger import logger


def set_optimized_data_loader(enabled: bool) -> None:
    """
    Set whether to use optimized tf.data pipelines for data loading.

    Args:
        enabled: True to enable optimized data loader, False to use traditional loader
    """
    # Note: This is a runtime change and won't persist across restarts
    # To make persistent changes, modify the .env file or config
    settings.USE_OPTIMIZED_DATA_LOADER = enabled

    status = "ENABLED" if enabled else "DISABLED"
    logger.info(f"Optimized data loader {status}")

    if enabled:
        logger.info("Benefits: Better performance, memory efficiency, prefetching, shuffling")
    else:
        logger.info("Using traditional numpy-based data loading")


def get_optimized_data_loader_status() -> bool:
    """
    Get the current status of optimized data loader setting.

    Returns:
        True if optimized loader is enabled, False otherwise
    """
    return settings.USE_OPTIMIZED_DATA_LOADER


def toggle_optimized_data_loader() -> bool:
    """
    Toggle the optimized data loader setting.

    Returns:
        The new status after toggling
    """
    new_status = not settings.USE_OPTIMIZED_DATA_LOADER
    set_optimized_data_loader(new_status)
    return new_status


def print_data_loader_info() -> None:
    """Print information about the current data loader configuration."""

    status = "ENABLED" if settings.USE_OPTIMIZED_DATA_LOADER else "DISABLED"

    print("=" * 60)
    print("DATA LOADER CONFIGURATION")
    print("=" * 60)
    print(f"Optimized Data Loader: {status}")
    print(f"Batch Size: {settings.BATCH_SIZE}")
    print(f"Test Size: {settings.TEST_SIZE}")

    if settings.USE_OPTIMIZED_DATA_LOADER:
        print("\nOptimized Loader Features:")
        print("  ✓ tf.data pipelines for efficient data loading")
        print("  ✓ Automatic batching and prefetching")
        print("  ✓ Data shuffling for better training")
        print("  ✓ Memory optimization and caching")
        print("  ✓ Performance monitoring and benchmarking")
        print("\nSupported Band Types:")
        print("  ✓ BAND_2_1 (Low/High bands)")
        print("  ✓ BAND_1_1 (Close price bands)")
    else:
        print("\nTraditional Loader Features:")
        print("  • NumPy array-based data loading")
        print("  • Manual batching in model.fit()")
        print("  • Compatible with all band types")
        print("  • More predictable memory usage")

    print("\nTo change the setting:")
    print("  set_optimized_data_loader(True)   # Enable optimized loader")
    print("  set_optimized_data_loader(False)  # Disable optimized loader")
    print("  toggle_optimized_data_loader()    # Toggle current setting")
    print("=" * 60)


if __name__ == "__main__":
    # Demo usage
    print_data_loader_info()
