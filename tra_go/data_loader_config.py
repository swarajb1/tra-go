"""
Configuration helper for optimized data loader settings.

This module provides utilities to easily configure and control the use of
optimized tf.data pipelines vs traditional numpy-based data loading.
"""

from core.config import settings


def get_optimized_data_loader_status() -> bool:
    """
    Get the current status of optimized data loader setting.

    Returns:
        True if optimized loader is enabled, False otherwise
    """
    return settings.USE_OPTIMIZED_DATA_LOADER


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
