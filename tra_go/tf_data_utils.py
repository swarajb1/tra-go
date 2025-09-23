"""
Utility functions for configuring and optimizing tf.data pipelines.

This module provides functions to configure TensorFlow data pipelines for optimal
performance on different hardware platforms, with specific optimizations for Apple
Silicon Mac M4 systems. It handles GPU memory management, thread parallelism,
and provides performance monitoring and optimization tips.
"""

from core.logger import logger


def log_tf_data_performance_tips():
    """
    Log performance optimization tips for tf.data pipeline usage.

    This function outputs a comprehensive list of best practices and optimization
    techniques for using tf.data pipelines effectively. The tips cover various
    aspects of data pipeline performance including prefetching, caching, batching,
    and parallel processing.

    The logged tips include:
    - Using AUTOTUNE for automatic performance tuning
    - Data shuffling for improved model generalization
    - Memory caching strategies for small datasets
    - Prefetching to overlap data loading with training
    - Parallel processing with num_parallel_calls
    - Batch size optimization
    - Memory usage monitoring

    Note:
        These tips are logged at INFO level and can be viewed in the application
        logs. They serve as a reference for optimizing data loading performance.

    Example:
        >>> from tra_go.tf_data_utils import log_tf_data_performance_tips
        >>> log_tf_data_performance_tips()
        # Logs multiple performance tips for tf.data usage
    """

    tips = [
        "Use tf.data.AUTOTUNE for automatic performance tuning",
        "Enable shuffling for training data to improve model generalization",
        "Use .cache() for small datasets that fit in memory",
        "Use .prefetch() to overlap data loading with training",
        "Consider using .map() with num_parallel_calls=tf.data.AUTOTUNE for data transformations",
        "Batch your data before expensive operations like .cache() or .prefetch()",
        "Use appropriate batch sizes - larger batches can improve GPU utilization",
        "Monitor memory usage when using large shuffle buffers",
    ]

    logger.info("tf.data Performance Tips:")
    for i, tip in enumerate(tips, 1):
        logger.info(f"  {i}. {tip}")


def create_data_loading_summary(
    batch_size: int,
    train_size: int,
    test_size: int,
    shuffle_enabled: bool,
    cache_enabled: bool,
    prefetch_enabled: bool,
) -> dict:
    """
    Create a summary of data loading configuration.

    Returns:
        Dictionary with data loading configuration summary
    """

    summary = {
        "batch_size": batch_size,
        "train_samples": train_size,
        "test_samples": test_size,
        "train_steps_per_epoch": int((train_size + batch_size - 1) // batch_size),
        "test_steps_per_epoch": int((test_size + batch_size - 1) // batch_size),
        "shuffle_enabled": shuffle_enabled,
        "cache_enabled": cache_enabled,
        "prefetch_enabled": prefetch_enabled,
        "estimated_memory_per_batch_mb": (batch_size * 4 * 100) / (1024 * 1024),  # Rough estimate
    }

    return summary
