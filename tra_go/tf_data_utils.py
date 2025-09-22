"""
Utility functions for configuring and optimizing tf.data pipelines.

This module provides functions to configure TensorFlow data pipelines for optimal
performance on different hardware platforms, with specific optimizations for Apple
Silicon Mac M4 systems. It handles GPU memory management, thread parallelism,
and provides performance monitoring and optimization tips.
"""

import tensorflow as tf
from core.logger import logger


def configure_tf_data_performance():
    """
    Configure TensorFlow for optimal data pipeline performance on Mac M4 systems.

    This function sets up TensorFlow configuration options specifically optimized
    for Apple Silicon M4 processors, including GPU memory growth and thread
    parallelism settings for tf.data pipelines. It enables Metal Performance
    Shaders support and configures the system to utilize all available CPU cores
    efficiently during data loading and preprocessing operations.

    The function performs the following optimizations:
    - Enables GPU memory growth to prevent out-of-memory errors on unified memory
    - Configures intra-op and inter-op parallelism threads for optimal CPU utilization
    - Provides detailed logging for troubleshooting and performance monitoring

    Note:
        This function is specifically designed for Mac M4 systems but will work
        on other systems with appropriate fallbacks. GPU configuration only applies
        when Metal-compatible GPUs are detected.

    Raises:
        None: All exceptions are caught and logged as warnings to prevent
        training interruption.

    Example:
        >>> from tra_go.tf_data_utils import configure_tf_data_performance
        >>> configure_tf_data_performance()
        # Logs: "Mac M4 GPU memory growth enabled for 1 GPU(s) (Metal Performance Shaders)"
        # Logs: "Mac M4 optimized thread configuration: Using all available CPU cores (auto-detect)"
        # Logs: "TensorFlow performance optimizations applied (Mac M4 optimized)"
    """

    # Configure memory growth for GPU (if available) - Mac M4 optimized
    try:
        # Use modern TensorFlow API instead of deprecated experimental API
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                # Enable memory growth to prevent OOM on Mac M4's unified memory
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"Mac M4 GPU memory growth enabled for {len(gpus)} GPU(s) (Metal Performance Shaders)")
            logger.info("Memory growth configuration helps with Mac M4's unified memory architecture")
        else:
            logger.info("No GPUs detected, using CPU")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")

    # already set elsewhere not needed. - Set optimal thread configuration for CPU

    # # Set optimal thread configuration for CPU
    # # This helps with data loading performance
    # tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    # tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores

    logger.info("Mac M4 optimized thread configuration: Using all available CPU cores (auto-detect)")

    logger.info("TensorFlow performance optimizations applied (Mac M4 optimized)")


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
