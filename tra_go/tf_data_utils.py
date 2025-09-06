"""
Utility functions for configuring and optimizing tf.data pipelines.
"""

import tensorflow as tf
from core.logger import logger


def configure_tf_data_performance():
    """
    Configure TensorFlow for optimal data pipeline performance.

    This function sets up TensorFlow options that can improve data loading
    and training performance when using tf.data pipelines.
    """

    # Configure memory growth for GPU (if available)
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            logger.info("No GPUs detected, using CPU")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")

    # already set elsewhere not needed. - Set optimal thread configuration for CPU

    # # Set optimal thread configuration for CPU
    # # This helps with data loading performance
    # tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    # tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores

    logger.info("TensorFlow performance optimizations applied")


def log_tf_data_performance_tips():
    """Log performance tips for tf.data usage."""

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
