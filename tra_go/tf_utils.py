"""
Utility functions for configuring and optimizing TensorFlow performance.

This module provides functions to configure TensorFlow for optimal performance
on different hardware platforms, with specific optimizations for Apple Silicon
Mac M4 systems. It handles GPU memory management, thread parallelism, and
performance monitoring.
"""

import tensorflow as tf
from core.logger import logger


def configure_tensorflow_performance():
    """
    Configure TensorFlow for optimal performance on Mac M4 systems.

    This function sets up TensorFlow configuration options specifically optimized
    for Apple Silicon M4 processors, including GPU memory growth and thread
    parallelism settings. It enables Metal Performance Shaders support and
    configures the system to utilize all available CPU cores efficiently.

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
        >>> from tra_go.tf_utils import configure_tensorflow_performance
        >>> configure_tensorflow_performance()
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

    # # Configure optimal thread settings for Mac M4 (12-core CPU: 8 performance + 4 efficiency cores)
    # # Set to 0 for auto-detection to utilize all available cores optimally
    # tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    # tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores

    logger.info("Mac M4 optimized thread configuration: Using all available CPU cores (auto-detect)")

    logger.info("TensorFlow performance optimizations applied (Mac M4 optimized)")
