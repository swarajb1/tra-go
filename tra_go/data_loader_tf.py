"""
Enhanced DataLoader using tf.data pipelines with batching and prefetching for optimal performance.
"""

from typing import Optional

import numpy as np
import tensorflow as tf
from core.config import settings
from core.logger import logger
from data_loader import DataLoader

from database.enums import BandType, IntervalType, TickerOne


class TensorFlowDataLoader(DataLoader):
    """
    Enhanced DataLoader that uses tf.data pipelines for efficient data loading.

    Features:
    - tf.data.Dataset for memory-efficient data loading
    - Automatic batching with configurable batch size
    - Prefetching for better performance
    - Data shuffling for training
    - Memory optimization
    """

    def __init__(
        self,
        ticker: TickerOne,
        interval: IntervalType,
        x_type: BandType,
        y_type: BandType,
        test_size: float = 0.2,
        batch_size: Optional[int] = None,
        shuffle_buffer_size: int = 1000,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        enable_shuffle: bool = True,
    ):
        """
        Initialize TensorFlow DataLoader.

        Args:
            ticker: Stock ticker to load data for
            interval: Time interval for data
            x_type: Input data band type
            y_type: Output data band type
            test_size: Fraction of data to use for testing
            batch_size: Batch size for training. If None, uses settings.BATCH_SIZE
            shuffle_buffer_size: Buffer size for shuffling training data
            prefetch_buffer_size: Buffer size for prefetching. Use tf.data.AUTOTUNE for auto-tuning
            enable_shuffle: Whether to shuffle training data
        """
        # Initialize parent class to load and prepare all data
        super().__init__(ticker, interval, x_type, y_type, test_size)

        self.batch_size = batch_size or settings.BATCH_SIZE
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size
        self.enable_shuffle = enable_shuffle

        logger.info(f"TensorFlow DataLoader initialized with batch_size={self.batch_size}")
        logger.info(f"Shuffle enabled: {self.enable_shuffle}, buffer_size: {self.shuffle_buffer_size}")
        logger.info(f"Prefetch buffer_size: {self.prefetch_buffer_size}")

        # Create tf.data datasets
        self._create_tf_datasets()

    def _create_tf_datasets(self) -> None:
        """Create optimized tf.data.Dataset objects for training and testing."""

        # Convert numpy arrays to tf.data.Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            {"x": self.train_x_data.astype(np.float32), "y": self.train_y_data.astype(np.float32)},
        )

        test_dataset = tf.data.Dataset.from_tensor_slices(
            {"x": self.test_x_data.astype(np.float32), "y": self.test_y_data.astype(np.float32)},
        )

        # Extract x and y from the dictionary
        train_dataset = train_dataset.map(lambda data: (data["x"], data["y"]), num_parallel_calls=tf.data.AUTOTUNE)

        test_dataset = test_dataset.map(lambda data: (data["x"], data["y"]), num_parallel_calls=tf.data.AUTOTUNE)

        # Apply optimizations to training dataset
        if self.enable_shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=self.shuffle_buffer_size, reshuffle_each_iteration=True)
            logger.info(f"Training data shuffling enabled with buffer size: {self.shuffle_buffer_size}")

        # Batch the datasets
        train_dataset = train_dataset.batch(batch_size=self.batch_size, drop_remainder=False)

        test_dataset = test_dataset.batch(batch_size=self.batch_size, drop_remainder=False)

        # Prefetch for performance
        train_dataset = train_dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        test_dataset = test_dataset.prefetch(buffer_size=self.prefetch_buffer_size)

        # Cache datasets in memory for better performance (if data fits in memory)
        # Note: Only cache if the dataset is reasonably small to avoid memory issues
        dataset_size_mb = (self.train_x_data.nbytes + self.train_y_data.nbytes) / (1024 * 1024)
        if dataset_size_mb < 1000:  # Cache if dataset is less than 1GB
            train_dataset = train_dataset.cache()
            test_dataset = test_dataset.cache()
            logger.info(f"Datasets cached in memory (size: {dataset_size_mb:.1f} MB)")
        else:
            logger.info(f"Datasets not cached due to size: {dataset_size_mb:.1f} MB")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        logger.info("tf.data datasets created successfully")
        logger.info(f"Training dataset: {train_dataset}")
        logger.info(f"Test dataset: {test_dataset}")

    def get_tf_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Get the optimized tf.data.Dataset objects for training and testing.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        return self.train_dataset, self.test_dataset

    def get_dataset_info(self) -> dict:
        """
        Get information about the datasets.

        Returns:
            Dictionary containing dataset information
        """
        return {
            "batch_size": self.batch_size,
            "train_samples": self.train_x_data.shape[0],
            "test_samples": self.test_x_data.shape[0],
            "input_shape": self.train_x_data.shape[1:],
            "output_shape": self.train_y_data.shape[1:],
            "train_steps_per_epoch": int(np.ceil(self.train_x_data.shape[0] / self.batch_size)),
            "test_steps_per_epoch": int(np.ceil(self.test_x_data.shape[0] / self.batch_size)),
            "shuffle_enabled": self.enable_shuffle,
            "shuffle_buffer_size": self.shuffle_buffer_size if self.enable_shuffle else None,
            "prefetch_buffer_size": self.prefetch_buffer_size,
        }

    def get_train_test_split_data_tf(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Get tf.data datasets instead of numpy arrays.
        This is the TensorFlow-optimized version of get_train_test_split_data().

        Returns:
            Tuple of (train_dataset, test_dataset) as tf.data.Dataset objects
        """
        return self.get_tf_datasets()

    def create_validation_dataset(self, validation_split: float = 0.2) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create a validation dataset by splitting the training data.

        Args:
            validation_split: Fraction of training data to use for validation

        Returns:
            Tuple of (new_train_dataset, validation_dataset)
        """
        if validation_split <= 0 or validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")

        # Calculate split sizes
        total_train_samples = self.train_x_data.shape[0]
        val_samples = int(total_train_samples * validation_split)
        new_train_samples = total_train_samples - val_samples

        # Split the data
        train_x_new = self.train_x_data[:new_train_samples]
        train_y_new = self.train_y_data[:new_train_samples]

        val_x = self.train_x_data[new_train_samples:]
        val_y = self.train_y_data[new_train_samples:]

        # Create new datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_x_new.astype(np.float32), train_y_new.astype(np.float32)),
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((val_x.astype(np.float32), val_y.astype(np.float32)))

        # Apply optimizations
        if self.enable_shuffle:
            train_dataset = train_dataset.shuffle(
                buffer_size=min(self.shuffle_buffer_size, new_train_samples),
                reshuffle_each_iteration=True,
            )

        train_dataset = train_dataset.batch(self.batch_size).prefetch(self.prefetch_buffer_size)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(self.prefetch_buffer_size)

        logger.info(f"Created validation split: train={new_train_samples}, val={val_samples}")

        return train_dataset, val_dataset

    def benchmark_performance(self, num_batches: int = 10) -> dict:
        """
        Benchmark the performance of data loading.

        Args:
            num_batches: Number of batches to benchmark

        Returns:
            Dictionary with performance metrics
        """
        import time

        logger.info(f"Benchmarking data loading performance with {num_batches} batches...")

        # Benchmark training dataset
        start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(self.train_dataset.take(num_batches)):
            # Simulate some processing
            _ = tf.reduce_mean(x_batch)
            if i >= num_batches - 1:
                break

        train_time = time.time() - start_time

        # Benchmark test dataset
        start_time = time.time()
        for i, (x_batch, y_batch) in enumerate(self.test_dataset.take(num_batches)):
            # Simulate some processing
            _ = tf.reduce_mean(x_batch)
            if i >= num_batches - 1:
                break

        test_time = time.time() - start_time

        info = self.get_dataset_info()

        performance = {
            "train_time_per_batch": round(train_time / num_batches, 4),
            "test_time_per_batch": round(test_time / num_batches, 4),
            "total_train_batches": int(info["train_steps_per_epoch"]),
            "total_test_batches": int(info["test_steps_per_epoch"]),
            "estimated_epoch_time": round((train_time / num_batches) * info["train_steps_per_epoch"], 4),
            "batch_size": int(self.batch_size),
            "samples_per_second_train": int((self.batch_size * num_batches) / train_time),
            "samples_per_second_test": int((self.batch_size * num_batches) / test_time),
        }

        logger.info("Performance benchmark results:")
        for key, value in performance.items():
            logger.info(f"  {key}: {value}")

        return performance


def create_optimized_data_loader(
    ticker: TickerOne,
    interval: IntervalType,
    x_type: BandType,
    y_type: BandType,
    test_size: float = 0.2,
    **kwargs,
) -> TensorFlowDataLoader:
    """
    Factory function to create an optimized TensorFlow DataLoader.

    Args:
        ticker: Stock ticker
        interval: Time interval
        x_type: Input data band type
        y_type: Output data band type
        test_size: Test set size
        **kwargs: Additional arguments for TensorFlowDataLoader

    Returns:
        Configured TensorFlowDataLoader instance
    """
    return TensorFlowDataLoader(
        ticker=ticker,
        interval=interval,
        x_type=x_type,
        y_type=y_type,
        test_size=test_size,
        **kwargs,
    )
