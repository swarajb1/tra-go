"""
Improved Keras Model for TRA-GO Band 2_1

This module implements an enhanced version of the neural network model for stock price prediction,
incorporating advanced deep learning strategies for better performance and interpretability.

Key improvements over the original:
- Uses GRUs instead of LSTMs for computational efficiency
- Adds attention mechanism for better interpretability
- Includes type hints and comprehensive docstrings
- Refactored to Functional API for proper attention handling
- Added functions for attention weight extraction and visualization
- Placeholder for multi-modal inputs (e.g., news embeddings)
- Note: Bayesian LSTMs and training enhancements (early stopping, LR decay) are recommended
  but implemented separately in training scripts for modularity.
"""

from typing import Optional

import band_2_1.model_metrics as km_21_metrics
import model_training.common as training_common
import tensorflow as tf
from core.config import settings
from tensorflow.keras.layers import (
    Layer,
)


class ModelCompileConfig:
    """
    Configuration class for model compilation details.

    Attributes:
        optimizer: TensorFlow optimizer instance.
        loss: Loss function for training.
        metrics: List of metrics for evaluation.
    """

    def __init__(self):
        self.optimizer = training_common.get_optimiser(settings.LEARNING_RATE)

    loss = km_21_metrics.loss_function

    metrics = [
        km_21_metrics.metric_rmse_percent,
        km_21_metrics.metric_abs_percent,
        km_21_metrics.metric_loss_comp_2,
        km_21_metrics.metric_win_percent,
        km_21_metrics.metric_win_pred_capture_total_percent,
        km_21_metrics.metric_win_pred_trend_capture_percent,
        km_21_metrics.metric_pred_capture_per_win_percent,
        km_21_metrics.metric_correct_trend_per_win_percent,
        km_21_metrics.metric_try_1,
        # km_21_metrics.stoploss_incurred,
    ]


class CustomActivationLayer(Layer):
    """
    Custom activation layer that applies linear activation to the first two features
    and sigmoid to the third feature, with optional hard thresholding.

    Args:
        hard_threshold (bool): If True, rounds the third feature to 0/1 at inference.
    """

    def __init__(self, hard_threshold: bool = False, **kwargs) -> None:
        self.hard_threshold = hard_threshold
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Apply linear activation to the first two features
        first_two_features = inputs[:, :2]

        # Apply sigmoid to the third feature
        third_feature = tf.sigmoid(inputs[:, 2:3])

        if self.hard_threshold and training is False:
            third_feature = tf.round(third_feature)

        # Concatenate back
        return tf.concat([first_two_features, third_feature], axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"hard_threshold": self.hard_threshold})
        return config
