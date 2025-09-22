"""
TRA-GO Band 2.1 Keras Model Implementation

This module defines the neural network architecture for Band 2.1 of the TRA-GO trading system.
It implements a bidirectional LSTM-based model for predicting stock price movements with
custom activation layers and comprehensive evaluation metrics.

The model architecture includes:
- Multiple bidirectional LSTM layers with decreasing neuron counts
- TimeDistributed dense layers for sequence processing
- Global average pooling for feature aggregation
- Custom activation layer for specialized output processing
- Extensive custom metrics for trading performance evaluation
"""

import band_2_1.model_metrics as km_21_metrics
import model_training.common as training_common
import tensorflow as tf
from core.config import settings
from numpy.typing import NDArray
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    TimeDistributed,
)
from tensorflow.keras.models import Model


class ModelCompileDetails:
    """
    Configuration class for model compilation settings.

    This class encapsulates all the compilation parameters for the Band 2.1 Keras model,
    including optimizer, loss function, and evaluation metrics. It provides a centralized
    way to manage model training configuration.

    Attributes:
        learning_rate (float): Learning rate for the optimizer (default: 0.001)
        optimizer: TensorFlow optimizer instance configured with the learning rate
        loss: Custom loss function from band_2_1 model metrics
        metrics (list): List of custom evaluation metrics for trading performance
    """

    def __init__(self):
        self.learning_rate: float = 0.001
        self.optimizer = training_common.get_optimiser(self.learning_rate)

        self.loss = km_21_metrics.loss_function

        self.metrics = [
            # km_tf.metric_rmse_percent,
            # km_tf.metric_abs_percent,
            training_common.metric_rmse_percent,
            training_common.metric_abs_percent,
            km_21_metrics.metric_correct_trends_full,
            km_21_metrics.metric_loss_comp_2,
            km_21_metrics.metric_win_percent,
            km_21_metrics.metric_win_pred_capture_percent,
            km_21_metrics.metric_win_pred_capture_total_percent,
            km_21_metrics.metric_win_correct_trend_percent,
            km_21_metrics.metric_win_pred_trend_capture_percent,
            km_21_metrics.metric_try_1,
            # km_21_metrics.metric_try_2,
            km_21_metrics.stoploss_incurred,
        ]


class CustomActivationLayer(Layer):
    def __init__(self, hard_threshold: bool = False, **kwargs):
        """Custom activation that leaves first two features linear and applies a sigmoid
        to the third feature.

        Args:
            hard_threshold: If True, the third feature will be rounded to 0/1 when
                the layer is called with training=False (i.e. at inference). When
                False (default) the layer outputs a continuous sigmoid value which
                is differentiable for training.
        """
        self.hard_threshold = hard_threshold
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        # Apply linear activation to the first two features
        first_two_features = inputs[:, :2]

        # Apply sigmoid activation to the third feature. Do NOT round during
        # training so gradients can flow. If hard_threshold is enabled, round only
        # at inference time (when training is False).
        third_feature = tf.sigmoid(inputs[:, 2:3])

        if self.hard_threshold and training is False:
            third_feature = tf.round(third_feature)

        # Concatenate the features back together
        return tf.concat([first_two_features, third_feature], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"hard_threshold": self.hard_threshold})
        return config


def get_untrained_model_old(X_train: NDArray, Y_train: NDArray) -> Model:
    """
    Create an older version of the untrained Band 2.1 Keras model.

    This function builds a sequential neural network model with bidirectional LSTM layers,
    dropout regularization, and a custom activation layer. The architecture uses ReLU
    activation for LSTM layers and includes time-distributed dense layers for sequence
    processing.

    Args:
        X_train (NDArray): Training input data used to determine input shape
        Y_train (NDArray): Training target data (not used in model creation, kept for API consistency)

    Returns:
        Model: Uncompiled Keras Sequential model ready for training

    Note:
        This is a legacy model architecture. Use get_untrained_model() for the current implementation.
    """
    model = tf.keras.models.Sequential()

    model.add(Input(shape=(X_train[0].shape)))

    for layer_num in range(settings.NUMBER_OF_LAYERS):
        model.add(
            Bidirectional(
                LSTM(
                    units=settings.NUMBER_OF_NEURONS // (pow(2, layer_num)),
                    return_sequences=True,
                    activation="relu",
                ),
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(
            Dropout(
                pow(
                    1 + settings.INITIAL_DROPOUT,
                    1 / (layer_num + 1),
                )
                - 1,
            ),
        )

    # ---start - Plan - london
    # model.add(TimeDistributed(Dense(units=NUMBER_OF_NEURONS)))

    # model.add(GlobalAveragePooling1D())
    # ---end - Plan - london

    # ---start - Plan - now
    model.add(TimeDistributed(Dense(units=3)))

    model.add(GlobalAveragePooling1D())
    # ---end - Plan - now

    # ---start - Plan zero
    # model.add(
    #     TimeDistributed(
    #         Dense(units=settings.NUMBER_OF_NEURONS // (pow(2, settings.NUMBER_OF_LAYERS)), activation="relu")
    #     )
    # )

    # model.add(Flatten())

    # model.add(Dense(units=3))
    # ---end - Plan zero

    # ---start - Plan polo
    # model.add(Dense(units=3))
    # ---end - Plan polo

    model.add(CustomActivationLayer())

    compile_details = ModelCompileDetails()

    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    model.summary()

    print("\n" * 2)

    return model


def get_untrained_model(X_train: NDArray, Y_train: NDArray) -> Model:
    """
    Create the current untrained Band 2.1 Keras model for stock price prediction.

    This function builds a sequential neural network model optimized for trading predictions.
    The architecture features:
    - Multiple bidirectional LSTM layers with tanh activation and recurrent dropout
    - Exponentially decreasing dropout rates for regularization
    - TimeDistributed dense layers for sequence processing
    - Global average pooling for feature aggregation
    - Custom activation layer for specialized output processing

    The model is designed to predict stock price movements with comprehensive evaluation
    metrics including RMSE, trend accuracy, win percentage, and trading-specific metrics.

    Args:
        X_train (NDArray): Training input data used to determine input shape.
            Expected shape: (batch_size, sequence_length, features)
        Y_train (NDArray): Training target data (not used in model creation, kept for API consistency)

    Returns:
        Model: Compiled Keras Sequential model ready for training with custom optimizer,
               loss function, and comprehensive evaluation metrics

    Note:
        The model uses settings from core.config for hyperparameters like NUMBER_OF_LAYERS,
        NUMBER_OF_NEURONS, and INITIAL_DROPOUT_PERCENT.
    """
    model = tf.keras.models.Sequential()

    model.add(Input(shape=(X_train[0].shape)))

    for layer_num in range(settings.NUMBER_OF_LAYERS):
        model.add(
            Bidirectional(
                LSTM(
                    units=settings.NUMBER_OF_NEURONS // (pow(2, layer_num)),
                    return_sequences=True,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    recurrent_dropout=0,
                    unroll=False,
                ),
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(
            Dropout(
                pow(
                    1 + settings.INITIAL_DROPOUT,
                    1 / (layer_num + 1),
                )
                - 1,
            ),
        )

    # ---start - Plan - london
    # model.add(TimeDistributed(Dense(units=NUMBER_OF_NEURONS)))

    # model.add(GlobalAveragePooling1D())
    # ---end - Plan - london

    # ---start - Plan - now
    model.add(TimeDistributed(Dense(units=3)))

    model.add(GlobalAveragePooling1D())
    # ---end - Plan - now

    # ---start - Plan zero
    # model.add(
    #     TimeDistributed(
    #         Dense(units=settings.NUMBER_OF_NEURONS // (pow(2, settings.NUMBER_OF_LAYERS)), activation="relu")
    #     )
    # )

    # model.add(Flatten())

    # model.add(Dense(units=3))
    # ---end - Plan zero

    # ---start - Plan polo
    # model.add(Dense(units=3))
    # ---end - Plan polo

    model.add(CustomActivationLayer())

    compile_details = ModelCompileDetails()

    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    model.summary()

    print("\n" * 2)

    return model
