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
import tensorflow as tf
from band_2_1.keras_model_common import CustomActivationLayer, ModelCompileConfig
from core.config import settings
from numpy.typing import NDArray
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    TimeDistributed,
)
from tensorflow.keras.models import Model


def get_custom_scope():
    """
    Get the custom scope dictionary containing all metrics and custom objects
    required for model loading and evaluation.

    This function centralizes the definition of custom objects needed when loading
    Keras models with custom metrics and layers. It includes all the custom metrics
    from band_2_1.model_metrics and the CustomActivationLayer.

    Returns:
        dict: Dictionary mapping custom object names to their implementations,
              suitable for use with tf.keras.models.load_model(custom_objects=...)
    """
    return {
        "loss_function": km_21_metrics.loss_function,
        "metric_rmse_percent": km_21_metrics.metric_rmse_percent,
        "metric_abs_percent": km_21_metrics.metric_abs_percent,
        "metric_loss_comp_2": km_21_metrics.metric_loss_comp_2,
        "metric_win_percent": km_21_metrics.metric_win_percent,
        "metric_win_pred_trend_capture_percent": km_21_metrics.metric_win_pred_trend_capture_percent,
        "metric_win_pred_capture_total_percent": km_21_metrics.metric_win_pred_capture_total_percent,
        "metric_pred_capture_per_win_percent": km_21_metrics.metric_pred_capture_per_win_percent,
        "metric_correct_trend_per_win_percent": km_21_metrics.metric_correct_trend_per_win_percent,
        "metric_try_1": km_21_metrics.metric_try_1,
        "metric_try_2": km_21_metrics.metric_try_2,
        "stoploss_incurred": km_21_metrics.stoploss_incurred,
        "CustomActivationLayer": CustomActivationLayer,
        "metric_win_pred_capture_percent": km_21_metrics.metric_placeholder,
        "metric_win_correct_trend_percent": km_21_metrics.metric_placeholder,
        "metric_correct_trends_full": km_21_metrics.metric_placeholder,
    }


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

    compile_details = ModelCompileConfig()

    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    # Build optimizer with model trainable variables for TensorFlow 2.15+ compatibility
    compile_details.optimizer.build(model.trainable_variables)

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

    compile_details = ModelCompileConfig()

    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    # Build optimizer with model trainable variables for TensorFlow 2.15+ compatibility
    compile_details.optimizer.build(model.trainable_variables)

    model.summary()

    print("\n" * 2)

    return model
