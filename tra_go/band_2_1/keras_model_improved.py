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
import matplotlib.pyplot as plt
import model_training.common as training_common
import numpy as np
import tensorflow as tf
from core.config import settings
from numpy.typing import NDArray
from tensorflow.keras.layers import (
    GRU,
    Attention,
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
    Configuration class for model compilation details.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        optimizer: TensorFlow optimizer instance.
        loss: Loss function for training.
        metrics: List of metrics for evaluation.
    """

    def __init__(self) -> None:
        self.learning_rate: float = 0.001
        self.optimizer = training_common.get_optimiser(self.learning_rate)
        self.loss = km_21_metrics.loss_function
        self.metrics = [
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
            km_21_metrics.stoploss_incurred,
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


class AttentionWithWeights(Layer):
    """
    Custom attention layer that returns both attended output and attention weights for interpretability.
    """

    def __init__(self, use_scale: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = Attention(use_scale=use_scale)

    def call(self, inputs: list) -> tuple[tf.Tensor, tf.Tensor]:
        query, key, value = inputs
        attended = self.attention([query, key, value])
        # Compute attention weights (simplified; assumes single-head)
        scores = tf.matmul(query, key, transpose_b=True)
        if self.attention.use_scale:
            scores /= tf.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        return attended, weights

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"use_scale": self.attention.use_scale})
        return config


def get_untrained_model(
    X_train: NDArray,
    Y_train: NDArray,
    include_attention: bool = True,
    use_gru: bool = True,
) -> Model:
    """
    Builds an improved untrained Keras model for stock price prediction using Functional API.

    This enhanced version uses GRUs for efficiency, adds attention for interpretability,
    and includes better documentation.

    Args:
        X_train (NDArray): Training input data for shape inference.
        Y_train (NDArray): Training target data (unused but kept for consistency).
        include_attention (bool): Whether to include attention mechanism.
        use_gru (bool): Whether to use GRUs instead of LSTMs.

    Returns:
        Model: Compiled Keras model ready for training. If attention is included,
               the model outputs both predictions and attention weights.

    Notes:
        - For multi-modal inputs (e.g., news embeddings), extend the input layer.
        - Bayesian variants can be implemented using TensorFlow Probability.
        - Training enhancements like early stopping should be added in training scripts.
    """
    inputs = Input(shape=X_train[0].shape, name="input_layer")
    x = inputs

    # Recurrent layers with GRUs or LSTMs
    rnn_layer = GRU if use_gru else tf.keras.layers.LSTM
    for layer_num in range(settings.NUMBER_OF_LAYERS):
        units = settings.NUMBER_OF_NEURONS // (2**layer_num)
        x = Bidirectional(
            rnn_layer(
                units=units,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                recurrent_dropout=settings.RECURRENT_DROPOUT,
                unroll=False,
                name=f"bidirectional_rnn_{layer_num}",
            ),
            name=f"bidirectional_{layer_num}",
        )(x)
        # Exponential dropout
        dropout_rate = (1 + settings.INITIAL_DROPOUT) ** (1 / (layer_num + 1)) - 1
        x = Dropout(dropout_rate, name=f"dropout_{layer_num}")(x)

    # TimeDistributed Dense
    x = TimeDistributed(Dense(units=3), name="time_distributed_dense")(x)

    # Attention mechanism (optional)
    attention_weights = None
    if include_attention:
        attended, attention_weights = AttentionWithWeights(use_scale=True, name="attention_with_weights")([x, x, x])
        x = attended

    # Global pooling
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Custom activation
    outputs = CustomActivationLayer(name="custom_activation")(x)

    # If attention, include weights in outputs for extraction
    if include_attention:
        model = Model(inputs=inputs, outputs=[outputs, attention_weights], name="Improved_TRA_GO_Model")
    else:
        model = Model(inputs=inputs, outputs=outputs, name="Improved_TRA_GO_Model")

    # Compile
    compile_details = ModelCompileDetails()
    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    model.summary()
    print("\n" * 2)

    return model


def extract_attention_weights(model: Model, input_data: NDArray) -> np.ndarray:
    """
    Extracts attention weights from a trained model with attention.

    Args:
        model (Model): Trained Keras model with attention.
        input_data (NDArray): Input sequence for which to extract weights.

    Returns:
        np.ndarray: Attention weights array of shape (batch_size, seq_len, seq_len).

    Raises:
        ValueError: If the model does not include attention.
    """
    if not any("attention" in layer.name for layer in model.layers):
        raise ValueError("Model does not include attention layer.")

    # Get the attention weights output
    attention_model = Model(inputs=model.input, outputs=model.get_layer("attention_with_weights").output[1])
    weights = attention_model.predict(input_data)
    return weights


def plot_attention_weights(weights: np.ndarray, sample_idx: int = 0, save_path: Optional[str] = None) -> None:
    """
    Plots attention weights for a given sample.

    Args:
        weights (np.ndarray): Attention weights from extract_attention_weights.
        sample_idx (int): Index of the sample to plot (default: 0).
        save_path (Optional[str]): Path to save the plot (optional).
    """
    if weights.ndim != 3:
        raise ValueError("Weights must be 3D (batch_size, seq_len, seq_len).")

    attn = weights[sample_idx]  # Shape: (seq_len, seq_len)

    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Position (Time Step)")
    plt.ylabel("Query Position (Time Step)")
    plt.title(f"Attention Weights for Sample {sample_idx}")
    plt.xticks(range(0, attn.shape[1], max(1, attn.shape[1] // 10)))
    plt.yticks(range(0, attn.shape[0], max(1, attn.shape[0] // 10)))
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Placeholder for multi-modal model extension
def get_multi_modal_model(X_train: NDArray, Y_train: NDArray, news_embedding_dim: int = 128) -> Model:
    """
    Placeholder for a multi-modal model incorporating news embeddings.

    Args:
        X_train (NDArray): Time-series input.
        Y_train (NDArray): Targets.
        news_embedding_dim (int): Dimension of news embeddings.

    Returns:
        Model: Multi-modal model (not fully implemented).

    Note: This is a stub. Implement by adding news input and fusion layers.
    """
    # Time-series branch
    ts_input = Input(shape=X_train[0].shape, name="time_series_input")
    # ... (add RNN layers as above)

    # News branch
    news_input = Input(shape=(news_embedding_dim,), name="news_input")
    # ... (add dense layers for news)

    # Fusion
    # combined = Concatenate()([ts_output, news_output])
    # ... (add final layers)

    # For now, return single-modal model
    return get_untrained_model(X_train, Y_train)
