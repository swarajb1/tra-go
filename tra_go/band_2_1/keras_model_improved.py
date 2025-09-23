"""
TRA-GO Band 2_1 Improved Keras Model

This module implements an advanced neural network architecture for stock price prediction
within the TRA-GO trading system. The model is specifically designed for Band 2_1, which
focuses on predicting price movements using a combination of recurrent neural networks
and attention mechanisms.

Architecture Overview:
- Recurrent Layers: Uses GRU or LSTM layers for sequence processing
- Attention Mechanism: Optional attention layer for improved interpretability
- Custom Activation: Specialized activation function for financial predictions
- Multi-Output Support: Can output both predictions and attention weights

Key Enhancements:
- Functional API implementation for better control over model architecture
- Configurable recurrent layers (GRU vs LSTM) for performance optimization
- Attention weights extraction for model interpretability
- Type hints and comprehensive documentation
- Modular design supporting future extensions (multi-modal inputs)

Technical Details:
- Input: Time-series financial data (OHLC + volume)
- Output: Predicted price range (min/max) with optional attention weights
- Loss Function: Custom multi-objective loss combining RMSE and domain-specific metrics
- Optimization: Configurable optimizer with learning rate scheduling

Future Extensions:
- Bayesian neural networks for uncertainty quantification
- Multi-modal inputs (news sentiment, technical indicators)
- Transformer-based architectures for longer sequence dependencies

Note: Training enhancements (early stopping, learning rate decay) are implemented
separately in training scripts to maintain modularity.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from band_2_1.keras_model_common import CustomActivationLayer, ModelCompileConfig
from core.config import settings
from numpy.typing import NDArray
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Attention,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    TimeDistributed,
)
from tensorflow.keras.models import Model


class AttentionWithWeights(Layer):
    """
    Custom Keras layer implementing attention mechanism with weight extraction.

    This layer extends the standard TensorFlow Attention layer to provide both
    the attended output and the attention weights for interpretability. The attention
    weights can be used to understand which parts of the input sequence the model
    focuses on when making predictions.

    The layer computes scaled dot-product attention where:
    - Query: Current time step representation
    - Key: All time step representations
    - Value: All time step representations (same as key in self-attention)

    Attributes:
        attention (Attention): Underlying TensorFlow attention layer

    Args:
        use_scale (bool): Whether to scale attention scores by sqrt(d_k).
                         Defaults to True for better gradient flow.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: (attended_output, attention_weights)
            - attended_output: Weighted sum of values based on attention scores
            - attention_weights: Softmax-normalized attention scores

    Example:
        >>> attention_layer = AttentionWithWeights(use_scale=True)
        >>> attended, weights = attention_layer([query, key, value])
        >>> # weights shape: (batch_size, seq_len, seq_len)
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
    include_attention: bool = False,
    use_gru: bool = False,
) -> Model:
    """
    Construct an untrained Keras model for stock price prediction.

    This function builds a deep recurrent neural network optimized for financial time series
    prediction. The model uses a stacked RNN architecture with optional attention mechanism
    and custom activation functions tailored for trading applications.

    Model Architecture:
    1. Input Layer: Accepts time-series sequences of financial data
    2. Recurrent Layers: Configurable number of GRU/LSTM layers with decreasing units
    3. Dropout: Exponential decay dropout for regularization
    4. Time-Distributed Dense: Per-timestep feature extraction
    5. Attention (optional): Self-attention mechanism for sequence weighting
    6. Global Pooling: Aggregates sequence information
    7. Custom Activation: Specialized activation for price prediction

    Args:
        X_train (NDArray): Training input data used for shape inference.
                           Shape: (samples, sequence_length, features)
        Y_train (NDArray): Training target data. Currently unused but maintained
                          for API consistency. Shape: (samples, output_features)
        include_attention (bool): If True, includes attention mechanism and returns
                                 attention weights alongside predictions. Defaults to False.
        use_gru (bool): If True, uses GRU layers instead of LSTM for computational
                       efficiency. Defaults to True.

    Returns:
        Model: Compiled Keras model ready for training.
            - Single output: Predictions only (if include_attention=False)
            - Multi-output: [predictions, attention_weights] (if include_attention=True)
            - Model name: "Improved_TRA_GO_Model"

    Configuration:
        - Number of layers: Determined by settings.NUMBER_OF_LAYERS
        - Neurons per layer: settings.NUMBER_OF_NEURONS with exponential decay
        - Recurrent dropout: settings.RECURRENT_DROPOUT
        - Initial dropout: settings.INITIAL_DROPOUT
        - Optimizer: Configured via ModelCompileConfig
        - Loss function: Custom multi-objective loss
        - Metrics: Defined in ModelCompileConfig

    Notes:
        - The model automatically prints a summary after compilation
        - For multi-modal extensions, see get_multi_modal_model()
        - Bayesian variants should be implemented using TensorFlow Probability
        - Training callbacks (early stopping, LR scheduling) are handled externally

    Example:
        >>> model = get_untrained_model(X_train, Y_train, include_attention=True, use_gru=True)
        >>> # Train with: model.fit(X_train, [Y_train, None], ...) for attention
    """
    inputs = Input(shape=X_train[0].shape, name="input_layer")
    x = inputs

    # Recurrent layers with GRUs or LSTMs
    rnn_layer = GRU if use_gru else LSTM

    for layer_num in range(settings.NUMBER_OF_LAYERS):
        units = settings.NUMBER_OF_NEURONS // (2**layer_num)
        x = rnn_layer(
            units=units,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            recurrent_dropout=settings.RECURRENT_DROPOUT,
            unroll=False,
            name=f"rnn_{'gru' if use_gru else 'lstm'}_{layer_num}",
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

    model_compile_config = ModelCompileConfig()

    # Compile
    model.compile(
        optimizer=model_compile_config.optimizer,
        loss=ModelCompileConfig.loss,
        metrics=ModelCompileConfig.metrics,
    )

    model.summary()
    print("\n" * 2)

    return model


def extract_attention_weights(model: Model, input_data: NDArray) -> np.ndarray:
    """
    Extract attention weights from a trained model with attention mechanism.

    This function creates a sub-model that isolates the attention weights output,
    allowing analysis of which parts of the input sequence the model attends to
    during prediction. Useful for model interpretability and debugging.

    Args:
        model (Model): A trained Keras model that includes attention mechanism.
                      Must have been created with include_attention=True.
        input_data (NDArray): Input sequences for weight extraction.
                             Shape: (batch_size, sequence_length, features)

    Returns:
        np.ndarray: Attention weights tensor.
                   Shape: (batch_size, sequence_length, sequence_length)
                   Values represent attention scores between query and key positions.

    Raises:
        ValueError: If the model does not contain an attention layer.

    Notes:
        - Weights are softmax-normalized and represent relative importance
        - Higher values indicate stronger attention between time steps
        - Can be visualized using plot_attention_weights()

    Example:
        >>> weights = extract_attention_weights(model, X_test)
        >>> print(f"Attention shape: {weights.shape}")  # (batch, seq_len, seq_len)
        >>> plot_attention_weights(weights, sample_idx=0)
    """
    if not any("attention" in layer.name for layer in model.layers):
        raise ValueError("Model does not include attention layer.")

    # Get the attention weights output
    attention_model = Model(inputs=model.input, outputs=model.get_layer("attention_with_weights").output[1])
    weights = attention_model.predict(input_data)
    return weights


def plot_attention_weights(weights: np.ndarray, sample_idx: int = 0, save_path: Optional[str] = None) -> None:
    """
    Visualize attention weights as a heatmap for model interpretability.

    Creates a heatmap showing how different time steps in the input sequence
    attend to each other. The x-axis represents key positions, y-axis represents
    query positions. Brighter colors indicate stronger attention connections.

    Args:
        weights (np.ndarray): Attention weights from extract_attention_weights().
                            Shape: (batch_size, sequence_length, sequence_length)
        sample_idx (int): Index of the sample to visualize from the batch.
                         Defaults to 0 (first sample).
        save_path (Optional[str]): File path to save the plot image.
                                  If None, displays the plot interactively.
                                  Supports common image formats (.png, .jpg, .pdf, etc.)

    Raises:
        ValueError: If weights array is not 3-dimensional.

    Notes:
        - Uses viridis colormap for better color perception
        - Automatically adjusts tick spacing for readability
        - Useful for understanding temporal dependencies learned by the model
        - Can help identify if the model focuses on recent vs. historical data

    Example:
        >>> weights = extract_attention_weights(model, X_sample)
        >>> plot_attention_weights(weights, sample_idx=0, save_path="attention.png")
        >>> # Shows heatmap and saves to file
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
    Placeholder function for future multi-modal model implementation.

    This function outlines the structure for a model that combines time-series financial
    data with additional modalities like news sentiment embeddings. Currently returns
    a single-modal model as a placeholder.

    Future Implementation Plan:
    1. Time-series Branch: RNN processing of OHLC + volume data
    2. News Branch: Dense layers processing news embeddings
    3. Fusion Layer: Concatenation or attention-based fusion
    4. Joint Prediction: Combined features for final price prediction

    Args:
        X_train (NDArray): Time-series training data for shape inference.
                          Shape: (samples, sequence_length, features)
        Y_train (NDArray): Training target data (currently unused).
        news_embedding_dim (int): Dimensionality of news embeddings.
                                Defaults to 128.

    Returns:
        Model: Currently returns single-modal model. Future versions will return
               a true multi-modal architecture.

    Notes:
        - This is a stub implementation for future development
        - Multi-modal models can improve prediction accuracy by incorporating
          external signals like news sentiment, social media, or economic indicators
        - Fusion strategies may include: concatenation, cross-attention, or
          hierarchical fusion networks

    Example:
        >>> # Future usage (not implemented yet)
        >>> model = get_multi_modal_model(X_train, Y_train, news_embedding_dim=256)
        >>> # Would process both time-series and news data
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
