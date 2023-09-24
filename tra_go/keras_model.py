from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras import backend as K
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf


def get_untrained_model(X_train, y_type):
    model = keras.Sequential()

    model.add(
        LSTM(
            units=256,
            input_shape=(X_train[0].shape),
            return_sequences=True,
            activation="relu",
        )
    )
    model.add(Dropout(0.20))

    model.add(
        LSTM(
            units=256,
            return_sequences=True,
            activation="relu",
        )
    )
    model.add(Dropout(0.10))

    model.add(
        LSTM(
            units=256,
            return_sequences=True,
            activation="relu",
        )
    )
    model.add(Dropout(0.20))

    model.add(
        LSTM(
            units=256,
            return_sequences=True,
            activation="relu",
        )
    )
    model.add(Dropout(0.10))

    if y_type == "hl":
        model.add(Flatten())

    if y_type in ["band", "hl"]:
        model.add(Dense(2))

    if y_type == "2_mods":
        model.add(Dense(1))

    model.summary()

    return model


def get_optimiser(learning_rate: float):
    return keras.optimizers.legacy.Adam(learning_rate=learning_rate)


ERROR_AMPLIFICATION_FACTOR = 1


def custom_loss_2_mods_high(y_true, y_pred):
    error = y_true - y_pred
    negative_error = K.maximum(-error, 0)
    positive_error = K.maximum(error, 0)
    return K.sqrt(K.mean(K.square(error) + positive_error * ERROR_AMPLIFICATION_FACTOR))


def custom_loss_2_mods_low(y_true, y_pred):
    error = y_true - y_pred
    negative_error = K.maximum(-error, 0)
    positive_error = K.maximum(error, 0)
    return K.sqrt(K.mean(K.square(error) + negative_error * ERROR_AMPLIFICATION_FACTOR))


def custom_loss_band(y_true, y_pred):
    error = y_true - y_pred
    error_l = y_true[..., 0] - y_pred[..., 0]
    error_h = y_true[..., 1] - y_pred[..., 1]

    positive_error_h = K.maximum(error_h, 0)
    negative_error_l = K.maximum(-error_l, 0)

    squared_error = K.square(error)
    error_amplified = (positive_error_h + negative_error_l) * ERROR_AMPLIFICATION_FACTOR
    error_amplified = K.expand_dims(error_amplified, axis=-1)  # Reshape error_amplified

    return K.sqrt(K.mean(squared_error + error_amplified))


def metric_msr(y_true, y_pred):
    error = y_true - y_pred
    return K.sqrt(K.mean(K.square(error)))


class LossDifferenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(LossDifferenceCallback, self).__init__()
        self.previous_loss = None
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_train_begin(self, logs=None):
        self.loss_difference_values = []

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if self.previous_loss is not None:
            loss_difference = current_loss - self.previous_loss
            self.loss_difference_values.append(loss_difference)
            self.update_tensorboard(epoch, loss_difference)

        self.previous_loss = current_loss

    def update_tensorboard(self, epoch, loss_difference):
        with self.writer.as_default():
            tf.summary.scalar("Loss Difference", loss_difference, step=epoch)
