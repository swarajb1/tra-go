import os

import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential

load_dotenv()


NUMBER_OF_NEURONS: int = int(os.getenv("NUMBER_OF_NEURONS"))
NUMBER_OF_LAYERS: int = int(os.getenv("NUMBER_OF_LAYERS"))
INITIAL_DROPOUT_PERCENT: float = float(os.getenv("INITIAL_DROPOUT_PERCENT"))


def get_untrained_model(X_train, Y_train):
    model = Sequential()

    model.add(Input(shape=(X_train[0].shape)))

    for i in range(NUMBER_OF_LAYERS):
        model.add(
            Bidirectional(
                LSTM(
                    units=NUMBER_OF_NEURONS,
                    return_sequences=True,
                    activation="relu",
                ),
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(Dropout(pow(1 + INITIAL_DROPOUT_PERCENT / 100, 1 / (i + 1)) - 1))

    model.add(TimeDistributed(Dense(units=Y_train[0].shape[1])))

    model.summary()

    print("\n" * 2)

    return model


def get_optimiser(learning_rate: float):
    return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)


def metric_rmse(y_true, y_pred):
    # Calculate the root mean squared error (RMSE)

    error = y_true - y_pred

    return tf.sqrt(tf.reduce_mean(tf.square(error)))


def metric_abs(y_true, y_pred):
    # Calculate the absolute mean error (MAE)

    error = y_true - y_pred

    return tf.reduce_mean(tf.abs(error))


def metric_abs_percent(y_true, y_pred):
    error = y_true - y_pred

    return tf.reduce_mean(tf.abs(error)) / tf.reduce_mean(tf.abs(y_true)) * 100


def metric_rmse_percent(y_true, y_pred):
    error = y_true - y_pred

    return tf.sqrt(tf.reduce_mean(tf.square(error))) / tf.reduce_mean(tf.abs(y_true)) * 100
