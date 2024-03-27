import keras
import keras.backend as K
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from keras.regularizers import L1

NUMBER_OF_NEURONS: int = 512
NUMBER_OF_LAYERS: int = 6
INITIAL_DROPOUT_PERCENT: float = 0.01


def get_untrained_model(X_train, y_type):
    model = keras.Sequential()

    model.add(
        LSTM(
            units=NUMBER_OF_NEURONS,
            input_shape=(X_train[0].shape),
            return_sequences=True,
            activation="relu",
        ),
    )

    model.add(Dropout(INITIAL_DROPOUT_PERCENT / 100))

    for i in range(NUMBER_OF_LAYERS - 1):
        model.add(
            LSTM(
                units=NUMBER_OF_NEURONS,
                return_sequences=True,
                activation="relu",
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(Dropout(pow(INITIAL_DROPOUT_PERCENT, 1 / (i + 2)) / 100))

    if y_type == "band_2":
        model.add(Dense(2))

    elif y_type == "band_4":
        model.add(Dense(4))

    model.summary()
    print("\n" * 2)

    return model


def get_untrained_model_new(X_train):
    model = keras.models.Sequential()

    model.add(Input(shape=(X_train[0].shape)))

    for i in range(NUMBER_OF_LAYERS):
        model.add(
            Bidirectional(
                LSTM(
                    units=NUMBER_OF_NEURONS,
                    return_sequences=True,
                    activation="relu",
                    kernel_regularizer=L1(0.00001),
                ),
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(Dropout(pow(1 + INITIAL_DROPOUT_PERCENT / 100, 1 / (i + 1)) - 1))

    model.add(Dense(4))

    model.summary()

    print("\n" * 2)

    return model


def get_optimiser(learning_rate: float):
    return keras.optimizers.legacy.Adam(learning_rate=learning_rate)


def metric_rmse(y_true, y_pred):
    # Calculate the root mean squared error (RMSE)

    error = y_true - y_pred

    return K.sqrt(K.mean(K.square(error)))
    # return tf.sqrt(tf.reduce_mean(tf.square(error)))


def metric_abs(y_true, y_pred):
    # Calculate the absolute mean error (MAE)

    error = y_true - y_pred

    return K.mean(K.abs(error))
    # return tf.reduce_mean(tf.abs(error))


def metric_abs_percent(y_true, y_pred):
    error = y_true - y_pred

    return K.mean(K.abs(error)) / K.mean(K.abs(y_true)) * 100
    # return tf.reduce_mean(tf.abs(error)) / tf.reduce_mean(tf.abs(y_true)) * 100


def metric_rmse_percent(y_true, y_pred):
    error = y_true - y_pred

    return K.sqrt(K.mean(K.square(error))) / K.mean(K.abs(y_true)) * 100
    # return tf.sqrt(tf.reduce_mean(tf.square(error))) / tf.reduce_mean(tf.abs(y_true)) * 100


def rmse_average(array):
    # weight average of an array with mea and rmse

    return K.sqrt(K.mean(K.square(array)))
