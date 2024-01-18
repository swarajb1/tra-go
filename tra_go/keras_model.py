from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from tensorflow import keras

# PYTHONPATH = /Users/bisane.s/my_files/my_codes/tra-go/.venv/bin/python

# keep total neurons below 2700 (900 * 3)

NUMBER_OF_NEURONS = 900
NUMBER_OF_LAYERS = 3
INITIAL_DROPOUT = 0

WEIGHT_FOR_MEA = 0


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
    model.add(Dropout(INITIAL_DROPOUT / 100))

    for i in range(NUMBER_OF_LAYERS - 1):
        model.add(
            LSTM(
                units=NUMBER_OF_NEURONS,
                return_sequences=True,
                activation="relu",
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(Dropout(pow(INITIAL_DROPOUT, 1 / (i + 2)) / 100))

    if y_type == "band_2":
        model.add(Dense(2))

    elif y_type == "band_4" or y_type == "band_5":
        model.add(Dense(4))

    model.summary()

    return model


def get_optimiser(learning_rate: float):
    return keras.optimizers.legacy.Adam(learning_rate=learning_rate)


def metric_rmse(y_true, y_pred):
    # Calculate the root mean squared error (RMSE)

    error = y_true - y_pred

    return K.sqrt(K.mean(K.square(error)))


def metric_abs(y_true, y_pred):
    # Calculate the absolute mean error (MAE)

    error = y_true - y_pred

    return K.mean(K.abs(error))


def metric_abs_percent(y_true, y_pred):
    error = y_true - y_pred

    return K.mean(K.abs(error)) / K.mean(K.abs(y_true)) * 100


def metric_rmse_percent(y_true, y_pred):
    error = y_true - y_pred

    return K.sqrt(K.mean(K.square(error))) / K.mean(K.abs(y_true)) * 100


def weighted_average(array):
    # weight average of an array with mea and rmse
    return WEIGHT_FOR_MEA * K.mean(K.abs(array)) + (1 - WEIGHT_FOR_MEA) * K.sqrt(K.mean(K.square(array)))
