import keras
import keras.backend as K
from core.config import settings
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, TimeDistributed


def get_untrained_model(X_train, Y_train):
    model = keras.models.Sequential()

    model.add(Input(shape=(X_train[0].shape)))

    for i in range(settings.NUMBER_OF_LAYERS):
        model.add(
            Bidirectional(
                LSTM(
                    units=settings.NUMBER_OF_NEURONS,
                    return_sequences=True,
                    activation="relu",
                ),
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(Dropout(pow(1 + settings.INITIAL_DROPOUT_PERCENT / 100, 1 / (i + 1)) - 1))

    model.add(TimeDistributed(Dense(units=Y_train[0].shape[1])))

    # model.add(Dense(units=Y_train[0].shape[1]))

    model.summary()

    print("\n" * 2)

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
