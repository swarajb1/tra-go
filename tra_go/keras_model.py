from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM, Dropout


def get_untrained_model(X_train, y_type):
    model = keras.Sequential()

    model.add(
        LSTM(
            units=128,
            input_shape=(X_train[0].shape),
            return_sequences=True,
            activation="relu",
        )
    )
    # model.add(Dropout(0.1))

    model.add(
        LSTM(
            units=128,
            return_sequences=True,
            activation="relu",
        )
    )
    # model.add(Dropout(0.1))

    model.add(
        LSTM(
            units=128,
            return_sequences=True,
            activation="relu",
        )
    )
    # model.add(Dropout(0.1))

    if y_type == "hl":
        model.add(Flatten())

    model.add(Dense(2))

    model.summary()

    return model


def get_optimiser(learning_rate: float):
    return keras.optimizers.legacy.Adam(learning_rate=learning_rate)
