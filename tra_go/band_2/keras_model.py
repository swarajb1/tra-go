import tensorflow as tf
from core.config import settings
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    TimeDistributed,
)

# RISK_TO_REWARD_RATIO: float = 0.2
SKIP_FIRST_PERCENTILE: float = 0.15
SKIP_LAST_PERCENTILE: float = 0.15


def get_untrained_model(X_train, Y_train):
    model = tf.keras.models.Sequential()

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
        model.add(Dropout(pow(1 + settings.INITIAL_DROPOUT, 1 / (i + 1)) - 1))

    model.add(TimeDistributed(Dense(units=Y_train[0].shape[1])))

    # model.add(Dense(units=Y_train[0].shape[1]))

    model.summary()

    print("\n" * 2)

    return model
