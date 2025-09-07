import band_2_1.model_metrics as km_21_metrics
import keras_model_tf as km_tf
import tensorflow as tf
from core.config import settings
from numpy.typing import NDArray
from tensorflow.keras.layers import (
    LSTM,
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
    def __init__(self):
        self.learning_rate: float = 0.001
        self.optimizer = km_tf.get_optimiser(self.learning_rate)

        self.loss = km_21_metrics.loss_function

        self.metrics = [
            # km_tf.metric_rmse_percent,
            # km_tf.metric_abs_percent,
            km_21_metrics.metric_rmse_percent,
            km_21_metrics.metric_abs_percent,
            km_21_metrics.metric_correct_trends_full,
            km_21_metrics.metric_loss_comp_2,
            km_21_metrics.metric_win_percent,
            km_21_metrics.metric_win_pred_capture_percent,
            km_21_metrics.metric_win_pred_capture_total_percent,
            km_21_metrics.metric_win_correct_trend_percent,
            km_21_metrics.metric_win_pred_trend_capture_percent,
            km_21_metrics.metric_try_1,
            # km_21_metrics.metric_try_2,
            km_21_metrics.stoploss_incurred,
        ]


class CustomActivationLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Apply linear activation to the first two features
        first_two_features = inputs[:, :2]

        # Apply sigmoid activation and then round to the third feature
        third_feature = tf.round(tf.sigmoid(inputs[:, 2:3]))

        # Concatenate the features back together
        return tf.concat([first_two_features, third_feature], axis=-1)


def get_untrained_model_old(X_train: NDArray, Y_train: NDArray) -> Model:
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
                    1 + settings.INITIAL_DROPOUT_PERCENT / 100,
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

    compile_details = ModelCompileDetails()

    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    model.summary()

    print("\n" * 2)

    return model


def get_untrained_model(X_train: NDArray, Y_train: NDArray) -> Model:
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
                    recurrent_dropout=0.0,
                    unroll=False,
                ),
            ),
        )
        #  dropout value decreases in exponential fashion.
        model.add(
            Dropout(
                pow(
                    1 + settings.INITIAL_DROPOUT_PERCENT / 100,
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

    compile_details = ModelCompileDetails()

    model.compile(
        optimizer=compile_details.optimizer,
        loss=compile_details.loss,
        metrics=compile_details.metrics,
    )

    model.summary()

    print("\n" * 2)

    return model
