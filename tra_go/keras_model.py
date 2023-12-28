import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from tensorflow import keras

# PYTHONPATH = /Users/bisane.s/my_files/my_codes/tra-go/.venv/bin/python

# keep total neurons below 2700 (900 * 3)

NUMBER_OF_NEURONS = 512
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

    model.add(Dense(2))

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


def weighted_average(array):
    # weight average of an array with mea and rmse
    return WEIGHT_FOR_MEA * K.mean(K.abs(array)) + (1 - WEIGHT_FOR_MEA) * K.sqrt(
        K.mean(K.square(array)),
    )


def metric_band_base_percent(y_true, y_pred):
    error_avg = y_true[..., 0] - y_pred[..., 0]

    error_height = y_true[..., 1] - y_pred[..., 1]

    error_avg_mean = K.mean(K.abs(error_avg))

    error_height_mean = K.mean(K.abs(error_height))

    return (
        (error_avg_mean + error_height_mean / 2)
        / (K.mean(y_true[..., 0]) + K.mean(y_true[..., 1]) / 2)
    ) * 100


def metric_band_hl_correction(y_true, y_pred):
    # band height cannot be negative
    band_height_array = y_pred[..., 1]

    error_1 = K.maximum(-band_height_array, 0)

    error_hl_correction = weighted_average(error_1)

    return error_hl_correction


def metric_band_hl_correction_percent(y_true, y_pred):
    # band height cannot be negative
    band_height_array = y_pred[..., 1]

    error_1 = K.maximum(-band_height_array, 0)

    error_hl_correction = weighted_average(error_1)

    return error_hl_correction / K.mean(y_true[..., 1]) * 100


def metric_band_hl_wrongs_percent(y_true, y_pred):
    band_height_array = y_pred[..., 1]

    negative_hl_count = tf.reduce_sum(
        tf.cast(tf.less(band_height_array, 0), tf.float32),
    )

    total_count = K.cast(K.shape(band_height_array)[1], dtype=K.floatx()) * K.cast(
        K.shape(band_height_array)[0],
        dtype=K.floatx(),
    )

    return negative_hl_count / total_count * 100


def metric_band_height(y_true, y_pred):
    # band height to approach true band height
    error = y_true[..., 1] - y_pred[..., 1]

    error_band_height = weighted_average(error)

    return error_band_height


def metric_band_height_percent(y_true, y_pred):
    return metric_band_height(y_true, y_pred) / K.mean(y_true[..., 1]) * 100


def metric_band_average(y_true, y_pred):
    # average should approach true average
    error = y_true[..., 0] - y_pred[..., 0]

    error_avg = weighted_average(error)

    return error_avg


def metric_band_average_percent(y_true, y_pred):
    return metric_band_average(y_true, y_pred) / K.mean(y_true[..., 0]) * 100


def metric_band_avg_correction(y_true, y_pred):
    # average to inside the band.

    error_1 = (y_true[..., 0] + y_true[..., 1] / 2) - y_pred[..., 0]
    error_2 = (y_true[..., 0] - y_true[..., 1] / 2) - y_pred[..., 0]

    error_1_1 = K.maximum(-error_1, 0)
    error_2_2 = K.maximum(error_2, 0)

    error_avg_correction = weighted_average(error_1_1) + weighted_average(error_2_2)

    return error_avg_correction


def metric_band_avg_correction_percent(y_true, y_pred):
    # average to inside the band.

    error_1 = (y_true[..., 0] + y_true[..., 1] / 2) - y_pred[..., 0]
    error_2 = (y_true[..., 0] - y_true[..., 1] / 2) - y_pred[..., 0]

    error_1_1 = K.maximum(-error_1, 0)
    error_2_2 = K.maximum(error_2, 0)

    error_avg_correction = weighted_average(error_1_1) + weighted_average(error_2_2)

    return error_avg_correction / K.mean(y_true[..., 0]) * 100


def support_new_idea_1(y_true, y_pred):
    min_pred = K.min(y_pred[:, :, 0] - y_pred[:, :, 1] / 2, axis=1)
    max_pred = K.max(y_pred[:, :, 0] + y_pred[:, :, 1] / 2, axis=1)

    min_true = K.min(y_true[:, :, 0] - y_true[:, :, 1] / 2, axis=1)
    max_true = K.max(y_true[:, :, 0] + y_true[:, :, 1] / 2, axis=1)

    wins = K.all(
        [max_true >= max_pred, max_pred >= min_pred, min_pred >= min_true],
        axis=0,
    )

    return min_pred, max_pred, min_true, max_true, wins


def support_new_idea_2(min_pred, max_pred, min_true, max_true, wins):
    cond_1 = K.all([max_true >= max_pred], axis=0)
    cond_2 = K.all([max_pred >= min_pred], axis=0)
    cond_3 = K.all([min_pred >= min_true], axis=0)

    z_1 = K.mean((1 - K.cast(cond_1, dtype=K.floatx())) * K.abs(max_true - max_pred))

    z_2 = K.mean((1 - K.cast(cond_2, dtype=K.floatx())) * K.abs(max_pred - min_pred))

    z_3 = K.mean((1 - K.cast(cond_3, dtype=K.floatx())) * K.abs(min_pred - min_true))

    win_amt_true = K.mean(
        (1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_pred - min_pred),
    )

    return z_1, z_2, z_3, win_amt_true


def metric_new_idea_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    z_1, z_2, z_3, win_amt_true = support_new_idea_2(
        min_pred,
        max_pred,
        min_true,
        max_true,
        wins,
    )

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum(max_true / min_true - 1)

    pred_capture_fraction = pred_capture / total_capture_possible

    loss_amt = (
        metric_band_average(y_true, y_pred)
        + metric_band_avg_correction(y_true, y_pred) * 50
        + metric_band_height(y_true, y_pred) * 100
        + metric_band_hl_correction(y_true, y_pred) * 100
    )

    loss_percent = (
        metric_band_average_percent(y_true, y_pred)
        + metric_band_avg_correction_percent(y_true, y_pred) * 10
        + metric_band_height_percent(y_true, y_pred)
        + metric_band_hl_correction_percent(y_true, y_pred)
    )

    loss_comp_1 = (
        z_1
        + z_2
        + z_3
        + win_amt_true * 5
        + (1 - pred_capture_fraction) * K.mean(max_true - min_true) * 10
    )

    return loss_amt + loss_percent / 100 + loss_comp_1


def metric_new_idea_2_good(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    z_1, z_2, z_3, win_amt_true = support_new_idea_2(
        min_pred,
        max_pred,
        min_true,
        max_true,
        wins,
    )

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum(max_true / min_true - 1)

    pred_capture_fraction = pred_capture / total_capture_possible

    loss_amt = (
        metric_band_average(y_true, y_pred) * 5
        + (
            metric_band_height(y_true, y_pred)
            + metric_band_hl_correction(y_true, y_pred) * 3
        )
        * 100
    )

    loss_percent = (
        metric_band_average_percent(y_true, y_pred) * 5
        + metric_band_height_percent(y_true, y_pred)
        + metric_band_hl_correction_percent(y_true, y_pred)
    )

    loss_comp_1 = (
        z_1
        + z_2
        + z_3
        + win_amt_true * 5
        + (1 - pred_capture_fraction) * K.mean(max_true - min_true) * 10
    )

    return loss_amt + loss_comp_1 + loss_percent / 100


def metric_loss_comp_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    cond_1 = K.all([max_true >= max_pred], axis=0)
    cond_2 = K.all([max_pred >= min_pred], axis=0)
    cond_3 = K.all([min_pred >= min_true], axis=0)

    z_1 = K.mean(1 - K.cast(cond_1, dtype=K.floatx()))

    z_2 = K.mean(1 - K.cast(cond_2, dtype=K.floatx()))

    z_3 = K.mean(1 - K.cast(cond_3, dtype=K.floatx()))

    win_true = K.mean(1 - K.cast(wins, dtype=K.floatx()))

    return z_1 + z_2 + z_3 + win_true


def metric_win_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_win_pred_capture_possible = K.sum(
        (max_true / min_true - 1) * K.cast(wins, dtype=K.floatx()),
    )

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum(max_true / min_true - 1)

    return pred_capture / total_capture_possible * 100


def metric_win_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    win_fraction = K.mean(K.cast(wins, dtype=K.floatx()))

    return win_fraction * 100
