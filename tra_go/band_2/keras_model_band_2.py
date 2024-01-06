import tensorflow as tf
from keras import backend as K
from keras_model import weighted_average


def metric_band_base_percent(y_true, y_pred):
    error_avg = y_true[..., 0] - y_pred[..., 0]

    error_height = y_true[..., 1] - y_pred[..., 1]

    error_avg_mean = K.mean(K.abs(error_avg))

    error_height_mean = K.mean(K.abs(error_height))

    return ((error_avg_mean + error_height_mean / 2) / (K.mean(y_true[..., 0]) + K.mean(y_true[..., 1]) / 2)) * 100


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
        [
            max_true >= max_pred,
            max_pred >= min_pred,  # valid_pred
            min_pred >= min_true,
        ],
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

    loss_comp_1 = z_1 + z_2 + z_3 + win_amt_true * 5 + (1 - pred_capture_fraction) * K.mean(max_true - min_true) * 10

    return loss_amt + loss_percent / 100 + loss_comp_1
