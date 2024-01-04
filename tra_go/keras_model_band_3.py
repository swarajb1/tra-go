import tensorflow as tf
from keras import backend as K
from keras_model import metric_rmse, weighted_average


def metric_band_hl_correction(y_true, y_pred):
    # hl band height cannot be negative
    hl_band_height_array = y_pred[..., 1] - y_pred[..., 0]

    error_1 = K.maximum(-hl_band_height_array, 0)

    error_hl_correction = weighted_average(error_1)

    return error_hl_correction


def metric_band_hl_correction_percent(y_true, y_pred):
    mean_hl_height = K.mean(y_true[..., 1] - y_true[..., 0])

    return metric_band_hl_correction(y_true, y_pred) / mean_hl_height * 100


def metric_band_hl_wrongs_percent(y_true, y_pred):
    hl_band_height_array = y_pred[..., 1] - y_pred[..., 0]

    negative_hl_count = tf.reduce_sum(
        tf.cast(tf.less(hl_band_height_array, 0), tf.float32),
    )

    total_count = K.cast(K.shape(hl_band_height_array)[1], dtype=K.floatx()) * K.cast(
        K.shape(hl_band_height_array)[0],
        dtype=K.floatx(),
    )

    return negative_hl_count / total_count * 100


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, :, 0] + y_pred[:, :, 1]) / 2
    average_true = (y_true[:, :, 0] + y_true[:, :, 1]) / 2

    error = average_pred - average_true

    return weighted_average(error)


def support_new_idea_1(y_true, y_pred):
    min_pred = K.min(y_pred[:, :, 0], axis=1)
    max_pred = K.max(y_pred[:, :, 1], axis=1)

    min_true = K.min(y_true[:, :, 0], axis=1)
    max_true = K.max(y_true[:, :, 1], axis=1)

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


def metric_new_idea_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    z_1, z_2, z_3, win_amt_true = support_new_idea_2(min_pred, max_pred, min_true, max_true, wins)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum(max_true / min_true - 1)

    pred_capture_fraction = pred_capture / total_capture_possible

    loss_amt = (
        metric_rmse(y_true, y_pred) * 3 + metric_all_candle_in(y_true, y_pred) + metric_average_in(y_true, y_pred) * 2
    )

    loss_comp_1 = z_1 + z_2 + z_3 + win_amt_true * 5 + (1 - pred_capture_fraction) * K.mean(max_true - min_true) * 10

    return loss_amt + loss_comp_1


def metric_all_candle_in(y_true, y_pred):
    min_true = y_true[..., 0]
    max_true = y_true[..., 1]

    error = (
        K.mean(
            (1 - K.cast(K.all([max_true >= y_pred[..., 0]], axis=0), dtype=K.floatx()))
            * K.abs(max_true - y_pred[..., 0]),
        )
        + K.mean(
            (1 - K.cast(K.all([max_true >= y_pred[..., 1]], axis=0), dtype=K.floatx()))
            * K.abs(max_true - y_pred[..., 1]),
        )
        + K.mean(
            (1 - K.cast(K.all([max_true >= y_pred[..., 2]], axis=0), dtype=K.floatx()))
            * K.abs(max_true - y_pred[..., 2]),
        )
        + K.mean(
            (1 - K.cast(K.all([max_true >= y_pred[..., 3]], axis=0), dtype=K.floatx()))
            * K.abs(max_true - y_pred[..., 3]),
        )
    ) + (
        K.mean(
            (1 - K.cast(K.all([min_true <= y_pred[..., 0]], axis=0), dtype=K.floatx()))
            * K.abs(min_true - y_pred[..., 0]),
        )
        + K.mean(
            (1 - K.cast(K.all([min_true <= y_pred[..., 1]], axis=0), dtype=K.floatx()))
            * K.abs(min_true - y_pred[..., 1]),
        )
        + K.mean(
            (1 - K.cast(K.all([min_true <= y_pred[..., 2]], axis=0), dtype=K.floatx()))
            * K.abs(min_true - y_pred[..., 2]),
        )
        + K.mean(
            (1 - K.cast(K.all([min_true <= y_pred[..., 3]], axis=0), dtype=K.floatx()))
            * K.abs(min_true - y_pred[..., 3]),
        )
    )

    return error


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
