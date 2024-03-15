import tensorflow as tf
from band_4.training_yf_band_4 import SAFETY_FACTOR, SKIP_FIRST_PERCENTILE, SKIP_LAST_PERCENTILE
from keras import backend as K
from keras_model import metric_rmse, rmse_average

SKIP_PERCENTILE: float = 0.18


def metric_new_idea(y_true, y_pred):
    return (
        metric_rmse(y_true, y_pred) * 4
        + metric_average_in(y_true, y_pred)
        + metric_loss_comp_2(y_true, y_pred)
        + (1 - metric_win_checkpoint(y_true, y_pred) / 100) / 30
    )


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, :, 0] + y_pred[:, :, 1]) / 2
    average_true = (y_true[:, :, 0] + y_true[:, :, 1]) / 2

    error = average_pred - average_true

    return rmse_average(error)


def support_idea_1(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index * 2

    min_pred_s = K.min(y_pred[:, start_index:end_index, 0], axis=1)
    max_pred_s = K.max(y_pred[:, start_index:end_index, 1], axis=1)

    min_pred = K.min(y_pred[:, :, 0], axis=1)
    max_pred = K.max(y_pred[:, :, 1], axis=1)

    min_true = K.min(y_true[:, :, 0], axis=1)
    max_true = K.max(y_true[:, :, 1], axis=1)

    wins = K.all(
        [
            max_true >= max_pred_s,
            max_pred_s >= min_pred_s,  # valid_pred
            min_pred_s >= min_true,
        ],
        axis=0,
    )

    return min_pred, max_pred, min_true, max_true, wins


def support_idea_1_new(y_true, y_pred):
    # create new y_pred with .80 as safety factor from average along axis=1

    y_pred_safety_min = (y_pred[:, :, 1] + y_pred[:, :, 0]) / 2 - (
        (y_pred[:, :, 1] - y_pred[:, :, 0]) / 2 * SAFETY_FACTOR
    )
    y_pred_safety_max = (y_pred[:, :, 0] + y_pred[:, :, 1]) / 2 + (
        (y_pred[:, :, 1] - y_pred[:, :, 0]) / 2 * SAFETY_FACTOR
    )

    tf.tensor_scatter_nd_update

    n: int = y_pred.shape[1]
    start_index: int = int(n * SKIP_FIRST_PERCENTILE)
    end_index: int = n - int(n * SKIP_FIRST_PERCENTILE) - int(n * SKIP_LAST_PERCENTILE)

    min_pred_s = K.min(y_pred_safety_min[:, start_index:end_index], axis=1)
    max_pred_s = K.max(y_pred_safety_max[:, start_index:end_index], axis=1)

    min_true = K.min(y_true[:, :, 0], axis=1)
    max_true = K.max(y_true[:, :, 1], axis=1)

    band_inside = K.all(
        [
            max_true >= max_pred_s,
            max_pred_s >= min_pred_s,  # valid_pred
            min_pred_s >= min_true,
        ],
        axis=0,
    )

    return min_pred_s, max_pred_s, min_true, max_true, band_inside


def support_idea_2(min_pred, max_pred, min_true, max_true, wins):
    z_1 = K.mean((1 - K.cast(K.all([max_true >= max_pred], axis=0), dtype=K.floatx())) * K.abs(max_true - max_pred))

    z_2 = K.mean((1 - K.cast(K.all([max_pred >= min_pred], axis=0), dtype=K.floatx())) * K.abs(max_pred - min_pred))

    z_3 = K.mean((1 - K.cast(K.all([min_pred >= min_true], axis=0), dtype=K.floatx())) * K.abs(min_pred - min_true))

    win_amt_true = K.sum((1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true))

    return z_1, z_2, z_3, win_amt_true


def support_idea_3(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index

    min_pred_index = K.argmin(y_pred[:, start_index:end_index, 0], axis=1)
    max_pred_index = K.argmax(y_pred[:, start_index:end_index, 1], axis=1)

    min_true_index = K.argmin(y_true[:, :, 0], axis=1)
    max_true_index = K.argmax(y_true[:, :, 1], axis=1)

    correct_trends_buy = K.all(
        [
            max_pred_index > min_pred_index,
            max_true_index > min_true_index,
        ],
        axis=0,
    )

    correct_trends_sell = K.all(
        [
            max_pred_index < min_pred_index,
            max_true_index < min_true_index,
        ],
        axis=0,
    )

    correct_trends = tf.logical_or(correct_trends_buy, correct_trends_sell)

    return correct_trends


def metric_loss_comp_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    z_1 = K.mean((1 - K.cast(K.all([max_true >= max_pred], axis=0), dtype=K.floatx())) * K.abs(max_true - max_pred))

    z_2 = K.mean((1 - K.cast(K.all([max_pred >= min_pred], axis=0), dtype=K.floatx())) * K.abs(max_pred - min_pred))

    z_3 = K.mean((1 - K.cast(K.all([min_pred >= min_true], axis=0), dtype=K.floatx())) * K.abs(min_pred - min_true))

    win_amt_true_error = K.mean((1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true))

    win_amt_pred_error = K.mean(
        K.abs(max_true - min_true) - (K.cast(wins, dtype=K.floatx()) * K.abs(max_pred - min_pred)),
    )

    return z_1 + z_2 + z_3 + win_amt_true_error + win_amt_pred_error

    # correct_trends = support_idea_3(y_true, y_pred)

    # trend_error = K.mean((1 - K.cast(correct_trends, dtype=K.floatx())) * K.abs(max_true - min_true))

    # trend_error_win = K.mean(
    #     (1 - K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx())) * K.abs(max_true - min_true),
    # )

    # trend_error_win_pred = K.mean(
    #     (1 - K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx())) * K.abs(max_pred - min_pred),
    # )

    # return (
    #     (z_1 + z_2 + z_3) * 3
    #     + (trend_error + trend_error_win + trend_error_win_pred * 2 + win_amt_true_error + win_amt_pred_error * 3)


def metric_win_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_win_pred_capture_possible = K.sum(
        (max_true / min_true - 1) * K.cast(wins, dtype=K.floatx()),
    )

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum(max_true / min_true - 1)

    return pred_capture / total_capture_possible * 100


def metric_win_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    win_fraction = K.mean(K.cast(wins, dtype=K.floatx()))

    return win_fraction * 100


def metric_correct_win_trend_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1(y_true, y_pred)

    correct_trends = support_idea_3(y_true, y_pred)

    correct_win_trend = K.mean(K.cast(correct_trends, dtype=K.floatx()) * K.cast(wins, dtype=K.floatx())) / K.mean(
        K.cast(wins, dtype=K.floatx()),
    )

    metric = tf.where(tf.math.is_nan(correct_win_trend), tf.zeros_like(correct_win_trend), correct_win_trend)

    return metric * 100


def metric_win_checkpoint(y_true, y_pred):
    n: int = y_pred.shape[1]
    start_index: int = int(n * SKIP_FIRST_PERCENTILE)
    end_index: int = n - int(n * SKIP_FIRST_PERCENTILE) - int(n * SKIP_LAST_PERCENTILE)

    min_pred_s, max_pred_s, min_true, max_true, band_inside = support_idea_1_new(y_true, y_pred)

    # getting trends.

    min_pred_index = K.argmin(y_pred[:, start_index:end_index, 0], axis=1)
    max_pred_index = K.argmax(y_pred[:, start_index:end_index, 1], axis=1)

    min_true_index = K.argmin(y_true[:, :, 0], axis=1)
    max_true_index = K.argmax(y_true[:, :, 1], axis=1)

    correct_trends_buy = K.all(
        [
            max_pred_index > min_pred_index,
            max_true_index > min_true_index,
        ],
        axis=0,
    )

    correct_trends_sell = K.all(
        [
            max_pred_index < min_pred_index,
            max_true_index < min_true_index,
        ],
        axis=0,
    )

    correct_trends = tf.logical_or(correct_trends_buy, correct_trends_sell)

    pred_capture = K.sum(
        (max_pred_s / min_pred_s - 1)
        * K.cast(band_inside, dtype=K.floatx())
        * K.cast(correct_trends, dtype=K.floatx()),
    )

    total_capture_possible = K.sum(max_true / min_true - 1)

    return pred_capture / total_capture_possible * 100
