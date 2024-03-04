import tensorflow as tf
from keras_model import metric_rmse, weighted_average

SKIP_PERCENTILE: float = 0.18


def metric_new_idea(y_true, y_pred):
    return metric_rmse(y_true, y_pred) * 2 + metric_average_in(y_true, y_pred) + metric_loss_comp_2(y_true, y_pred)


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, :, 0] + y_pred[:, :, 1]) / 2
    average_true = (y_true[:, :, 0] + y_true[:, :, 1]) / 2

    error = average_pred - average_true

    return weighted_average(error)


def support_new_idea_1(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index

    min_pred_s = tf.reduce_min(y_pred[:, start_index:end_index, 0], axis=1)
    max_pred_s = tf.reduce_max(y_pred[:, start_index:end_index, 1], axis=1)

    min_pred = tf.reduce_min(y_pred[:, :, 0], axis=1)
    max_pred = tf.reduce_max(y_pred[:, :, 1], axis=1)

    min_true = tf.reduce_min(y_true[:, :, 0], axis=1)
    max_true = tf.reduce_max(y_true[:, :, 1], axis=1)

    wins = tf.reduce_all(
        [
            tf.math.greater_equal(max_true, max_pred_s),
            tf.math.greater_equal(max_pred_s, min_pred_s),  # valid_pred
            tf.math.greater_equal(min_pred_s, min_true),
        ],
        axis=0,
    )

    return min_pred, max_pred, min_true, max_true, wins


def support_new_idea_2(min_pred, max_pred, min_true, max_true, wins):
    z_1 = tf.reduce_mean(
        (1 - tf.cast(tf.math.greater_equal(max_true, max_pred), dtype=tf.float32)) * tf.abs(max_true - max_pred),
    )

    z_2 = tf.reduce_mean(
        (1 - tf.cast(tf.math.greater_equal(max_pred, min_pred), dtype=tf.float32)) * tf.abs(max_pred - min_pred),
    )

    z_3 = tf.reduce_mean(
        (1 - tf.cast(tf.math.greater_equal(min_pred, min_true), dtype=tf.float32)) * tf.abs(min_pred - min_true),
    )

    win_amt_true = tf.reduce_sum((1 - tf.cast(wins, dtype=tf.float32)) * tf.abs(max_true - min_true))

    return z_1, z_2, z_3, win_amt_true


def support_new_idea_3(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index

    min_pred_index = tf.argmin(y_pred[:, start_index:end_index, 0], axis=1)
    max_pred_index = tf.argmax(y_pred[:, start_index:end_index, 1], axis=1)

    min_true_index = tf.argmin(y_true[:, :, 0], axis=1)
    max_true_index = tf.argmax(y_true[:, :, 1], axis=1)

    correct_trends_buy = tf.reduce_all(
        [
            tf.greater(max_pred_index, min_pred_index),
            tf.greater(max_true_index, min_true_index),
        ],
        axis=0,
    )

    correct_trends_sell = tf.reduce_all(
        [
            tf.less(max_pred_index, min_pred_index),
            tf.less(max_true_index, min_true_index),
        ],
        axis=0,
    )

    correct_trends = tf.logical_or(correct_trends_buy, correct_trends_sell)

    return correct_trends


def metric_loss_comp_2(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    z_1 = tf.reduce_mean(
        (1 - tf.cast(tf.reduce_all([tf.math.greater_equal(max_true, max_pred)], axis=0), dtype=tf.float32))
        * tf.abs(max_true - max_pred),
    )

    z_2 = tf.reduce_mean(
        (1 - tf.cast(tf.reduce_all([tf.math.greater_equal(max_pred, min_pred)], axis=0), dtype=tf.float32))
        * tf.abs(max_pred - min_pred),
    )

    z_3 = tf.reduce_mean(
        (1 - tf.cast(tf.reduce_all([tf.math.greater_equal(min_pred, min_true)], axis=0), dtype=tf.float32))
        * tf.abs(min_pred - min_true),
    )

    win_amt_true_error = tf.reduce_mean((1 - tf.cast(wins, dtype=tf.float32)) * tf.abs(max_true - min_true))

    win_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - (tf.cast(wins, dtype=tf.float32) * tf.abs(max_pred - min_pred)),
    )

    correct_trends = support_new_idea_3(y_true, y_pred)

    trend_error = tf.reduce_mean((1 - tf.cast(correct_trends, dtype=tf.float32)) * tf.abs(max_true - min_true))

    trend_error_win = tf.reduce_mean(
        (1 - tf.cast(wins, dtype=tf.float32) * tf.cast(correct_trends, dtype=tf.float32))
        * tf.abs(max_true - min_true),
    )

    trend_error_win_pred = tf.reduce_mean(
        (1 - tf.cast(wins, dtype=tf.float32) * tf.cast(correct_trends, dtype=tf.float32))
        * tf.abs(max_pred - min_pred),
    )

    return (
        (z_1 + z_2 + z_3) * 2
        + (trend_error + trend_error_win + trend_error_win_pred * 2 + win_amt_true_error + win_amt_pred_error)
    ) / 1.5


def metric_win_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    pred_capture = tf.reduce_sum((max_pred / min_pred - 1) * tf.cast(wins, dtype=tf.float32))

    total_win_pred_capture_possible = tf.reduce_sum((max_true / min_true - 1) * tf.cast(wins, dtype=tf.float32))

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    pred_capture = tf.reduce_sum((max_pred / min_pred - 1) * tf.cast(wins, dtype=tf.float32))

    total_capture_possible = tf.reduce_sum(max_true / min_true - 1)

    return pred_capture / total_capture_possible * 100


def metric_win_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    win_fraction = tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    return win_fraction * 100


def metric_correct_win_trend_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_new_idea_1(y_true, y_pred)

    correct_trends = support_new_idea_3(y_true, y_pred)

    correct_win_trend = tf.reduce_mean(
        tf.cast(correct_trends, dtype=tf.float32) * tf.cast(wins, dtype=tf.float32),
    ) / tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    metric = tf.where(
        tf.math.is_nan(correct_win_trend),
        tf.zeros_like(correct_win_trend),
        correct_win_trend,
    )

    return metric * 100


def metric_win_checkpoint(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index

    min_pred_s = tf.reduce_min(y_pred[:, start_index:end_index, 0], axis=1)
    max_pred_s = tf.reduce_max(y_pred[:, start_index:end_index, 1], axis=1)

    min_true = tf.reduce_min(y_true[:, :, 0], axis=1)
    max_true = tf.reduce_max(y_true[:, :, 1], axis=1)

    wins = tf.reduce_all(
        [
            tf.math.greater_equal(max_true, max_pred_s),
            tf.math.greater_equal(max_pred_s, min_pred_s),  # valid_pred
            tf.math.greater_equal(min_pred_s, min_true),
        ],
        axis=0,
    )

    correct_trends = support_new_idea_3(y_true, y_pred)

    trend_error_win_pred = tf.reduce_mean(
        tf.cast(wins, dtype=tf.float32) * tf.cast(correct_trends, dtype=tf.float32) * tf.abs(max_pred_s - min_pred_s),
    ) / tf.reduce_mean(tf.abs(max_true - min_true))

    return trend_error_win_pred * 100
