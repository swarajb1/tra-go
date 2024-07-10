import tensorflow as tf
from keras import backend as K
from keras_model import metric_abs, metric_rmse

SKIP_FIRST_PERCENTILE: float = 0.15
SKIP_LAST_PERCENTILE: float = 0.15


def metric_new_idea(y_true, y_pred):
    return (
        metric_rmse(y_true, y_pred)
        + metric_abs(y_true, y_pred) / 3
        + metric_average_in(y_true, y_pred) / 3
        + metric_loss_comp_2(y_true, y_pred)
    )


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, 0] + y_pred[:, 1]) / 2
    average_true = (y_true[:, 0] + y_true[:, 1]) / 2

    return (
        metric_abs(average_true, average_pred)
        + metric_abs(y_true[:, 0], y_pred[:, 0])
        + metric_abs(y_true[:, 1], y_pred[:, 1])
    )


def get_band_inside(y_true, y_pred) -> tf.Tensor[bool]:
    is_valid_pred: tf.Tensor[bool] = tf.math.greater_equal(y_pred[:, 1], y_pred[:, 0])

    is_max_pred_less_than_max_true: tf.Tensor[bool] = tf.math.less_equal(y_pred[:, 1], y_true[:, 1])

    is_min_pred_more_than_min_true: tf.Tensor[bool] = tf.math.greater_equal(y_pred[:, 0], y_true[:, 0])

    band_inside: tf.Tensor[bool] = is_valid_pred & is_max_pred_less_than_max_true & is_min_pred_more_than_min_true

    return band_inside


def get_correct_trends(y_true, y_pred) -> tf.Tensor[bool]:
    correct_trends: tf.Tensor[bool] = y_pred[:2] == y_true[:2]

    return correct_trends


def metric_loss_comp_2(y_true, y_pred) -> tf.Tensor:
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    z_max_above_error = K.mean(
        (1 - K.cast(K.all([max_true >= max_pred], axis=0), dtype=K.floatx())) * K.abs(max_true - max_pred),
    )

    z_pred_valid_error = K.mean(
        (1 - K.cast(K.all([max_pred >= min_pred], axis=0), dtype=K.floatx())) * K.abs(max_pred - min_pred),
    )

    z_min_below_error = K.mean(
        (1 - K.cast(K.all([min_pred >= min_true], axis=0), dtype=K.floatx())) * K.abs(min_pred - min_true),
    )

    win_amt_true_error = K.mean((1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true))

    win_amt_pred_error = K.mean(
        K.abs(max_true - min_true) - (K.cast(wins, dtype=K.floatx()) * K.abs(max_pred - min_pred)),
    )

    correct_trends = get_correct_trends(y_true, y_pred)

    trend_error_win = K.mean(
        K.cast(wins, dtype=K.floatx()) * (1 - K.cast(correct_trends, dtype=K.floatx())) * K.abs(max_true - min_true),
    )

    trend_error_win_pred_error = K.mean(
        K.cast(wins, dtype=K.floatx())
        * (K.abs(max_true - min_true) - (1 - K.cast(correct_trends, dtype=K.floatx())) * K.abs(max_pred - min_pred)),
    )

    return (
        z_max_above_error
        + z_pred_valid_error
        + z_min_below_error
        + win_amt_true_error * 2
        + win_amt_pred_error * 6
        + trend_error_win * 16
        + trend_error_win_pred_error * 39
    )


def metric_win_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    win_fraction = K.mean(K.cast(wins, dtype=K.floatx()))

    return win_fraction * 100


def metric_win_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.mean(K.abs(max_pred - min_pred) * K.cast(wins, dtype=K.floatx()))

    total_win_pred_capture_possible = K.mean(
        (max_true - min_true) * K.cast(wins, dtype=K.floatx()),
    )

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_win_correct_trend_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    correct_trends = get_correct_trends(y_true, y_pred)

    correct_win_trend = K.mean(
        K.cast(correct_trends, dtype=K.floatx()) * K.cast(wins, dtype=K.floatx()),
    ) / K.mean(
        K.cast(wins, dtype=K.floatx()),
    )

    metric = tf.where(
        tf.math.is_nan(correct_win_trend),
        tf.zeros_like(correct_win_trend),
        correct_win_trend,
    )

    return metric * 100


def metric_pred_capture(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.mean(K.abs(max_pred - min_pred) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.mean(K.abs(max_true - min_true))

    return total_capture_possible - pred_capture


def metric_pred_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.mean(K.abs(max_pred - min_pred) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.mean(K.abs(max_true - min_true))

    return pred_capture / total_capture_possible * 100


def metric_pred_trend_capture_percent(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    correct_trends = get_correct_trends(y_true, y_pred)

    pred_capture = K.mean(
        K.abs(max_pred - min_pred) * K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx()),
    )

    total_capture_possible = K.mean(K.abs(max_true - min_true))

    return pred_capture / total_capture_possible * 100


def metric_win_checkpoint(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    correct_trends = get_correct_trends(y_true, y_pred)

    pred_capture = K.mean(
        K.abs(max_pred - min_pred) * K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx()),
    )

    total_capture_possible = K.mean(K.abs(max_true - min_true))

    return total_capture_possible - pred_capture
