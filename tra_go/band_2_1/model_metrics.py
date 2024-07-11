import tensorflow as tf
from keras_model import metric_abs, metric_rmse

SKIP_FIRST_PERCENTILE: float = 0.15
SKIP_LAST_PERCENTILE: float = 0.15


def loss_function(y_true, y_pred):
    return (
        metric_rmse(y_true, y_pred)
        + metric_abs(y_true, y_pred) / 3
        + metric_average_in(y_true, y_pred) / 3
        # + metric_loss_comp_2(y_true, y_pred)
    )


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, 0] + y_pred[:, 1]) / 2
    average_true = (y_true[:, 0] + y_true[:, 1]) / 2

    return (
        metric_abs(average_true, average_pred)
        + metric_abs(y_true[:, 0], y_pred[:, 0])
        + metric_abs(y_true[:, 1], y_pred[:, 1])
    )


def get_band_inside(y_true, y_pred) -> tf.Tensor:
    is_valid_pred: tf.Tensor = tf.math.greater_equal(y_pred[:, 1], y_pred[:, 0])

    is_max_pred_less_than_max_true: tf.Tensor = tf.math.less_equal(y_pred[:, 1], y_true[:, 1])

    is_min_pred_more_than_min_true: tf.Tensor = tf.math.greater_equal(y_pred[:, 0], y_true[:, 0])

    band_inside: tf.Tensor = is_valid_pred & is_max_pred_less_than_max_true & is_min_pred_more_than_min_true

    return band_inside


def get_correct_trends(y_true, y_pred) -> tf.Tensor:
    correct_trends = tf.equal(y_pred[:, 2], y_true[:, 2])

    return correct_trends


def metric_loss_comp_2(y_true, y_pred) -> tf.Tensor:
    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    is_valid_pred: tf.Tensor = tf.math.greater_equal(y_pred[:, 1], y_pred[:, 0])

    is_max_pred_less_than_max_true: tf.Tensor = tf.math.less_equal(y_pred[:, 1], y_true[:, 1])

    is_min_pred_more_than_min_true: tf.Tensor = tf.math.greater_equal(y_pred[:, 0], y_true[:, 0])

    band_inside: tf.Tensor = is_valid_pred & is_max_pred_less_than_max_true & is_min_pred_more_than_min_true

    z_max_above_error = tf.reduce_mean(
        (1 - tf.cast(is_max_pred_less_than_max_true, dtype=tf.float32)) * tf.abs(max_true - max_pred),
    )

    z_pred_valid_error = tf.reduce_mean(
        (1 - tf.cast(is_valid_pred, dtype=tf.float32)) * tf.abs(max_pred - min_pred),
    )

    z_min_below_error = tf.reduce_mean(
        (1 - tf.cast(is_min_pred_more_than_min_true, dtype=tf.float32)) * tf.abs(min_pred - min_true),
    )

    win_amt_true_error = tf.reduce_mean((1 - tf.cast(band_inside, dtype=tf.float32)) * tf.abs(max_true - min_true))

    win_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - (tf.cast(band_inside, dtype=tf.float32) * tf.abs(max_pred - min_pred)),
    )

    correct_trends = get_correct_trends(y_true, y_pred)

    trend_error_win = tf.reduce_mean(
        tf.cast(band_inside, dtype=tf.float32)
        * (1 - tf.cast(correct_trends, dtype=tf.float32))
        * tf.abs(max_true - min_true),
    )

    trend_error_win_pred_error = tf.reduce_mean(
        (
            tf.cast(band_inside, dtype=tf.float32) * tf.abs(max_true - min_true)
            - (
                tf.cast(band_inside, dtype=tf.float32)
                * (1 - tf.cast(correct_trends, dtype=tf.float32))
                * tf.abs(max_pred - min_pred)
            )
        ),
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
    wins = get_band_inside(y_true, y_pred)

    win_fraction = tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    return win_fraction * 100


def metric_win_pred_capture_percent(y_true, y_pred):
    wins = get_band_inside(y_true, y_pred)

    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_win_pred_capture_possible = tf.reduce_mean(
        (max_true - min_true) * tf.cast(wins, dtype=tf.float32),
    )

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_win_correct_trend_percent(y_true, y_pred):
    wins = get_band_inside(y_true, y_pred)

    correct_trends = get_correct_trends(y_true, y_pred)

    correct_win_trend = tf.reduce_mean(
        tf.cast(correct_trends, dtype=tf.float32) * tf.cast(wins, dtype=tf.float32),
    ) / tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    metric = tf.where(
        tf.math.is_nan(correct_win_trend),
        tf.zeros_like(correct_win_trend),
        correct_win_trend,
    )

    return metric * 100


def metric_pred_capture(y_true, y_pred):
    wins = get_band_inside(y_true, y_pred)

    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))

    return total_capture_possible - pred_capture


def metric_pred_capture_percent(y_true, y_pred):
    wins = get_band_inside(y_true, y_pred)

    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))

    return pred_capture / total_capture_possible * 100


def metric_win_pred_trend_capture_percent(y_true, y_pred):
    wins = get_band_inside(y_true, y_pred)

    correct_trends = get_correct_trends(y_true, y_pred)

    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    pred_trend_capture = tf.reduce_mean(
        tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32) * tf.cast(correct_trends, dtype=tf.float32),
    )

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))

    return pred_trend_capture / total_capture_possible * 100
