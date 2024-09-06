import tensorflow as tf
from keras_model_tf import metric_abs


def loss_function(y_true, y_pred):
    return (
        # metric_rmse(y_true[:, :2], y_pred[:, :2])
        # +
        # metric_abs(y_true[:, :2], y_pred[:, :2])
        # +
        # metric_abs(y_true[:, 2:], y_pred[:, 2:]) / 20
        # +
        metric_average_in(y_true, y_pred) * 3
        + metric_loss_comp_2(y_true, y_pred)
    )


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, 0] + y_pred[:, 1]) / 2
    average_true = (y_true[:, 0] + y_true[:, 1]) / 2

    # return (
    #     metric_rmse(average_true, average_pred)
    #     + metric_rmse(y_true[:, 0], y_pred[:, 0])
    #     + metric_rmse(y_true[:, 1], y_pred[:, 1])
    # )

    return (
        metric_abs(average_true, average_pred)
        # + metric_abs(y_true[:, 0], y_pred[:, 0])
        # + metric_abs(y_true[:, 1], y_pred[:, 1])
    )


def _get_band_inside(y_true, y_pred):
    is_valid_pred = tf.math.greater_equal(y_pred[:, 1], y_pred[:, 0])

    is_max_pred_less_than_max_true = tf.math.less_equal(y_pred[:, 1], y_true[:, 1])

    is_min_pred_more_than_min_true = tf.math.greater_equal(y_pred[:, 0], y_true[:, 0])

    band_inside = is_valid_pred & is_max_pred_less_than_max_true & is_min_pred_more_than_min_true

    return band_inside


def _get_correct_trends(y_true, y_pred):
    correct_trends = tf.equal(y_pred[:, 2], y_true[:, 2])

    return correct_trends


def metric_correct_trends_full(y_true, y_pred):
    correct_trends = tf.equal(y_pred[:, 2], y_true[:, 2])

    correct_win_trend = tf.reduce_mean(tf.cast(correct_trends, dtype=tf.float32))

    return correct_win_trend * 100


def metric_loss_comp_2(y_true, y_pred):
    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    is_valid_pred = tf.math.greater_equal(y_pred[:, 1], y_pred[:, 0])

    is_max_pred_less_than_max_true = tf.math.less_equal(y_pred[:, 1], y_true[:, 1])

    is_min_pred_more_than_min_true = tf.math.greater_equal(y_pred[:, 0], y_true[:, 0])

    is_max_min_diff_more_than_00_1 = tf.math.greater_equal(y_pred[:, 1] - y_pred[:, 0], 0.01)

    band_inside = is_valid_pred & is_max_pred_less_than_max_true & is_min_pred_more_than_min_true

    z_max_above_error = tf.reduce_mean(
        (1 - tf.cast(is_max_pred_less_than_max_true, dtype=tf.float32)) * tf.abs(max_true - max_pred),
    )

    z_pred_valid_error = tf.reduce_mean(
        (1 - tf.cast(is_valid_pred, dtype=tf.float32)) * tf.abs(max_pred - min_pred),
    )

    z_max_min_diff_error = tf.reduce_mean(
        (1 - tf.cast(is_max_min_diff_more_than_00_1, dtype=tf.float32)) * (tf.abs(max_pred - min_pred) - 0.01),
    )

    z_min_below_error = tf.reduce_mean(
        (1 - tf.cast(is_min_pred_more_than_min_true, dtype=tf.float32)) * tf.abs(min_pred - min_true),
    )

    win_amt_true_error = tf.reduce_mean((1 - tf.cast(band_inside, dtype=tf.float32)) * tf.abs(max_true - min_true))

    win_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - (tf.cast(band_inside, dtype=tf.float32) * tf.abs(max_pred - min_pred)),
    )

    correct_trends = _get_correct_trends(y_true, y_pred)

    trend_error_win = tf.reduce_mean(
        (1 - (tf.cast(band_inside, dtype=tf.float32) * tf.cast(correct_trends, dtype=tf.float32)))
        * tf.abs(max_true - min_true),
    )

    penalty_half_inside = tf.reduce_mean(
        (
            tf.cast(
                (tf.math.logical_not(is_max_pred_less_than_max_true) & is_min_pred_more_than_min_true),
                dtype=tf.float32,
            )
            * tf.abs(max_pred - max_true)
        )
        # max outside, min inside
        + (
            tf.cast(
                (tf.math.logical_not(is_min_pred_more_than_min_true) & is_max_pred_less_than_max_true),
                dtype=tf.float32,
            )
            * tf.abs(min_pred - min_true)
        ),
        # max inside, min outside
    )

    trend_error_win_pred_error = tf.reduce_mean(
        (
            tf.abs(max_true - min_true)
            - (
                tf.cast(band_inside, dtype=tf.float32)
                * tf.cast(correct_trends, dtype=tf.float32)
                * tf.abs(max_pred - min_pred)
            )
        ),
    )

    return (
        z_max_above_error
        + z_pred_valid_error
        + z_min_below_error
        + z_max_min_diff_error
        + penalty_half_inside
        + win_amt_true_error * 1.5
        + win_amt_pred_error * 2
        + trend_error_win * 3
        + trend_error_win_pred_error * 4
    )


def metric_win_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    win_fraction = tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    return win_fraction * 100


def metric_win_pred_capture_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

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
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    correct_win_trend = tf.reduce_mean(
        tf.cast(correct_trends, dtype=tf.float32) * tf.cast(wins, dtype=tf.float32),
    ) / tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    metric = tf.where(
        tf.math.is_nan(correct_win_trend),
        tf.zeros_like(correct_win_trend),
        correct_win_trend,
    )

    return metric * 100


def metric_win_pred_trend_capture_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    pred_trend_capture = tf.reduce_mean(
        tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32) * tf.cast(correct_trends, dtype=tf.float32),
    )

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))

    return pred_trend_capture / total_capture_possible * 100


def metric_abs_percent(y_true, y_pred):
    error = y_true - y_pred

    return tf.reduce_mean(tf.abs(error[:, :2])) / tf.reduce_mean(tf.abs(y_true[:, :2])) * 100


def metric_rmse_percent(y_true, y_pred):
    error = y_true - y_pred

    return tf.sqrt(tf.reduce_mean(tf.square(error[:, :2]))) / tf.reduce_mean(tf.abs(y_true[:, :2])) * 100
