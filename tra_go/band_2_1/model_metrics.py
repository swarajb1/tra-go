import tensorflow as tf
from core.config import settings
from keras_model_tf import metric_abs


def loss_function(y_true, y_pred):
    return (
        # metric_abs(y_true[:, :2], y_pred[:, :2])
        # +
        # metric_abs(y_true[:, 2:], y_pred[:, 2:]) / 20
        # +
        # metric_average_in(y_true, y_pred)
        # +
        metric_loss_comp_2(y_true, y_pred)
    )


def metric_average_in(y_true, y_pred):
    average_true = (y_true[:, 0] + y_true[:, 1]) / 2
    average_pred = (y_pred[:, 0] + y_pred[:, 1]) / 2

    return (
        metric_abs(average_true, average_pred)
        # + metric_abs(y_true[:, 0], y_pred[:, 0])
        # + metric_abs(y_true[:, 1], y_pred[:, 1])
    )


def _get_min_max_values(y_true, y_pred):
    min_true = y_true[:, 0]
    max_true = y_true[:, 1]

    min_pred = y_pred[:, 0]
    max_pred = y_pred[:, 1]

    return min_true, max_true, min_pred, max_pred


def _get_band_inside(y_true, y_pred):
    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    is_valid_pred = tf.greater_equal(max_pred, min_pred)

    is_max_pred_less_than_max_true = tf.less_equal(max_pred, max_true)

    is_min_pred_more_than_min_true = tf.greater_equal(min_pred, min_true)

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
    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    is_valid_pred = tf.greater_equal(max_pred, min_pred)

    is_max_pred_less_than_max_true = tf.less_equal(max_pred, max_true)

    is_min_pred_more_than_min_true = tf.greater_equal(min_pred, min_true)

    is_max_min_diff_more_than_00_1 = tf.greater_equal(max_pred - min_pred, 0.01)

    band_inside = is_valid_pred & is_max_pred_less_than_max_true & is_min_pred_more_than_min_true

    z_max_above_error = tf.reduce_mean(
        (1 - tf.cast(is_max_pred_less_than_max_true, dtype=tf.float32)) * tf.abs(max_true - max_pred),
    )

    z_pred_valid_error = tf.reduce_mean(
        (1 - tf.cast(tf.greater_equal(min_pred, min_true), dtype=tf.float32)) * tf.abs(max_pred - min_pred),
    )

    z_max_min_diff_error = tf.reduce_mean(
        (1 - tf.cast(is_max_min_diff_more_than_00_1, dtype=tf.float32)) * (tf.abs(max_pred - min_pred) - 0.03),
    )

    z_min_below_error = tf.reduce_mean(
        (1 - tf.cast(is_min_pred_more_than_min_true, dtype=tf.float32)) * tf.abs(min_pred - min_true),
    )

    win_amt_true_error = tf.reduce_mean((1 - tf.cast(band_inside, dtype=tf.float32)) * tf.abs(max_true - min_true))

    win_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - tf.cast(band_inside, dtype=tf.float32) * tf.abs(max_pred - min_pred),
    )

    correct_trends = _get_correct_trends(y_true, y_pred)

    trend_amt_true_error = tf.reduce_mean(
        (1 - tf.cast(correct_trends, dtype=tf.float32)) * tf.abs(max_true - min_true),
    )

    trend_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - tf.cast(correct_trends, dtype=tf.float32) * tf.abs(max_pred - min_pred),
    )

    trend_win_true_error = tf.reduce_mean(
        (1 - tf.cast(band_inside & correct_trends, dtype=tf.float32)) * tf.abs(max_true - min_true),
    )

    trend_win_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true)
        - tf.cast(band_inside & correct_trends, dtype=tf.float32) * tf.abs(max_pred - min_pred),
    )

    return (
        z_max_above_error
        + z_min_below_error
        + z_pred_valid_error
        + z_max_min_diff_error
        + penalty_half_inside(y_true, y_pred)
        + win_amt_true_error
        + win_amt_pred_error
        + trend_amt_true_error
        + trend_amt_pred_error
        + trend_win_true_error
        + trend_win_pred_error
        + stoploss_incurred(y_true, y_pred) / 6
    )


def penalty_half_inside(y_true, y_pred):
    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    trend_pred = y_pred[:, 2]

    is_max_pred_less_than_max_true = tf.less_equal(max_pred, max_true)

    is_min_pred_more_than_min_true = tf.greater_equal(min_pred, min_true)

    #  the part of the band inside is the error
    penalty_half_inside = tf.reduce_mean(
        (
            tf.cast(
                (tf.logical_not(is_max_pred_less_than_max_true) & is_min_pred_more_than_min_true),
                dtype=tf.float32,
            )
            * tf.abs(max_true - min_pred)
        )
        # max outside, min inside,
        + (
            tf.cast(
                (tf.logical_not(is_min_pred_more_than_min_true) & is_max_pred_less_than_max_true),
                dtype=tf.float32,
            )
            * tf.abs(max_true - min_true)
        ),
        # max inside, min outside
    )

    penalty_half_inside_trend = tf.reduce_mean(
        (
            tf.cast(
                (
                    tf.logical_not(is_max_pred_less_than_max_true)
                    & is_min_pred_more_than_min_true
                    & tf.equal(trend_pred, 1)
                ),
                dtype=tf.float32,
            )
            * tf.abs(max_true - min_pred)
        )
        # max outside, min inside, order type buy
        + (
            tf.cast(
                (
                    tf.logical_not(is_min_pred_more_than_min_true)
                    & is_max_pred_less_than_max_true
                    & tf.equal(trend_pred, 0)
                ),
                dtype=tf.float32,
            )
            * tf.abs(max_true - min_true)
        ),
        # max inside, min outside, order type sell
    )

    return penalty_half_inside + penalty_half_inside_trend


def metric_win_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    win_fraction = tf.reduce_mean(tf.cast(wins, dtype=tf.float32))

    return win_fraction * 100


def metric_win_pred_capture_percent(y_true, y_pred):
    return metric_pred_capture_per_win_percent(y_true, y_pred)


def metric_pred_capture_per_win_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_win_pred_capture_possible = tf.reduce_mean(
        (max_true - min_true) * tf.cast(wins, dtype=tf.float32),
    )

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_win_pred_capture_total_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_capture_possible = tf.reduce_mean(max_true - min_true)

    return pred_capture / total_capture_possible * 100


def metric_win_correct_trend_percent(y_true, y_pred):
    return metric_correct_trend_per_win_percent(y_true, y_pred)


def metric_correct_trend_per_win_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    correct_win_trend = tf.reduce_mean(tf.cast(correct_trends & wins, dtype=tf.float32)) / tf.reduce_mean(
        tf.cast(wins, dtype=tf.float32),
    )

    metric = tf.where(
        tf.math.is_nan(correct_win_trend),
        tf.zeros_like(correct_win_trend),
        correct_win_trend,
    )

    return metric * 100


def metric_win_pred_trend_capture_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    pred_trend_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins & correct_trends, dtype=tf.float32))

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))

    return pred_trend_capture / total_capture_possible * 100


def metric_try_1(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    pred_trend_capture = tf.reduce_mean(
        tf.abs(max_pred - min_pred) * tf.cast(wins & correct_trends, dtype=tf.float32),
    )

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))

    return (pred_trend_capture - stoploss_incurred(y_true, y_pred)) / total_capture_possible * 100


def stoploss_incurred(y_true, y_pred):
    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    trend_pred = y_pred[:, 2]

    is_valid_pred = tf.greater_equal(max_pred, min_pred)

    buy_trade_possible = (
        (tf.less_equal(min_true, min_pred) & tf.less_equal(min_pred, max_true))
        # min_inside
        & tf.equal(trend_pred, 1)
        # pred trend buy
    )

    sell_trade_possible = (
        (tf.less_equal(min_true, max_pred) & tf.less_equal(max_pred, max_true))
        # max inside
        & tf.equal(trend_pred, 0)
        # pred trend sell
    )

    # when the trend was buy, but the price went down
    stoploss_1_price = min_pred - tf.abs(max_pred - min_pred) * settings.RISK_TO_REWARD_RATIO

    # when the trend was sell, but the price went up
    stoploss_2_price = max_pred + tf.abs(max_pred - min_pred) * settings.RISK_TO_REWARD_RATIO

    stoploss_hit_on_buy = tf.less_equal(stoploss_1_price, min_true) & buy_trade_possible

    stoploss_hit_on_sell = tf.greater_equal(stoploss_2_price, max_true) & sell_trade_possible

    return (
        tf.reduce_mean(
            tf.cast(
                (stoploss_hit_on_buy | stoploss_hit_on_sell) & is_valid_pred,
                dtype=tf.float32,
            )
            * tf.abs(max_pred - min_pred),
        )
        * settings.RISK_TO_REWARD_RATIO
    )


def metric_try_2(y_true, y_pred):
    # metric_win_pred_trend_capture_percent
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

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
