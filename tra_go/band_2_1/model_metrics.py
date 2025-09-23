import model_training.common as training_common
import tensorflow as tf
from core.config import settings

# Small epsilon for safe divisions in percent metrics
EPS = tf.constant(1e-7, dtype=tf.float32)


def loss_function(y_true, y_pred):
    return (
        training_common.metric_abs(y_true[:, :2], y_pred[:, :2])
        + training_common.metric_rmse(y_true[:, :2], y_pred[:, :2])
        + metric_loss_comp_2(y_true, y_pred)
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
    pred_trend = tf.greater_equal(y_pred[:, 2], 0.5)  # bool
    true_trend = tf.cast(y_true[:, 2], dtype=tf.bool)  # ensure bool

    correct_trends = tf.equal(pred_trend, true_trend)

    return correct_trends


def metric_correct_trends_full(y_true, y_pred):
    correct_trends = _get_correct_trends(y_true, y_pred)

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
        tf.cast(tf.logical_not(is_max_pred_less_than_max_true), dtype=tf.float32) * tf.abs(max_true - max_pred),
    )

    z_pred_valid_error = tf.reduce_mean(
        tf.cast(tf.logical_not(tf.greater_equal(min_pred, min_true)), dtype=tf.float32) * tf.abs(max_pred - min_pred),
    )

    z_max_min_diff_error = tf.reduce_mean(
        tf.cast(tf.logical_not(is_max_min_diff_more_than_00_1), dtype=tf.float32)
        * (tf.abs(max_pred - min_pred) - 0.03),
    )

    z_min_below_error = tf.reduce_mean(
        tf.cast(tf.logical_not(is_min_pred_more_than_min_true), dtype=tf.float32) * tf.abs(min_pred - min_true),
    )

    win_amt_true_error = tf.reduce_mean(
        tf.cast(tf.logical_not(band_inside), dtype=tf.float32) * tf.abs(max_true - min_true),
    )

    win_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - tf.cast(band_inside, dtype=tf.float32) * tf.abs(max_pred - min_pred),
    )

    correct_trends = _get_correct_trends(y_true, y_pred)

    trend_amt_true_error = tf.reduce_mean(
        tf.cast(tf.logical_not(correct_trends), dtype=tf.float32) * tf.abs(max_true - min_true),
    )

    trend_amt_pred_error = tf.reduce_mean(
        tf.abs(max_true - min_true) - tf.cast(correct_trends, dtype=tf.float32) * tf.abs(max_pred - min_pred),
    )

    trend_win_true_error = tf.reduce_mean(
        tf.cast(tf.logical_not(band_inside & correct_trends), dtype=tf.float32) * tf.abs(max_true - min_true),
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

    # Use thresholded boolean trend for consistent behavior with other metrics
    pred_trend = tf.greater_equal(y_pred[:, 2], 0.5)  # bool

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
                (tf.logical_not(is_max_pred_less_than_max_true) & is_min_pred_more_than_min_true & pred_trend),
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
                    & tf.logical_not(pred_trend)
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


def metric_pred_capture_per_win_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_win_pred_capture_possible = tf.reduce_mean(
        (max_true - min_true) * tf.cast(wins, dtype=tf.float32),
    )

    x = pred_capture / (total_win_pred_capture_possible + EPS)
    return x * 100


def metric_win_pred_capture_total_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    pred_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins, dtype=tf.float32))

    total_capture_possible = tf.reduce_mean(max_true - min_true)
    return pred_capture / (total_capture_possible + EPS) * 100


def metric_win_correct_trend_percent(y_true, y_pred):
    return metric_correct_trend_per_win_percent(y_true, y_pred)


def metric_correct_trend_per_win_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)
    num = tf.reduce_mean(tf.cast(correct_trends & wins, dtype=tf.float32))
    denom = tf.reduce_mean(tf.cast(wins, dtype=tf.float32))
    correct_win_trend = num / (denom + EPS)
    return correct_win_trend * 100


def metric_win_pred_trend_capture_percent(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    pred_trend_capture = tf.reduce_mean(tf.abs(max_pred - min_pred) * tf.cast(wins & correct_trends, dtype=tf.float32))

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))
    return pred_trend_capture / (total_capture_possible + EPS) * 100


def metric_try_1(y_true, y_pred):
    wins = _get_band_inside(y_true, y_pred)

    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    wins = _get_band_inside(y_true, y_pred)

    correct_trends = _get_correct_trends(y_true, y_pred)

    pred_trend_capture = tf.reduce_mean(
        tf.abs(max_pred - min_pred) * tf.cast(wins & correct_trends, dtype=tf.float32),
    )

    total_capture_possible = tf.reduce_mean(tf.abs(max_true - min_true))
    return (pred_trend_capture - stoploss_incurred(y_true, y_pred)) / (total_capture_possible + EPS) * 100


def stoploss_incurred(y_true, y_pred):
    min_true, max_true, min_pred, max_pred = _get_min_max_values(y_true, y_pred)

    is_valid_pred = tf.greater_equal(max_pred, min_pred)

    pred_trend = tf.greater_equal(y_pred[:, 2], 0.5)  # bool
    buy_trade_possible = (
        (tf.less_equal(min_true, min_pred) & tf.less_equal(min_pred, max_true))
        # min_inside
        & pred_trend
        # where pred_trend True == buy
    )

    sell_trade_possible = (
        (tf.less_equal(min_true, max_pred) & tf.less_equal(max_pred, max_true))
        # max inside
        & tf.logical_not(pred_trend)
        # where pred_trend False == sell
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
    return pred_trend_capture / (total_capture_possible + EPS) * 100


def metric_abs_percent(y_true, y_pred):
    return training_common.metric_abs_percent(y_true[:, :2], y_pred[:, :2])


def metric_rmse_percent(y_true, y_pred):
    return training_common.metric_rmse_percent(y_true[:, :2], y_pred[:, :2])
