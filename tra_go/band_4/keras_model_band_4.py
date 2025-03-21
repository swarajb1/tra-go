import tensorflow as tf
from keras import backend as K
from keras_model import metric_abs

SKIP_PERCENTILE: float = 0.18

SKIP_FIRST_PERCENTILE: float = 0.18
SKIP_LAST_PERCENTILE: float = 0.18


RISK_TO_REWARD_RATIO: float = 0.2


def metric_new_idea(y_true, y_pred):
    return (
        # metric_rmse(y_true, y_pred) * 2
        metric_abs(y_true, y_pred) * 2
        + metric_average_in(y_true, y_pred)
        + metric_loss_comp_2(y_true, y_pred)
        + (1 - metric_win_checkpoint(y_true, y_pred) / 100) / 40
    )


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, :, 2] + y_pred[:, :, 1]) / 2
    average_true = (y_true[:, :, 2] + y_true[:, :, 1]) / 2

    return (
        metric_abs(average_true, average_pred)
        + metric_abs(y_true[:, :, 2], y_pred[:, :, 2])
        + metric_abs(y_true[:, :, 1], y_pred[:, :, 1])
    )


def support_idea_1(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index * 2

    min_pred_s = K.min(y_pred[:, start_index:end_index, 2], axis=1)
    max_pred_s = K.max(y_pred[:, start_index:end_index, 1], axis=1)

    min_pred = K.min(y_pred[:, :, 2], axis=1)
    max_pred = K.max(y_pred[:, :, 1], axis=1)

    min_true = K.min(y_true[:, :, 2], axis=1)
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
    y_pred_min = y_pred[:, :, 2]
    y_pred_max = y_pred[:, :, 1]

    n: int = y_pred.shape[1]
    start_index: int = int(n * SKIP_FIRST_PERCENTILE)
    end_index: int = n - int(n * SKIP_LAST_PERCENTILE)

    min_pred_s = K.min(y_pred_min[:, start_index:end_index], axis=1)
    max_pred_s = K.max(y_pred_max[:, start_index:end_index], axis=1)

    min_true = K.min(y_true[:, :, 2], axis=1)
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
    z_1 = K.mean(
        (1 - K.cast(K.all([max_true >= max_pred], axis=0), dtype=K.floatx()))
        * K.abs(max_true - max_pred),
    )

    z_2 = K.mean(
        (1 - K.cast(K.all([max_pred >= min_pred], axis=0), dtype=K.floatx()))
        * K.abs(max_pred - min_pred),
    )

    z_3 = K.mean(
        (1 - K.cast(K.all([min_pred >= min_true], axis=0), dtype=K.floatx()))
        * K.abs(min_pred - min_true),
    )

    win_amt_true = K.sum(
        (1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true),
    )

    return z_1, z_2, z_3, win_amt_true


def support_idea_3(y_true, y_pred):
    n = y_pred.shape[1]
    start_index = int(n * SKIP_PERCENTILE)
    end_index = n - start_index

    min_pred_index = K.argmin(y_pred[:, start_index:end_index, 2], axis=1)
    max_pred_index = K.argmax(y_pred[:, start_index:end_index, 1], axis=1)

    # min_pred_index = K.argmin(y_pred[:, :, 2], axis=1)
    # max_pred_index = K.argmax(y_pred[:, :, 1], axis=1)

    min_true_index = K.argmin(y_true[:, :, 2], axis=1)
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

    z_max_above_error = K.mean(
        (1 - K.cast(K.all([max_true >= max_pred], axis=0), dtype=K.floatx()))
        * K.abs(max_true - max_pred),
    )

    z_pred_valid_error = K.mean(
        (1 - K.cast(K.all([max_pred >= min_pred], axis=0), dtype=K.floatx()))
        * K.abs(max_pred - min_pred),
    )

    z_min_below_error = K.mean(
        (1 - K.cast(K.all([min_pred >= min_true], axis=0), dtype=K.floatx()))
        * K.abs(min_pred - min_true),
    )

    win_amt_true_error = K.mean(
        (1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true),
    )

    win_amt_pred_error = K.mean(
        K.abs(max_true - min_true)
        - (K.cast(wins, dtype=K.floatx()) * K.abs(max_pred - min_pred)),
    )

    # return z_1 + z_2 + z_3 + win_amt_true_error + win_amt_pred_error

    correct_trends = support_idea_3(y_true, y_pred)

    trend_error_win = K.mean(
        (
            1
            - (
                K.cast(wins, dtype=K.floatx())
                * K.cast(correct_trends, dtype=K.floatx())
            )
        )
        * K.abs(max_true - min_true),
    )

    return (
        z_max_above_error
        + z_pred_valid_error
        + z_min_below_error
        + trend_error_win
        + win_amt_true_error * 2
        + win_amt_pred_error * 10
    )


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


def metric_win_checkpoint(y_true, y_pred):
    n: int = y_pred.shape[1]
    start_index: int = int(n * SKIP_FIRST_PERCENTILE)
    end_index: int = n - int(n * SKIP_LAST_PERCENTILE)

    min_pred_s, max_pred_s, min_true, max_true, band_inside = support_idea_1_new(
        y_true,
        y_pred,
    )

    # getting trends.

    min_pred_index = K.argmin(y_pred[:, start_index:end_index, 2], axis=1)
    max_pred_index = K.argmax(y_pred[:, start_index:end_index, 1], axis=1)

    min_true_index = K.argmin(y_true[:, :, 2], axis=1)
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

    max_inside_but_not_band = K.all(
        [
            max_true >= max_pred_s,
            max_pred_s >= min_true,
            min_pred_s <= min_true,
        ],
        axis=0,
    )

    min_inside_but_not_band = K.all(
        [
            max_true <= max_pred_s,
            max_true >= min_pred_s,
            min_pred_s >= min_true,
        ],
        axis=0,
    )

    # loss captured, when trade taken, but other side of trade no inside band, also considering here that risk_to_reward_ratio is unbounded.

    # case 1: when max inside, and sell trade
    # case 2: when min inside, and buy trade
    # closing both trades at the last closing tick price.

    # stoploss vs open_trade_till_last_tick

    stoploss_open_sell_trade = K.minimum(
        (max_pred_s / y_true[:, -1, 3] - 1),
        (max_pred_s / min_pred_s - 1) * RISK_TO_REWARD_RATIO,
    )

    stoploss_open_buy_trade = K.minimum(
        (1 - y_true[:, -1, 3] / min_pred_s),
        (1 - min_pred_s / max_pred_s) * RISK_TO_REWARD_RATIO,
    )

    open_trade_capture = (
        K.sum(
            stoploss_open_sell_trade
            * K.cast(max_inside_but_not_band, dtype=K.floatx())
            * K.cast(max_pred_index < min_pred_index, dtype=K.floatx()),
        )
        # open sell trade
        + K.sum(
            stoploss_open_buy_trade
            * K.cast(min_inside_but_not_band, dtype=K.floatx())
            * K.cast(max_pred_index > min_pred_index, dtype=K.floatx()),
        )
        # open buy trade
    )

    max_loss = (
        K.sum(
            (1 - min_true / max_true)
            * K.cast(max_true_index < min_true_index, dtype=K.floatx()),
        )
        # loss = buying high and selling low, sell trade loss
        + K.sum(
            (max_true / min_true - 1)
            * K.cast(max_true_index > min_true_index, dtype=K.floatx()),
        )
        # loss = selling low and buying high, buy trade loss
    )

    open_trade_capture_x = tf.where(
        tf.math.is_nan(open_trade_capture),
        tf.zeros_like(open_trade_capture),
        open_trade_capture,
    )

    total_capture_possible = K.sum(max_true / min_true - 1)

    # return (max_loss + open_trade_capture_x * 3 + pred_capture * 10) / (total_capture_possible * 10 + max_loss)
    return (open_trade_capture_x + pred_capture * 2) / (total_capture_possible)
