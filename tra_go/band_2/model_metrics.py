import tensorflow as tf
from model_training.common import metric_abs, metric_rmse
from tensorflow.keras import backend as K

# RISK_TO_REWARD_RATIO: float = 0.2
SKIP_FIRST_PERCENTILE: float = 0.15
SKIP_LAST_PERCENTILE: float = 0.15


def metric_new_idea(y_true, y_pred):
    return (
        metric_rmse(y_true, y_pred) * 2
        + metric_abs(y_true, y_pred) * 2 / 3
        + metric_average_in(y_true, y_pred) * 2 / 3
        + metric_loss_comp_2(y_true, y_pred)
        # + metric_pred_capture(y_true, y_pred) * 10
        # + metric_win_checkpoint(y_true, y_pred) * 20
    )


def metric_average_in(y_true, y_pred):
    average_pred = (y_pred[:, :, 0] + y_pred[:, :, 1]) / 2
    average_true = (y_true[:, :, 0] + y_true[:, :, 1]) / 2

    return (
        metric_abs(average_true, average_pred)
        + metric_abs(y_true[:, :, 0], y_pred[:, :, 0])
        + metric_abs(y_true[:, :, 1], y_pred[:, :, 1])
    )


def support_idea_1(y_true, y_pred):
    min_pred_s = K.min(y_pred[:, :, 0], axis=1)
    max_pred_s = K.max(y_pred[:, :, 1], axis=1)

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
    y_pred_min = y_pred[:, :, 0]
    y_pred_max = y_pred[:, :, 1]

    n: int = y_pred.shape[1]
    start_index: int = int(n * SKIP_FIRST_PERCENTILE)
    end_index: int = n - int(n * SKIP_LAST_PERCENTILE)

    min_pred_s = K.min(y_pred_min[:, start_index:end_index], axis=1)
    max_pred_s = K.max(y_pred_max[:, start_index:end_index], axis=1)

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
    z_1 = K.mean(
        (1 - K.cast(K.all([max_true >= max_pred], axis=0), dtype=K.floatx())) * K.abs(max_true - max_pred),
    )

    z_2 = K.mean(
        (1 - K.cast(K.all([max_pred >= min_pred], axis=0), dtype=K.floatx())) * K.abs(max_pred - min_pred),
    )

    z_3 = K.mean(
        (1 - K.cast(K.all([min_pred >= min_true], axis=0), dtype=K.floatx())) * K.abs(min_pred - min_true),
    )

    win_amt_true = K.mean(
        (1 - K.cast(wins, dtype=K.floatx())) * K.abs(max_true - min_true),
    )

    return z_1, z_2, z_3, win_amt_true


def get_correct_trends(y_true, y_pred):
    n = y_pred.shape[1]

    start_index: int = int(n * SKIP_FIRST_PERCENTILE)
    end_index: int = n - int(n * SKIP_LAST_PERCENTILE)

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
        (1 - (K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx()))) * K.abs(max_true - min_true),
    )

    trend_error_win_pred_error = K.mean(
        K.abs(max_true - min_true)
        - (K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx())) * K.abs(max_pred - min_pred),
    )

    # max_inside_but_not_band = K.all(
    #     [
    #         max_true >= max_pred,
    #         max_pred >= min_true,
    #         min_pred <= min_true,
    #     ],
    #     axis=0,
    # )

    # min_inside_but_not_band = K.all(
    #     [
    #         max_true <= max_pred,
    #         max_true >= min_pred,
    #         min_pred >= min_true,
    #     ],
    #     axis=0,
    # )

    # open_trade_error = K.mean(K.cast(max_inside_but_not_band, dtype=K.floatx()) * K.abs(max_true - min_true)) + K.mean(
    #     K.cast(min_inside_but_not_band, dtype=K.floatx()) * K.abs(max_true - min_true)
    # )

    return (
        z_max_above_error
        + z_pred_valid_error
        + z_min_below_error
        # + open_trade_error
        + win_amt_true_error * 2
        + win_amt_pred_error * 4
        + trend_error_win * 8
        + trend_error_win_pred_error * 16
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


# def metric_win_checkpoint_old(y_true, y_pred):
#     n: int = y_pred.shape[1]
#     start_index: int = int(n * SKIP_FIRST_PERCENTILE)
#     end_index: int = n - int(n * SKIP_LAST_PERCENTILE)

#     min_pred_s, max_pred_s, min_true, max_true, band_inside = support_idea_1_new(
#         y_true,
#         y_pred,
#     )

#     # getting trends.

#     min_pred_index = K.argmin(y_pred[:, start_index:end_index, 0], axis=1)
#     max_pred_index = K.argmax(y_pred[:, start_index:end_index, 1], axis=1)

#     min_true_index = K.argmin(y_true[:, :, 0], axis=1)
#     max_true_index = K.argmax(y_true[:, :, 1], axis=1)

#     correct_trends_buy = K.all(
#         [
#             max_pred_index > min_pred_index,
#             max_true_index > min_true_index,
#         ],
#         axis=0,
#     )

#     correct_trends_sell = K.all(
#         [
#             max_pred_index < min_pred_index,
#             max_true_index < min_true_index,
#         ],
#         axis=0,
#     )

#     correct_trends = tf.logical_or(correct_trends_buy, correct_trends_sell)

#     max_inside_but_not_band = K.all(
#         [
#             max_true >= max_pred_s,
#             max_pred_s >= min_true,
#             min_pred_s <= min_true,
#         ],
#         axis=0,
#     )

#     min_inside_but_not_band = K.all(
#         [
#             max_true <= max_pred_s,
#             max_true >= min_pred_s,
#             min_pred_s >= min_true,
#         ],
#         axis=0,
#     )

#     # loss captured, when trade taken, but other side of trade no inside band, also considering here that risk_to_reward_ratio is unbounded.

#     # case 1: when max inside, and sell trade
#     # case 2: when min inside, and buy trade
#     # closing both trades at the last closing tick price.

#     # stoploss vs open_trade_till_last_tick

#     last_tick = (y_true[:, -1, 0] + y_true[:, -1, 1]) / 2

#     stoploss_open_sell_trade = K.maximum(
#         (max_pred_s - last_tick),
#         K.abs(max_true - min_true) * RISK_TO_REWARD_RATIO * -1,
#     )

#     stoploss_open_buy_trade = K.maximum(
#         (last_tick - min_pred_s),
#         K.abs(max_true - min_true) * RISK_TO_REWARD_RATIO * -1,
#     )

#     open_trade_capture_new = (
#         K.mean(
#             stoploss_open_sell_trade
#             * K.cast(max_inside_but_not_band, dtype=K.floatx())
#             * K.cast(max_pred_index < min_pred_index, dtype=K.floatx()),
#         )
#         # open sell trade
#         + K.mean(
#             stoploss_open_buy_trade
#             * K.cast(min_inside_but_not_band, dtype=K.floatx())
#             * K.cast(max_pred_index > min_pred_index, dtype=K.floatx()),
#         )
#         # open buy trade
#     )

#     pred_capture_new = K.mean(
#         (max_pred_s - min_pred_s) * K.cast(band_inside, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx()),
#     )

#     total_capture_possible = K.mean(K.abs(max_true - min_true))

#     return total_capture_possible - (open_trade_capture_new + pred_capture_new * 2)


def metric_win_checkpoint(y_true, y_pred):
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    correct_trends = get_correct_trends(y_true, y_pred)

    pred_capture = K.mean(
        K.abs(max_pred - min_pred) * K.cast(wins, dtype=K.floatx()) * K.cast(correct_trends, dtype=K.floatx()),
    )

    total_capture_possible = K.mean(K.abs(max_true - min_true))

    return total_capture_possible - pred_capture
