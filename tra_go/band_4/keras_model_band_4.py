import tensorflow as tf
from keras import backend as K
from keras_model import metric_abs

# Consolidate constants with same values
SKIP_PERCENTILE: float = 0.18
SKIP_FIRST_PERCENTILE: float = SKIP_PERCENTILE
SKIP_LAST_PERCENTILE: float = SKIP_PERCENTILE

RISK_TO_REWARD_RATIO: float = 0.2


def metric_new_idea(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return (
        # metric_rmse(y_true, y_pred) * 2
        metric_abs(y_true, y_pred) * 2
        + metric_average_in(y_true, y_pred)
        + metric_loss_comp_2(y_true, y_pred)
        + (1 - metric_win_checkpoint(y_true, y_pred) / 100) / 40
    )


def metric_average_in(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # Extract slices once to avoid repeated indexing
    y_true_min = y_true[:, :, 2]
    y_true_max = y_true[:, :, 1]
    y_pred_min = y_pred[:, :, 2]
    y_pred_max = y_pred[:, :, 1]

    average_pred = (y_pred_min + y_pred_max) / 2
    average_true = (y_true_min + y_true_max) / 2

    return (
        metric_abs(average_true, average_pred)
        + metric_abs(y_true_min, y_pred_min)
        + metric_abs(y_true_max, y_pred_max)
    )


def _calculate_indices(y_pred: tf.Tensor, skip_first: float, skip_last: float) -> tuple[int, int]:
    """Calculate start and end indices based on skip percentages."""
    n: int = y_pred.shape[1]
    start_index: int = int(n * skip_first)
    end_index: int = n - int(n * skip_last)
    return start_index, end_index


def support_idea_1(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Extract slices once
    y_pred_min = y_pred[:, :, 2]
    y_pred_max = y_pred[:, :, 1]
    y_true_min = y_true[:, :, 2]
    y_true_max = y_true[:, :, 1]

    n: int = y_pred.shape[1]
    start_index: int = int(n * SKIP_PERCENTILE)
    end_index: int = n - start_index * 2

    min_pred_s = K.min(y_pred_min[:, start_index:end_index], axis=1)
    max_pred_s = K.max(y_pred_max[:, start_index:end_index], axis=1)

    min_pred = K.min(y_pred_min, axis=1)
    max_pred = K.max(y_pred_max, axis=1)

    min_true = K.min(y_true_min, axis=1)
    max_true = K.max(y_true_max, axis=1)

    # Combine conditions in one tensor operation
    conditions = tf.stack(
        [
            max_true >= max_pred_s,
            max_pred_s >= min_pred_s,
            min_pred_s >= min_true,
        ],
        axis=0,
    )
    wins = K.all(conditions, axis=0)

    return min_pred, max_pred, min_true, max_true, wins


def support_idea_1_new(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Extract slices once
    y_pred_min = y_pred[:, :, 2]
    y_pred_max = y_pred[:, :, 1]
    y_true_min = y_true[:, :, 2]
    y_true_max = y_true[:, :, 1]

    start_index, end_index = _calculate_indices(y_pred, SKIP_FIRST_PERCENTILE, SKIP_LAST_PERCENTILE)

    min_pred_s = K.min(y_pred_min[:, start_index:end_index], axis=1)
    max_pred_s = K.max(y_pred_max[:, start_index:end_index], axis=1)

    min_true = K.min(y_true_min, axis=1)
    max_true = K.max(y_true_max, axis=1)

    # Combine conditions in one tensor operation
    conditions = tf.stack(
        [
            max_true >= max_pred_s,
            max_pred_s >= min_pred_s,  # valid_pred
            min_pred_s >= min_true,
        ],
        axis=0,
    )
    band_inside = K.all(conditions, axis=0)

    return min_pred_s, max_pred_s, min_true, max_true, band_inside


def _calculate_error_metric(condition: tf.Tensor, diff: tf.Tensor, use_mean: bool = True) -> tf.Tensor:
    """Helper to calculate error metric based on condition."""
    cast_cond = K.cast(condition, dtype=K.floatx())
    result = (1 - cast_cond) * diff
    return K.mean(result) if use_mean else K.sum(result)


def support_idea_2(
    min_pred: tf.Tensor,
    max_pred: tf.Tensor,
    min_true: tf.Tensor,
    max_true: tf.Tensor,
    wins: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    max_pred_valid = K.all([max_true >= max_pred], axis=0)
    min_pred_valid = K.all([max_pred >= min_pred], axis=0)
    min_true_valid = K.all([min_pred >= min_true], axis=0)

    z_1 = _calculate_error_metric(max_pred_valid, K.abs(max_true - max_pred))
    z_2 = _calculate_error_metric(min_pred_valid, K.abs(max_pred - min_pred))
    z_3 = _calculate_error_metric(min_true_valid, K.abs(min_pred - min_true))

    win_amt_true = _calculate_error_metric(wins, K.abs(max_true - min_true), use_mean=False)

    return z_1, z_2, z_3, win_amt_true


def support_idea_3(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # Extract slices once
    y_pred_min = y_pred[:, :, 2]
    y_pred_max = y_pred[:, :, 1]
    y_true_min = y_true[:, :, 2]
    y_true_max = y_true[:, :, 1]

    start_index, end_index = _calculate_indices(y_pred, SKIP_PERCENTILE, SKIP_PERCENTILE)

    min_pred_index = K.argmin(y_pred_min[:, start_index:end_index], axis=1)
    max_pred_index = K.argmax(y_pred_max[:, start_index:end_index], axis=1)

    min_true_index = K.argmin(y_true_min, axis=1)
    max_true_index = K.argmax(y_true_max, axis=1)

    # Determine buy/sell conditions
    buy_condition = tf.logical_and(max_pred_index > min_pred_index, max_true_index > min_true_index)

    sell_condition = tf.logical_and(max_pred_index < min_pred_index, max_true_index < min_true_index)

    return tf.logical_or(buy_condition, sell_condition)


def metric_loss_comp_2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # Get key metrics once to avoid recalculation
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    # Reuse common calculations
    abs_true_range = K.abs(max_true - min_true)
    abs_pred_range = K.abs(max_pred - min_pred)
    wins_float = K.cast(wins, dtype=K.floatx())

    # Calculate error metrics efficiently
    z_max_above_error = _calculate_error_metric(K.all([max_true >= max_pred], axis=0), K.abs(max_true - max_pred))

    z_pred_valid_error = _calculate_error_metric(K.all([max_pred >= min_pred], axis=0), abs_pred_range)

    z_min_below_error = _calculate_error_metric(K.all([min_pred >= min_true], axis=0), K.abs(min_pred - min_true))

    win_amt_true_error = K.mean((1 - wins_float) * abs_true_range)

    win_amt_pred_error = K.mean(abs_true_range - (wins_float * abs_pred_range))

    # Get trend information
    correct_trends = support_idea_3(y_true, y_pred)
    correct_trends_float = K.cast(correct_trends, dtype=K.floatx())

    trend_error_win = K.mean((1 - (wins_float * correct_trends_float)) * abs_true_range)

    # Combine all metrics
    return (
        z_max_above_error
        + z_pred_valid_error
        + z_min_below_error
        + trend_error_win
        + win_amt_true_error * 2
        + win_amt_pred_error * 10
    )


def metric_win_pred_capture_percent(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_win_pred_capture_possible = K.sum(
        (max_true / min_true - 1) * K.cast(wins, dtype=K.floatx()),
    )

    x = pred_capture / total_win_pred_capture_possible

    metric = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

    return metric * 100


def metric_pred_capture_percent(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    pred_capture = K.sum((max_pred / min_pred - 1) * K.cast(wins, dtype=K.floatx()))

    total_capture_possible = K.sum(max_true / min_true - 1)

    return pred_capture / total_capture_possible * 100


def metric_win_percent(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    min_pred, max_pred, min_true, max_true, wins = support_idea_1_new(y_true, y_pred)

    win_fraction = K.mean(K.cast(wins, dtype=K.floatx()))

    return win_fraction * 100


def metric_correct_win_trend_percent(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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


def metric_win_checkpoint(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
            (1 - min_true / max_true) * K.cast(max_true_index < min_true_index, dtype=K.floatx()),
        )
        # loss = buying high and selling low, sell trade loss
        + K.sum(
            (max_true / min_true - 1) * K.cast(max_true_index > min_true_index, dtype=K.floatx()),
        )
        # loss = selling low and buying high, buy trade loss
    )

    open_trade_capture_x = tf.where(
        tf.math.is_nan(open_trade_capture),
        tf.zeros_like(open_trade_capture),
        open_trade_capture,
    )

    # Calculate total capture
    total_capture_possible = K.sum(max_true / min_true - 1)

    # Calculate final metric with safe division
    return (open_trade_capture_x + pred_capture * 2) / (total_capture_possible + K.epsilon())
