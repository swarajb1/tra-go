import keras_model as km
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import custom_object_scope
from tensorflow import keras

import tra_go.band_4.keras_model_band_4 as km_4


def get_number_of_epochs():
    from main import NUMBER_OF_EPOCHS

    return NUMBER_OF_EPOCHS


def custom_evaluate_safety_factor(
    X_test,
    Y_test,
    y_type: str,
    testsize: float,
    now_datetime: str,
):
    # convert y_test to same format as y_pred
    with custom_object_scope(
        {
            "metric_rmse_percent": km.metric_rmse_percent,
            "metric_abs_percent": km.metric_abs_percent,
            "metric_new_idea_2": km_4.metric_new_idea_2,
            "metric_loss_comp_2": km_4.metric_loss_comp_2,
            "metric_win_percent": km_4.metric_win_percent,
            "metric_pred_capture_percent": km_4.metric_pred_capture_percent,
            "metric_win_pred_capture_percent": km_4.metric_win_pred_capture_percent,
            "metric_all_candle_in": km_4.metric_all_candle_in,
        },
    ):
        model = keras.models.load_model(
            f"training/models/model - {now_datetime} - {y_type}",
        )
        model.summary()

    y_pred = model.predict(X_test)

    SKIP_FIRST_PERCENTILE = 0.2

    # low, high, open, close

    y_pred = transform_y_array(y_arr=y_pred, skip_first_percentile=SKIP_FIRST_PERCENTILE, safety_factor=1)

    Y_test = transform_y_array(y_arr=Y_test)

    y_pred = correct_pred_values(y_pred)

    function_make_win_graph(
        y_true=Y_test,
        y_pred=y_pred,
        testsize=testsize,
        y_type=y_type,
        now_datetime=now_datetime,
    )

    function_error_132_graph(y_pred=y_pred, y_test=Y_test, now_datetime=now_datetime, y_type=y_type)

    return


def transform_y_array(
    y_arr: np.ndarray,
    skip_first_percentile: float = 0,
    safety_factor: float = 1,
) -> np.ndarray:
    first_non_eiminated_element_index: int = int(skip_first_percentile * y_arr.shape[1])

    res: np.ndarray = y_arr.copy()

    for i in range(first_non_eiminated_element_index):
        res[:, i, :] = y_arr[:, first_non_eiminated_element_index, :]

    if safety_factor < 1:
        res[:, :, 0] = (res[:, :, 0] + res[:, :, 1]) / 2 - (res[:, :, 1] - res[:, :, 0]) / 2 * safety_factor
        res[:, :, 1] = (res[:, :, 0] + res[:, :, 1]) / 2 + (res[:, :, 1] - res[:, :, 0]) / 2 * safety_factor

    return res


def correct_pred_values(y_arr: np.ndarray) -> np.ndarray:
    res = y_arr.copy()
    # step 1 - correct /exchange low/high values if needed., for each candle

    for i_day in range(res.shape[0]):
        for i_tick in range(res.shape[1]):
            if res[i_day, i_tick, 0] > res[i_day, i_tick, 1]:
                res[i_day, i_tick, 0], res[i_day, i_tick, 1] = res[i_day, i_tick, 1], res[i_day, i_tick, 0]

    # step 2 - correct values of open/close so that they are inside the low/high
    # for i_day in range(res.shape[0]):
    #     for i_tick in range(res.shape[1]):
    #         min_val_index: int = np.argmin(res[i_day, i_tick, :])
    #         max_val_index: int = np.argmax(res[i_day, i_tick, :])

    #         # swapping min/max values
    #         if min_val_index != 0:
    #             (res[i_day, i_tick, 0], res[i_day, i_tick, min_val_index]) = (
    #                 res[i_day, i_tick, min_val_index],
    #                 res[i_day, i_tick, 0],
    #             )

    #         if max_val_index != 1:
    #             (res[i_day, i_tick, 1], res[i_day, i_tick, max_val_index]) = (
    #                 res[i_day, i_tick, max_val_index],
    #                 res[i_day, i_tick, 1],
    #             )

    return res


def function_error_132_graph(y_pred, y_test, now_datetime, y_type):
    error_a = np.abs(y_pred - y_test)

    new_array = np.empty(shape=(0, 4))

    # average error np array
    for i_tick in range(error_a.shape[1]):
        low = error_a[:, i_tick, 0].sum()
        high = error_a[:, i_tick, 1].sum()
        open = error_a[:, i_tick, 2].sum()
        close = error_a[:, i_tick, 3].sum()

        to_add_array = np.array(
            [
                low / error_a.shape[0],
                high / error_a.shape[0],
                open / error_a.shape[0],
                close / error_a.shape[0],
            ],
        )

        new_array = np.concatenate((new_array, np.array([to_add_array])), axis=0)

    y1 = new_array[:, 0] * 100
    y2 = new_array[:, 1] * 100
    y3 = new_array[:, 2] * 100
    y4 = new_array[:, 3] * 100

    # Create x-axis values
    x = np.arange(len(new_array))

    fig = plt.figure(figsize=(16, 9))

    plt.plot(x, y1, label="low Δ")
    plt.plot(x, y2, label="high Δ")
    plt.plot(x, y3, label="open Δ")
    plt.plot(x, y4, label="close Δ")

    plt.title(
        f" name: {now_datetime}\n"
        + f"NUMBER_OF_NEURONS = {km.NUMBER_OF_NEURONS}  "
        + f"NUMBER_OF_LAYERS = {km.NUMBER_OF_LAYERS}\n"
        + f"NUMBER_OF_EPOCHS = {get_number_of_epochs()} | "
        + f"INITIAL_DROPOUT = {km.INITIAL_DROPOUT} | "
        + f"WEIGHT_FOR_MEA = {km.WEIGHT_FOR_MEA}",
        fontsize=20,
    )

    # Set labels and title
    plt.xlabel("serial", fontsize=15)
    plt.ylabel("perc", fontsize=15)
    plt.legend(fontsize=15)

    filename = f"training/graphs/{y_type} - {now_datetime} - abs.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    return


def function_make_win_graph(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_type: str,
    now_datetime: str,
    testsize: float,
):
    min_pred: np.ndarray = np.min(y_pred[:, :, 0], axis=1)
    max_pred: np.ndarray = np.max(y_pred[:, :, 1], axis=1)

    min_true: np.ndarray = np.min(y_true[:, :, 0], axis=1)
    max_true: np.ndarray = np.max(y_true[:, :, 1], axis=1)

    min_pred_index: np.ndarray = np.argmin(y_pred[:, :, 0], axis=1)
    max_pred_index: np.ndarray = np.argmax(y_pred[:, :, 1], axis=1)

    buy_order_pred: np.ndarray = np.all([max_pred_index > min_pred_index], axis=0)

    valid_actual: np.ndarray = np.all([max_true > min_true], axis=0)

    valid_pred: np.ndarray = np.all([max_pred > min_pred], axis=0)

    pred_average: np.ndarray = (max_pred + min_pred) / 2

    valid_min: np.ndarray = np.all([min_pred > min_true], axis=0)

    valid_max: np.ndarray = np.all([max_true > max_pred], axis=0)

    min_inside: np.ndarray = np.all(
        [
            max_true > min_pred,
            valid_min,
        ],
        axis=0,
    )

    max_inside: np.ndarray = np.all(
        [
            max_pred > min_true,
            valid_max,
        ],
        axis=0,
    )

    wins: np.ndarray = np.all(
        [
            min_inside,
            max_inside,
            valid_pred,
        ],
        axis=0,
    )

    average_in: np.ndarray = np.all(
        [
            max_true > pred_average,
            pred_average > min_true,
        ],
        axis=0,
    )

    simulation(min_pred, max_pred, buy_order_pred, y_true)

    fraction_valid_actual: float = np.mean(valid_actual.astype(np.float32))

    fraction_valid_pred: float = np.mean(valid_pred.astype(np.float32))

    fraction_max_inside: float = np.mean(max_inside.astype(np.float32))

    fraction_min_inside: float = np.mean(min_inside.astype(np.float32))

    fraction_average_in: float = np.mean(average_in.astype(np.float32))

    fraction_win: float = np.mean(wins.astype(np.float32))

    all_days_pro_arr: np.ndarray = (max_pred / min_pred) * wins.astype(np.float32)
    all_days_pro_arr_non_zero: np.ndarray = all_days_pro_arr[all_days_pro_arr != 0]

    all_days_pro_cummulative_val: float = np.prod(all_days_pro_arr_non_zero)

    pred_capture_arr: np.ndarray = (max_pred / min_pred - 1) * wins.astype(np.float32)

    total_capture_possible_arr: np.ndarray = max_true / min_true - 1

    pred_capture_ratio: float = np.sum(pred_capture_arr) / np.sum(total_capture_possible_arr)

    pred_capture_percent_str: str = "{:.2f}".format(pred_capture_ratio * 100)

    win_percent_str: str = "{:.2f}".format(fraction_win * 100)

    average_in_percent_str: str = "{:.2f}".format(fraction_average_in * 100)

    cdgr: float = pow(all_days_pro_cummulative_val, 1 / len(wins)) - 1

    pro_250: float = pow(cdgr + 1, 250) - 1
    pro_250_5: float = pow(cdgr * 5 + 1, 250) - 1
    pro_250_str: str = "{:.2f}".format(pro_250 * 100)
    pro_250_5_str: str = "{:.2f}".format(pro_250_5 * 100)

    y_min = min(np.min(min_pred), np.min(min_true))
    y_max = max(np.max(max_pred), np.max(max_true))

    x: list[int] = [i + 1 for i in range(len(max_pred))]

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)

    plt.axvline(x=int(len(max_true) * (1 - testsize)) - 0.5, color="blue")

    plt.fill_between(x, min_true, max_true, color="yellow")

    # plt.scatter(x, list_min_actual, color="orange", s=50)
    # plt.scatter(x, list_max_actual, color="orange", s=50)

    plt.plot(x, pred_average, linestyle="dashed", c="red")

    for i in range(len(wins)):
        if wins[i]:
            plt.scatter(
                x=x[i],
                y=y_min - (y_max - y_min) / 100,
                c="yellow",
                linewidths=2,
                marker="^",
                edgecolor="red",
                s=125,
            )

        if valid_pred[i]:
            plt.vlines(
                x=x[i],
                ymin=min_pred[i],
                ymax=max_pred[i],
                colors="green",
            )
            ax.set_xlabel("days", fontsize=15)

    ax.set_ylabel("fraction of prev close", fontsize=15)

    print("\n\n")
    print("valid_act\t", round(fraction_valid_actual * 100, 2), " %")
    print("valid_pred\t", round(fraction_valid_pred * 100, 2), " %")
    print("max_inside\t", round(fraction_max_inside * 100, 2), " %")
    print("min_inside\t", round(fraction_min_inside * 100, 2), " %\n")
    print("average_in\t", average_in_percent_str, " %\n")

    print("win_days_perc\t", win_percent_str, " %")
    print("pred_capture\t", pred_capture_percent_str, " %")

    print("per_day\t\t", round(cdgr * 100, 4), " %")
    print("250 days:\t", pro_250_str)
    print("\nleverage:\t", pro_250_5_str)
    print("datetime:\t", now_datetime)

    ax.set_title(
        f" name: {now_datetime} \n\n"
        + f" wins: {win_percent_str}% || "
        + f" average_in: {average_in_percent_str}% || "
        + f" 250 days: {pro_250_str}",
        fontsize=20,
    )

    filename = f"training/graphs/{y_type} - {now_datetime} - Splot.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    print("\n\nNUMBER_OF_NEURONS\t\t", km.NUMBER_OF_NEURONS)
    print("NUMBER_OF_LAYERS\t\t", km.NUMBER_OF_LAYERS)
    print("NUMBER_OF_EPOCHS\t\t", get_number_of_epochs())
    print("INITIAL_DROPOUT\t\t\t", km.INITIAL_DROPOUT)
    print("WEIGHT_FOR_MEA\t\t\t", km.WEIGHT_FOR_MEA)

    # plt.show()

    return


def simulation(
    buy_price_arr: np.ndarray,
    sell_price_arr: np.ndarray,
    order_type_buy_arr: list[bool],
    real_price_arr: np.ndarray,
) -> None:
    RISK_TO_REWARD_RATIO = 0.32
    # 1.48

    # 3 order are placed when the similation starts
    #   buy order
    #   sell order
    #   stop_loss_order based on what type of whole order this is - buy/sell
    #       or trend whether max comes first or the min.
    #
    #   when the last tick happends. any pending order is executed that that time.
    #       it will be either partial reward or partial stop_loss
    #
    #

    is_more_than_5: bool = False

    for RISK_TO_REWARD_RATIO in np.arange(0, 1.55, 0.05):
        wins_day_wise_list: list[float] = []

        number_of_days: int = real_price_arr.shape[0]

        trade_taken_list: np.array = np.zeros(number_of_days, dtype=bool)
        trade_taken_and_out_list: np.array = np.zeros(number_of_days, dtype=bool)
        stop_loss_hit_list: np.array = np.zeros(number_of_days, dtype=bool)
        completed_at_closing_list: np.array = np.zeros(number_of_days, dtype=bool)
        expected_trades_list: np.array = np.zeros(number_of_days, dtype=bool)

        for i_day in range(real_price_arr.shape[0]):
            trade_taken: bool = False
            trade_taken_and_out: bool = False
            stop_loss_hit: bool = False

            is_trade_type_buy: bool = order_type_buy_arr[i_day]

            buy_price: float = buy_price_arr[i_day]
            sell_price: float = sell_price_arr[i_day]
            stop_loss: float = 0

            expected_reward: float = sell_price - buy_price

            net_day_reward: float = 0

            # step 1 - find stop loss based on trade type
            if is_trade_type_buy:
                # pred is up
                stop_loss = buy_price - expected_reward * RISK_TO_REWARD_RATIO
            else:
                # pred is down
                stop_loss = sell_price + expected_reward * RISK_TO_REWARD_RATIO

            # step 2 - similated each tick
            for i_tick in range(real_price_arr.shape[1]):
                tick_min = real_price_arr[i_day, i_tick, 0]
                tick_max = real_price_arr[i_day, i_tick, 1]

                if is_trade_type_buy:
                    # buy trade
                    if not trade_taken:
                        if tick_min < buy_price and buy_price < tick_max:
                            trade_taken = True

                    if trade_taken and not trade_taken_and_out:
                        if tick_min < sell_price and sell_price < tick_max:
                            trade_taken_and_out = True
                            net_day_reward = expected_reward

                        elif tick_min < stop_loss and stop_loss < tick_max:
                            trade_taken_and_out = True
                            stop_loss_hit = True
                            stop_price = (tick_min + tick_max) / 2

                            net_day_reward = stop_price - buy_price

                else:
                    # sell trade
                    if not trade_taken:
                        if tick_min < sell_price and sell_price < tick_max:
                            trade_taken = True

                    if trade_taken and not trade_taken_and_out:
                        if tick_min < buy_price and buy_price < tick_max:
                            trade_taken_and_out = True
                            net_day_reward = expected_reward

                        elif tick_min < stop_loss and stop_loss < tick_max:
                            trade_taken_and_out = True
                            stop_loss_hit = True
                            stop_price = (tick_min + tick_max) / 2

                            net_day_reward = sell_price - stop_price

            if trade_taken and not trade_taken_and_out:
                if is_trade_type_buy:
                    # buy trade
                    avg_close = (real_price_arr[i_day, -1, 0] + real_price_arr[i_day, -1, 1]) / 2
                    net_day_reward = avg_close - buy_price

                else:
                    # sell trade
                    avg_close = (real_price_arr[i_day, -1, 0] + real_price_arr[i_day, -1, 1]) / 2
                    net_day_reward = sell_price - avg_close

            if trade_taken:
                trade_taken_list[i_day] = True
            if trade_taken_and_out:
                trade_taken_and_out_list[i_day] = True
            if stop_loss_hit:
                stop_loss_hit_list[i_day] = True
            if trade_taken and not trade_taken_and_out:
                completed_at_closing_list[i_day] = True
            if trade_taken_and_out and not stop_loss_hit:
                expected_trades_list[i_day] = True

            wins_day_wise_list.append(net_day_reward)

        # print("\n\n")
        # print("-" * 30)
        # print("\n\n")

        # count_trade_taken: int = np.sum(trade_taken_list)
        # count_trade_taken_and_out: int = np.sum(trade_taken_and_out_list)
        # count_stop_loss_hit: int = np.sum(stop_loss_hit_list)
        # count_completed_at_closing: int = np.sum(completed_at_closing_list)
        # count_expected_trades: int = np.sum(expected_trades_list)

        # print("number_of_days\t\t", number_of_days, "\n")

        # print("percent_trade_taken\t\t", "{:.2f}".format(count_trade_taken / number_of_days * 100), " %")
        # print(
        #     "percent_trade_taken_and_out\t",
        #     "{:.2f}".format(count_trade_taken_and_out / number_of_days * 100),
        #     " %",
        # )
        # print("percent_stop_loss_hit\t\t", "{:.2f}".format(count_stop_loss_hit / number_of_days * 100), " %")
        # print(
        #     "percent_completed_at_closing\t",
        #     "{:.2f}".format(count_completed_at_closing / number_of_days * 100),
        #     " %",
        # )

        # print("percent_expected_trades\t\t", "{:.2f}".format(count_expected_trades / number_of_days * 100), " %")

        # total_winings: float = sum(wins_day_wise_list)

        # number_of_win_trades: int = np.sum(np.array(wins_day_wise_list) > 0)

        # print("\nnumber_of_win_trades\t\t", "{:.4f}".format(number_of_win_trades / number_of_days), "\n")

        # x = np.arange(0, number_of_days, 1)

        # arr = np.array(wins_day_wise_list)
        # plt.scatter(x, arr, color="orange", s=50)

        # plt.plot(arr)

        # arr2 = np.array(stop_loss_hit_list) * (-0.002)
        # plt.plot(arr2)

        # total_winings_per_day: float = total_winings / number_of_days
        # print("\n\ntotal_winings_per_day\t\t", "{:.5f}".format(total_winings_per_day))
        # print("total_winings_per_day_leverage\t", "{:.5f}".format(total_winings_per_day * 5))

        # print("\n\n250_days\t\t\t", "{:.4f}".format(pow(1 + total_winings_per_day, 250) - 1))
        # print("250_days_leverage\t\t", "{:.4f}".format(pow(1 + total_winings_per_day * 5, 250) - 1))

        total_winings: float = sum(wins_day_wise_list)
        total_winings_per_day: float = total_winings / number_of_days

        z = pow(1 + total_winings_per_day, 250) - 1

        if z > 0.03:
            print(
                "{:.2f}".format(RISK_TO_REWARD_RATIO),
                "\t250_days\t\t",
                "{:.4f}".format(z * 100),
                "\t\t",
                "{:.4f}".format((pow(1 + total_winings_per_day * 5, 250) - 1) * 100),
            )

            is_more_than_5 = True

    if not is_more_than_5:
        print("not iteration more than 1 perc")
    return
