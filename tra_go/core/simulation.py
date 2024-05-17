import numpy as np


def simulation(
    buy_price_arr: np.ndarray,
    sell_price_arr: np.ndarray,
    order_type_buy_arr: np.array,
    real_price_arr: np.ndarray,
) -> bool:
    # 3 order are placed when the simulation starts
    #   buy order
    #   sell order
    #   stop_loss_order - based on what type of whole order this is - buy/sell
    #       or trend whether max comes first or the min.
    #
    #   when the last tick happens. any pending order remain, the position is closed at market price..
    #       it will be either partial reward or partial stop_loss
    #
    #

    PERCENT_250_DAYS: int = 1
    PERCENT_250_DAYS_WORTH_SAVING: int = 4

    is_worth_saving: bool = False

    print("simulation started....")

    for RISK_TO_REWARD_RATIO in np.arange(0, 1.1, 0.1):
        number_of_days: int = real_price_arr.shape[0]

        wins_day_wise_list: np.array = np.zeros(number_of_days)
        invested_day_wise_list: np.array = np.zeros(number_of_days)

        trade_taken_list: np.array = np.zeros(number_of_days, dtype=bool)
        trade_taken_and_out_list: np.array = np.zeros(number_of_days, dtype=bool)
        stop_loss_hit_list: np.array = np.zeros(number_of_days, dtype=bool)
        completed_at_closing_list: np.array = np.zeros(number_of_days, dtype=bool)
        expected_trades_list: np.array = np.zeros(number_of_days, dtype=bool)

        for i_day in range(number_of_days):
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

            # step 2 - similating each tick inside the interval
            for i_tick in range(real_price_arr.shape[1]):
                tick_min = real_price_arr[i_day, i_tick, 0]
                tick_max = real_price_arr[i_day, i_tick, 1]

                if is_trade_type_buy:
                    # buy trade
                    if not trade_taken:
                        if tick_min <= buy_price and buy_price <= tick_max:
                            trade_taken = True

                    if trade_taken and not trade_taken_and_out:
                        if tick_min <= sell_price and sell_price <= tick_max:
                            trade_taken_and_out = True
                            net_day_reward = expected_reward

                        elif tick_min <= stop_loss and stop_loss <= tick_max:
                            trade_taken_and_out = True
                            stop_loss_hit = True

                            net_day_reward = stop_loss - buy_price

                else:
                    # sell trade
                    if not trade_taken:
                        if tick_min <= sell_price and sell_price <= tick_max:
                            trade_taken = True

                    if trade_taken and not trade_taken_and_out:
                        if tick_min <= buy_price and buy_price <= tick_max:
                            trade_taken_and_out = True
                            net_day_reward = expected_reward

                        elif tick_min <= stop_loss and stop_loss <= tick_max:
                            trade_taken_and_out = True
                            stop_loss_hit = True

                            net_day_reward = sell_price - stop_loss

            # if trade is still active at closing time, then closing the position at the closing price of the interval.
            if trade_taken and not trade_taken_and_out:
                avg_close_price = (real_price_arr[i_day, -1, 0] + real_price_arr[i_day, -1, 1]) / 2
                if is_trade_type_buy:
                    # buy trade
                    net_day_reward = avg_close_price - buy_price

                else:
                    # sell trade
                    net_day_reward = sell_price - avg_close_price

            # each day's stats
            if trade_taken:
                trade_taken_list[i_day] = True
                if is_trade_type_buy:
                    invested_day_wise_list[i_day] = buy_price
                else:
                    invested_day_wise_list[i_day] = sell_price

            if trade_taken_and_out:
                trade_taken_and_out_list[i_day] = True
            if stop_loss_hit:
                stop_loss_hit_list[i_day] = True
            if trade_taken and not trade_taken_and_out:
                completed_at_closing_list[i_day] = True
            if trade_taken_and_out and not stop_loss_hit:
                expected_trades_list[i_day] = True

            wins_day_wise_list[i_day] = net_day_reward

        # print("\n\n")
        # print("-" * 30)
        # print("\n\n")

        # count_trade_taken: int = np.sum(trade_taken_list)
        # count_trade_taken_and_out: int = np.sum(trade_taken_and_out_list)
        # count_stop_loss_hit: int = np.sum(stop_loss_hit_list)
        # count_completed_at_closing: int = np.sum(completed_at_closing_list)
        # count_expected_trades: int = np.sum(expected_trades_list)

        # print("number_of_days\t\t\t", number_of_days, "\n")

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

        # number_of_win_trades: int = np.sum(np.array(wins_day_wise_list) > 0)

        # print("\npercent_win_trades\t\t", "{:.2f}".format(number_of_win_trades / number_of_days * 100), " %")

        new_invested_day_wise_list = np.copy(invested_day_wise_list)
        new_invested_day_wise_list[new_invested_day_wise_list == 0] = 1

        arr = np.array(wins_day_wise_list) * 100
        arr_real = arr / new_invested_day_wise_list

        # plt.figure(figsize=(16, 9))

        # x = np.arange(0, number_of_days, 1)
        # plt.scatter(x, arr_real, color="orange", s=50)

        # plt.plot(arr_real)

        # arr2 = np.array(stop_loss_hit_list) * (-0.2)
        # plt.plot(arr2)

        # filename = f"training/graphs/{self.y_type} - {self.now_datetime} - Splot - sf={self.SAFETY_FACTOR} - model_{self.model_num}.png"
        # if self.test_size == 0:
        #     filename = filename[:-4] + "- valid.png"

        # plt.savefig(filename, dpi=300, bbox_inches="tight")

        avg_win_per_day = np.mean(arr_real / 100)

        days_250: float = (pow(1 + avg_win_per_day, 250) - 1) * 100

        # print("\n\navg_percent_win_per_day\t\t", "{:.5f}".format(avg_win_per_day * 100), " %")
        # print("avg_percent_win_per_day_leverage", "{:.5f}".format(avg_win_per_day * 100 * 5), " %")

        # print("\n\n250_days\t\t\t", "{:.2f}".format(days_250), " %")
        # print("250_days_leverage\t\t", "{:.2f}".format((pow(1 + avg_win_per_day * 5, 250) - 1) * 100), " %")

        # if self.test_size != 0:
        #     print(f"\n work: data = VALIDATION DATA, model={self.model_num} \n")
        # else:
        #     print(f"\n work: data = TRAINING DATA, model={self.model_num} \n")

        if days_250 > PERCENT_250_DAYS:
            print(
                "\t\t",
                "risk_to_reward_ratio:",
                "{:.2f}".format(RISK_TO_REWARD_RATIO),
                "\t",
                "250_days_s: ",
                "{:.2f}".format(days_250),
                " %",
                "\t" * 2,
                "\033[92m++\033[0m" if days_250 > PERCENT_250_DAYS_WORTH_SAVING else "",
            )

        if days_250 > PERCENT_250_DAYS_WORTH_SAVING:
            is_worth_saving = True

    return is_worth_saving
