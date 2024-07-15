import os

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import kurtosis, skew
from utils.functions import round_num_str

from database.enums import ProcessedDataType

load_dotenv()


RISK_TO_REWARD_RATIO: float = os.getenv("RISK_TO_REWARD_RATIO")

PERCENT_250_DAYS: int = 1
PERCENT_250_DAYS_WORTH_SAVING: int = 25


class Simulation:
    def __init__(
        self,
        buy_price_arr: np.ndarray[float],
        sell_price_arr: np.ndarray[float],
        order_type_buy_arr: np.ndarray[bool],
        real_price_arr: np.ndarray[float],
    ):
        self.buy_price_arr = buy_price_arr
        self.sell_price_arr = sell_price_arr
        self.order_type_buy_arr = order_type_buy_arr
        self.real_price_arr = real_price_arr

        self.is_worth_saving: bool = False

        self.is_worth_double_saving: bool = False

        self.real_data_for_analysis: NDArray
        self.stoploss_data_for_analysis: NDArray
        self.stoploss_rrr_for_analysis: float

        self.real_mean: float
        self.expected_mean: float
        self.real_full_reward_mean: float

        self.simulation()

        self.set_real_full_reward_mean()

        self.display_stats()

    def simulation(self) -> bool:
        # 3 orders are placed when the simulation starts
        #   buy order
        #   sell order
        #   stop_loss_order - based on what type of whole order this is - buy/sell
        #       or trend whether max comes first or the min.
        #
        #   when the last tick happens. any pending orders that remain, the position is closed at market price.
        #       it will be either partial reward or partial stop_loss

        print("simulation started....")

        for RISK_TO_REWARD_RATIO in np.arange(0, 1.1, 0.1):
            number_of_days: int = self.real_price_arr.shape[0]

            wins_day_wise_list: np.array = np.zeros(number_of_days)
            invested_day_wise_list: np.array = np.zeros(number_of_days)

            expected_reward_percent_day_wise_list: np.array = np.zeros(number_of_days)

            trade_taken_list: np.array = np.zeros(number_of_days, dtype=bool)
            trade_taken_and_out_list: np.array = np.zeros(number_of_days, dtype=bool)
            stop_loss_hit_list: np.array = np.zeros(number_of_days, dtype=bool)
            completed_at_closing_list: np.array = np.zeros(number_of_days, dtype=bool)
            expected_trades_list: np.array = np.zeros(number_of_days, dtype=bool)

            for i_day in range(number_of_days):
                trade_taken: bool = False
                trade_taken_and_out: bool = False
                stop_loss_hit: bool = False

                is_trade_type_buy: bool = self.order_type_buy_arr[i_day]

                buy_price: float = self.buy_price_arr[i_day]
                sell_price: float = self.sell_price_arr[i_day]
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
                for i_tick in range(self.real_price_arr.shape[1]):
                    tick_min = self.real_price_arr[i_day, i_tick, 0]
                    tick_max = self.real_price_arr[i_day, i_tick, 1]

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
                    avg_close_price = (self.real_price_arr[i_day, -1, 0] + self.real_price_arr[i_day, -1, 1]) / 2
                    if is_trade_type_buy:
                        # buy trade
                        net_day_reward = avg_close_price - buy_price

                    else:
                        # sell trade
                        net_day_reward = sell_price - avg_close_price

                # each day's stats
                trade_taken_list[i_day] = trade_taken

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
                expected_reward_percent_day_wise_list[i_day] = expected_reward / invested_day_wise_list[i_day] * 100

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

            if days_250 > PERCENT_250_DAYS_WORTH_SAVING:
                self.is_worth_saving = True

                if round(RISK_TO_REWARD_RATIO, 1) <= 0.5:
                    self.is_worth_double_saving = True

            if round(RISK_TO_REWARD_RATIO, 1) == 0.5:
                self.real_data_for_analysis = arr_real
                self.stoploss_data_for_analysis = expected_reward_percent_day_wise_list * 1
                self.stoploss_rrr_for_analysis = 1

            percent_prefix: str = " " if round(days_250, 2) < 10 else ""
            percent_val: str = percent_prefix + "{:.2f}".format(days_250) + " %"

            print(
                "\t\t",
                "risk_to_reward_ratio:",
                "{:.2f}".format(RISK_TO_REWARD_RATIO),
                "\t",
                "250_days_s: ",
                percent_val if days_250 > PERCENT_250_DAYS else "\t  --",
                "\t" * 2,
                "\033[92m++\033[0m" if self.is_worth_saving else "",
            )

    def display_stats(self) -> None:
        if not self.is_worth_saving:
            return

        print("\n\n\n", "-" * 30, "\nReal End of Data Stats\n")
        self.log_statistics(self.real_data_for_analysis, ProcessedDataType.REAL)

        print("\n\n\n", "-" * 30, f"\nStop Loss Data Stats , RRR = {self.stoploss_rrr_for_analysis}\n")
        self.log_statistics(self.stoploss_data_for_analysis, ProcessedDataType.EXPECTED_REWARD)

        print("\n\nCapture Return Percent:\t\t", "{:.2f}".format(self.real_mean / self.expected_mean * 100), " %")

    def log_statistics(self, arr: NDArray, data_type: ProcessedDataType) -> None:
        sorted_arr = np.sort(arr)

        print("Count: \t\t\t\t", np.size(sorted_arr))

        # Central Tendency
        mean = np.mean(sorted_arr)
        if data_type == ProcessedDataType.REAL:
            self.real_mean = mean
        elif data_type == ProcessedDataType.EXPECTED_REWARD:
            self.expected_mean = mean

        median = np.median(sorted_arr)
        stats.mode(np.round(sorted_arr, 2))

        std_deviation = np.std(sorted_arr)

        print("Mean: \t\t\t\t", round_num_str(mean, 4))
        print("Median: \t\t\t", round_num_str(median, 4))

        # Dispersion
        Q3, Q1 = np.percentile(sorted_arr, [75, 25])
        iq_range = Q3 - Q1
        print("Standard Deviation: \t\t", round_num_str(std_deviation, 4))
        print("Interquartile Range (IQR): \t", round_num_str(iq_range, 4))
        print("Min: \t\t\t\t", round_num_str(np.min(sorted_arr), 4))
        print("Max: \t\t\t\t", round_num_str(np.max(sorted_arr), 4))
        print("Peak to peak: \t\t\t", round_num_str(np.ptp(sorted_arr), 4))

        coefficient_of_variation = std_deviation / mean

        print("Coefficient of Variation: \t", round_num_str(coefficient_of_variation, 4))

        # Measures of Position
        z_scores = stats.zscore(sorted_arr)
        stats.rankdata(sorted_arr)

        if len(z_scores) > 6:
            z_scores_small = np.concatenate([sorted_arr[:3], sorted_arr[-3:]])
            print("Z-Scores: \t\t\t", [round_num_str(x, 3) for x in z_scores_small])
        else:
            print("Z-Scores: \t\t\t", [round_num_str(x, 3) for x in z_scores])

        print("Kurtosis: \t\t\t", round_num_str(kurtosis(sorted_arr), 4))
        print("Skewness: \t\t\t", round_num_str(skew(sorted_arr), 4))

        # print("Ranks: \t\t\t\t", ranks)

        # Normality Tests
        shapiro_test = stats.shapiro(sorted_arr)
        kolmogorov_smirnov_test = stats.kstest(sorted_arr, "norm")

        print("Shapiro-Wilk Test: \t\t", shapiro_test)
        print("Kolmogorov-Smirnov Test: \t", kolmogorov_smirnov_test)

        # Outliers and Anomalies
        anomalies = sorted_arr[np.abs(z_scores) > 2]
        outliers = sorted_arr[np.abs(z_scores) > 3]

        anomalies_percent = np.size(anomalies) / np.size(sorted_arr) * 100
        outliers_percent = np.size(outliers) / np.size(sorted_arr) * 100

        print("Anomalies: \t\t\t", round_num_str(anomalies_percent, 2), "%")
        print("Outliers: \t\t\t", round_num_str(outliers_percent, 2), "%")

        return

    def set_real_full_reward_mean(self) -> None:
        number_of_days: int = self.real_price_arr.shape[0]

        invested_day_wise_list: np.array = np.zeros(number_of_days)

        real_order_type_buy: np.array = np.zeros(number_of_days)

        real_full_reward_percent_day_wise_list: np.array = np.zeros(number_of_days)

        for i_day in range(number_of_days):
            net_day_reward: float = 0

            min_ticker_price: float = np.min(self.real_price_arr[i_day, :, 0])
            max_ticker_price: float = np.max(self.real_price_arr[i_day, :, 1])

            full_reward = max_ticker_price - min_ticker_price

            real_order_type_buy[i_day] = np.argmax(self.real_price_arr[i_day, :, 1]) > np.argmax(
                self.real_price_arr[i_day, :, 0],
            )

            invested_day_wise_list[i_day] = (
                self.buy_price_arr[i_day] if real_order_type_buy[i_day] else self.sell_price_arr[i_day]
            )

            real_full_reward_percent_day_wise_list[i_day] = full_reward / invested_day_wise_list[i_day] * 100

        self.real_full_reward_mean = np.mean(real_full_reward_percent_day_wise_list)
