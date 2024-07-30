import matplotlib.pyplot as plt
import numpy as np
from core.simulation import Simulation
from numpy.typing import NDArray


class WinGraph:
    def __init__(
        self,
        y_real: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ):
        assert y_real.ndim == 3, "Y Real Data array must be 3-dimensional"

        self.y_data_real: NDArray[np.float64] = y_real
        self.y_pred_real: NDArray[np.float64] = y_pred

        self.make_win_graph()

    def make_win_graph(self, y_true: NDArray, y_pred: NDArray, x_close: NDArray) -> None:
        min_true: NDArray = np.min(y_true[:, :, 0], axis=1)
        max_true: NDArray = np.max(y_true[:, :, 1], axis=1)

        min_pred: NDArray = y_pred[:, 0]
        max_pred: NDArray = y_pred[:, 1]

        buy_order_pred: NDArray = y_pred[:, 2]

        valid_pred: NDArray = np.all([max_pred > min_pred], axis=0)

        # step - using last close price of x data, and change min_pred and max_pred values, because of hypothesis that, the close price will be in the band
        close_below_band: NDArray = np.all([x_close <= min_pred], axis=0)

        close_above_band: NDArray = np.all([x_close >= max_pred], axis=0)

        close_in_band: NDArray = np.all(
            [
                x_close >= min_pred,
                x_close <= max_pred,
            ],
            axis=0,
        )

        for i in range(self.number_of_days):
            if not close_in_band[i] and valid_pred[i]:
                band_val: float = max_pred[i] - min_pred[i]

                if close_below_band[i]:
                    # buy_order_pred[i] = True
                    min_pred[i] = x_close[i]
                    max_pred[i] = x_close[i] + band_val

                elif close_above_band[i]:
                    # buy_order_pred[i] = False
                    min_pred[i] = x_close[i] - band_val
                    max_pred[i] = x_close[i]

        pred_average: NDArray = (max_pred + min_pred) / 2

        valid_min: NDArray = np.all([min_pred > min_true], axis=0)

        valid_max: NDArray = np.all([max_true > max_pred], axis=0)

        min_inside: NDArray = np.all(
            [
                max_true > min_pred,
                valid_min,
            ],
            axis=0,
        )

        max_inside: NDArray = np.all(
            [
                max_pred > min_true,
                valid_max,
            ],
            axis=0,
        )

        wins: NDArray = np.all(
            [
                min_inside,
                max_inside,
                valid_pred,
            ],
            axis=0,
        )

        average_in: NDArray = np.all(
            [
                max_true > pred_average,
                pred_average > min_true,
            ],
            axis=0,
        )

        simulation = Simulation(
            buy_price_arr=min_pred,
            sell_price_arr=max_pred,
            order_type_buy_arr=buy_order_pred,
            real_price_arr=y_true,
        )

        self.is_model_worth_saving, self.is_model_worth_double_saving = simulation.get_model_worthiness()

        fraction_valid_pred: float = np.mean(valid_pred.astype(np.float64))

        fraction_max_inside: float = np.mean(max_inside.astype(np.float64))

        fraction_min_inside: float = np.mean(min_inside.astype(np.float64))

        fraction_average_in: float = np.mean(average_in.astype(np.float64))

        fraction_win: float = np.mean(wins.astype(np.float64))

        all_days_pro_arr: NDArray = (max_pred / min_pred) * wins.astype(np.float64)
        all_days_pro_arr_non_zero: NDArray = all_days_pro_arr[all_days_pro_arr != 0]

        all_days_pro_cummulative_val: float = np.prod(all_days_pro_arr_non_zero)

        pred_capture_arr: NDArray = (max_pred / min_pred - 1) * wins.astype(
            np.float64,
        )

        total_capture_possible_arr: NDArray = max_true / min_true - 1

        pred_capture_ratio: float = np.sum(pred_capture_arr) / np.sum(
            total_capture_possible_arr,
        )

        pred_capture_percent_str: str = "{:.2f}".format(pred_capture_ratio * 100)

        win_percent_str: str = "{:.2f}".format(fraction_win * 100)

        average_in_percent_str: str = "{:.2f}".format(fraction_average_in * 100)
        # self.is_model_worth_double_saving = self.is_model_worth_double_saving and fraction_average_in > 0.5

        cdgr: float = pow(all_days_pro_cummulative_val, 1 / len(wins)) - 1

        pro_250: float = pow(cdgr + 1, 250) - 1
        pro_250_5: float = pow(cdgr * 5 + 1, 250) - 1
        pro_250_str: str = "{:.2f}".format(pro_250 * 100)
        pro_250_5_str: str = "{:.2f}".format(pro_250_5 * 100)

        self.win_250_days = round(pro_250 * 100, 2)

        print("\n\n")
        print(
            "Is Model Worth Double Saving\t",
            "\033[92m++++\033[0m" if self.is_model_worth_double_saving else "\033[91m----\033[0m",
        )

        print("\n\n")
        print("Valid Pred:\t\t\t", round(fraction_valid_pred * 100, 2), " %")
        print("Max Inside:\t\t\t", round(fraction_max_inside * 100, 2), " %")
        print("Min Inside:\t\t\t", round(fraction_min_inside * 100, 2), " %\n")
        print("Average In:\t\t\t", average_in_percent_str, " %\n")

        print("Win Days Perc:\t\t\t", win_percent_str, " %")
        print("Pred Capture:\t\t\t", pred_capture_percent_str, " %")

        print("Per Day:\t\t\t", round(cdgr * 100, 4), " %")
        print("250 days:\t\t\t", pro_250_str)
        print("\n")
        print("Leverage:\t\t\t", pro_250_5_str)
        print("Datetime:\t\t\t", self.now_datetime)
        # print("\n\nNUMBER_OF_NEURONS\t\t", NUMBER_OF_NEURONS)
        # print("NUMBER_OF_LAYERS\t\t", NUMBER_OF_LAYERS)
        # print("NUMBER_OF_EPOCHS\t\t", NUMBER_OF_EPOCHS)
        # print("INITIAL_DROPOUT\t\t\t", INITIAL_DROPOUT)

        print("file_name\t", self.model_file_name)

        return

    def day_candle_pred_true(self, i_day: int):
        error = self.y_data_real - self.y_pred_real

        y_l = error[i_day, :, 0]
        y_h = error[i_day, :, 1]

        x = np.arange(error.shape[1])

        plt.figure(figsize=(16, 9))
        plt.title(f"Day - {i_day}")

        plt.axhline(y=0, xmin=x[0], xmax=x[-1], color="blue")

        plt.plot(x, y_l, label="low Δ")
        plt.plot(x, y_h, label="high Δ")

        return None
