import matplotlib.pyplot as plt
import numpy as np
from core.simulation import Simulation
from numpy.typing import NDArray


class WinGraph:
    def __init__(
        self,
        max_pred: NDArray[np.float64],
        min_pred: NDArray[np.float64],
        order_type_buy: NDArray[np.bool_],
        y_real: NDArray[np.float64],
    ):
        assert y_real.ndim == 3, "Y Real Data array must be 3-dimensional"

        assert max_pred.ndim == 1, "Buy Price Data array must be 1-dimensional"
        assert min_pred.ndim == 1, "Sell Price Data array must be 1-dimensional"
        assert order_type_buy.ndim == 1, "Order Type Buy Data array must be 1-dimensional"

        assert np.issubdtype(order_type_buy.dtype, np.bool_), "Order Price Data array must be of type bool"

        self.y_data_real: NDArray[np.float64] = y_real

        self.max_pred: NDArray[np.float64] = max_pred
        self.min_pred: NDArray[np.float64] = min_pred
        self.order_type_buy: NDArray[np.bool_] = order_type_buy

        self.win_250_days: float = 0
        self.win_pred_capture_percent: float = 0

        self._make_win_graph()

    def _make_win_graph(self) -> None:
        min_true: NDArray = np.min(self.y_data_real[:, :, 2], axis=1)
        max_true: NDArray = np.max(self.y_data_real[:, :, 1], axis=1)

        pred_average: NDArray[np.float64] = (self.max_pred + self.min_pred) / 2

        valid_min: NDArray[np.bool_] = np.all([self.min_pred > min_true], axis=0)

        valid_max: NDArray[np.bool_] = np.all([max_true > self.max_pred], axis=0)

        min_inside: NDArray[np.bool_] = np.all(
            [
                max_true > self.min_pred,
                valid_min,
            ],
            axis=0,
        )

        max_inside: NDArray[np.bool_] = np.all(
            [
                self.max_pred > min_true,
                valid_max,
            ],
            axis=0,
        )

        wins: NDArray[np.bool_] = np.all(
            [
                min_inside,
                max_inside,
            ],
            axis=0,
        )

        average_in: NDArray[np.bool_] = np.all(
            [
                max_true > pred_average,
                pred_average > min_true,
            ],
            axis=0,
        )

        simulation = Simulation(
            buy_price_arr=self.min_pred,
            sell_price_arr=self.max_pred,
            order_type_buy_arr=self.order_type_buy,
            real_price_arr=self.y_data_real,
        )

        self.is_model_worth_saving, self.is_model_worth_double_saving = simulation.get_model_worthiness()

        fraction_max_inside: float = np.mean(max_inside.astype(np.float64))

        fraction_min_inside: float = np.mean(min_inside.astype(np.float64))

        fraction_average_in: float = np.mean(average_in.astype(np.float64))

        fraction_win: float = np.mean(wins.astype(np.float64))

        all_days_pro_arr: NDArray[np.float64] = (self.max_pred / self.min_pred) * wins.astype(np.float64)
        all_days_pro_arr_non_zero: NDArray[np.float64] = all_days_pro_arr[all_days_pro_arr != 0]

        all_days_pro_cumulative_value: float = np.prod(all_days_pro_arr_non_zero)

        pred_capture_arr: NDArray = (self.max_pred - self.min_pred) * wins.astype(np.float64)

        total_capture_possible_arr: NDArray = max_true - min_true

        pred_capture_ratio: float = np.sum(pred_capture_arr) / np.sum(total_capture_possible_arr)

        pred_capture_percent_str: str = "{:.2f}".format(pred_capture_ratio * 100)

        self.win_pred_capture_percent = float(pred_capture_percent_str)

        win_percent_str: str = "{:.2f}".format(fraction_win * 100)

        average_in_percent_str: str = "{:.2f}".format(fraction_average_in * 100)

        cdgr: float = pow(all_days_pro_cumulative_value, 1 / len(wins)) - 1

        pro_250: float = pow(cdgr + 1, 250) - 1
        pro_250_5: float = pow(cdgr * 5 + 1, 250) - 1
        pro_250_str: str = "{:.2f}".format(pro_250 * 100)
        pro_250_5_str: str = "{:.2f}".format(pro_250_5 * 100)

        self.win_250_days = round(pro_250 * 100, 2)

        # special condition, later make this with 'and' for worth double saving

        self.is_model_worth_saving &= fraction_win > 0.2

        self.is_model_worth_double_saving &= self.win_pred_capture_percent > 5

        if self.is_model_worth_saving:
            print("\n\nIs Model Worth Saving\t\t \033[92m+++\033[0m ")

        if self.is_model_worth_double_saving:
            print("\n\nIs Model Worth Double Saving\t \033[92m++++\033[0m ")

        print("\n")
        # print("Valid Pred:\t\t\t", round(fraction_valid_pred * 100, 2), " %")
        print("Max Inside:\t\t\t", round(fraction_max_inside * 100, 2), " %")
        print("Min Inside:\t\t\t", round(fraction_min_inside * 100, 2), " %\n")
        print("Average In:\t\t\t", average_in_percent_str, " %\n")

        print("Win Days Perc:\t\t\t", win_percent_str, " %")
        print("Pred Capture:\t\t\t", pred_capture_percent_str, " %")

        print("Per Day:\t\t\t", round(cdgr * 100, 3), " %")
        print("250 days:\t\t\t", pro_250_str)

        print("\n")
        # print("Leverage:\t\t\t", pro_250_5_str)
        # print("Datetime:\t\t\t", self.now_datetime)

        # print("\n\nNUMBER_OF_NEURONS\t\t", NUMBER_OF_NEURONS)
        # print("NUMBER_OF_LAYERS\t\t", NUMBER_OF_LAYERS)
        # print("NUMBER_OF_EPOCHS\t\t", NUMBER_OF_EPOCHS)
        # print("INITIAL_DROPOUT\t\t\t", INITIAL_DROPOUT)

        return

    def get_win_values(self):
        return self.win_pred_capture_percent, self.win_250_days

    def get_model_worthiness(self):
        return self.is_model_worth_saving, self.is_model_worth_double_saving

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
