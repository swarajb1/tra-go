import os

import keras_model_tf as km_tf
import matplotlib.pyplot as plt
import numpy as np
from core.simulation import Simulation
from keras.models import load_model
from keras.utils import custom_object_scope
from numpy.typing import NDArray
from training_yf import round_to_nearest_0_05

import tra_go.band_2_1.keras_model as km_21_model
import tra_go.band_2_1.model_metrics as km_21_metrics
from database.enums import BandType, ModelLocationType, TickerOne

SAFETY_FACTOR: float = float(os.getenv("SAFETY_FACTOR"))


class CoreEvaluation:
    def __init__(
        self,
        ticker: TickerOne,
        X_data: NDArray[np.float64],
        Y_data: NDArray[np.float64],
        Y_data_real: NDArray[np.float64],
        prev_close: NDArray[np.float64],
        x_type: BandType,
        y_type: BandType,
        test_size: float,
        now_datetime: str,
        model_location_type: ModelLocationType,
        model_num: int,
    ):
        self.ticker = ticker

        self.now_datetime = now_datetime
        self.model_num = model_num
        self.model_location_type = model_location_type

        self.model_file_path: str

        self.x_data = X_data
        self.y_data = Y_data
        self.y_data_real = Y_data_real

        self.prev_close = prev_close.reshape(len(prev_close))

        self.x_type = x_type
        self.y_type = y_type

        self.test_size = test_size

        self.safety_factor = SAFETY_FACTOR

        self.number_of_days = self.x_data.shape[0]

        self.is_model_worth_saving: bool = False
        self.is_model_worth_double_saving: bool = False

        self.win_250_days: float = 0

        if self.test_size > 0:
            print("TRAINING data now ...")
        else:
            print("\n" * 2, "_" * 140, "\n" * 2, sep="")
            print("VALIDATION data now ...")

        self.set_model_file_path()

        self.custom_evaluate_safety_factor()

    def set_model_file_path(self) -> None:
        self.model_file_path = f"model - {self.now_datetime} - {self.x_type.value} - {self.y_type.value} - {self.ticker.name} - modelCheckPoint-{self.model_num}.keras"

        file_path: str = os.path.join(self.model_location_type.value, self.model_file_path)

        # check this file exists or not
        if not os.path.exists(file_path):
            raise ValueError(f"WARNING: file not found at: \n{file_path}\n")

        self.model_file_path = file_path

        return

    def load_model(self, custom_objects: dict | None = None):
        if custom_objects is None:
            custom_objects = {}

        with custom_object_scope(custom_objects):
            model = load_model(self.model_file_path)

        return model

    def custom_evaluate_safety_factor(self):
        with custom_object_scope(
            {
                "loss_function": km_21_metrics.loss_function,
                "metric_rmse_percent": km_tf.metric_rmse_percent,
                "metric_abs_percent": km_tf.metric_abs_percent,
                "metric_loss_comp_2": km_21_metrics.metric_loss_comp_2,
                "metric_win_percent": km_21_metrics.metric_win_percent,
                "metric_win_pred_capture_percent": km_21_metrics.metric_win_pred_capture_percent,
                "metric_win_correct_trend_percent": km_21_metrics.metric_win_correct_trend_percent,
                "metric_win_pred_trend_capture_percent": km_21_metrics.metric_win_pred_trend_capture_percent,
                "CustomActivationLayer": km_21_model.CustomActivationLayer,
                "metric_correct_trends_full": km_21_metrics.metric_correct_trends_full,
            },
        ):
            model = load_model(file_path)
            # model.summary()

        self.y_pred: NDArray = model.predict(self.x_data)

        if self.x_type == BandType.BAND_4:
            x_last_zone_close: NDArray = self.x_data[:, -1, 3]
        elif self.x_type == BandType.BAND_2:
            x_last_zone_close: NDArray = (self.x_data[:, -1, 0] + self.x_data[:, -1, 1]) / 2

        x_last_zone_close_real: NDArray = round_to_nearest_0_05(x_last_zone_close * self.prev_close)

        # (low, high) = (0, 1)

        self.y_pred_real = self.y_pred
        self.y_pred_real[:, 0] = round_to_nearest_0_05(self.y_pred[:, 0] * self.prev_close)
        self.y_pred_real[:, 1] = round_to_nearest_0_05(self.y_pred[:, 1] * self.prev_close)

        y_data_real_try = round_to_nearest_0_05(self.y_data * self.prev_close[:, np.newaxis, np.newaxis])

        self.function_make_win_graph(
            # y_true=self.y_data_real,
            y_true=y_data_real_try,
            y_pred=self.y_pred_real,
            x_close=x_last_zone_close_real,
        )

        return

    def correct_pred_values(self, y_arr: NDArray) -> NDArray:
        res = y_arr.copy()
        # step 1 - correct /exchange low/high values if needed., for each candle

        for i_day in range(res.shape[0]):
            for i_tick in range(res.shape[1]):
                if res[i_day, i_tick, 0] > res[i_day, i_tick, 1]:
                    # (low > high)
                    res[i_day, i_tick, 0], res[i_day, i_tick, 1] = (
                        res[i_day, i_tick, 1],
                        res[i_day, i_tick, 0],
                    )

        return res

    def function_make_win_graph(
        self,
        y_true: NDArray,
        y_pred: NDArray,
        x_close: NDArray,
    ):
        min_true: NDArray = np.min(y_true[:, :, 2], axis=1)
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

        self.is_model_worth_saving, self.is_model_worth_double_saving = simulation.get_is_worth_values()

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

        print("file_name\t", self.model_file_path)

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
