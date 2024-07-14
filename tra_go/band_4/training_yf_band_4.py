import os

import keras_model as km
import matplotlib.pyplot as plt
import numpy as np
from core.simulation import Simulation
from keras.models import load_model
from keras.utils import custom_object_scope
from numpy.typing import NDArray
from training_yf import round_to_nearest_0_05

import tra_go.band_4.keras_model_band_4 as km_4
from database.enums import BandType, ModelLocationType, TickerOne

SAFETY_FACTOR: float = float(os.getenv("SAFETY_FACTOR"))

NUMBER_OF_EPOCHS: int = int(os.getenv("NUMBER_OF_EPOCHS"))

NUMBER_OF_NEURONS: int = int(os.getenv("NUMBER_OF_NEURONS"))
NUMBER_OF_LAYERS: int = int(os.getenv("NUMBER_OF_LAYERS"))
INITIAL_DROPOUT_PERCENT: float = float(os.getenv("INITIAL_DROPOUT_PERCENT"))


class CustomEvaluation:
    def __init__(
        self,
        ticker: TickerOne,
        X_data: NDArray[np.float32],
        Y_data: NDArray[np.float32],
        prev_close: NDArray[np.float32],
        x_type: BandType,
        y_type: BandType,
        test_size: float,
        now_datetime: str,
        model_location_type: ModelLocationType,
        model_num: int = 1,
        skip_first_percentile: float = 0.18,
        skip_last_percentile: float = 0.18,
        safety_factor=0.8,
    ):
        self.ticker = ticker

        self.X_data = X_data
        self.Y_data = Y_data
        self.prev_close = prev_close.reshape(len(prev_close))

        self.x_type = x_type
        self.y_type = y_type

        self.test_size = test_size
        self.now_datetime = now_datetime
        self.model_num = model_num

        self.model_location_type = model_location_type

        self.safety_factor = SAFETY_FACTOR

        self.number_of_days = self.X_data.shape[0]

        self.is_model_worth_saving: bool = False
        self.is_model_worth_double_saving: bool = False

        self.win_250_days: float = 0

        print("\n" * 4, "*" * 200, "\n" * 4, sep="")

        if self.test_size > 0:
            print("TRAINING data now ...")
        else:
            print("VALIDATION data now ...")

        self.custom_evaluate_safety_factor()

    def custom_evaluate_safety_factor(self):
        self.model_file_name = f"model - {self.now_datetime} - {self.x_type.value} - {self.y_type.value} - {self.ticker.name} - modelCheckPoint-{self.model_num}.keras"

        file_path: str = os.path.join(self.model_location_type.value, self.model_file_name)

        if not os.path.exists(file_path):
            print(
                f"WARNING: file not found: \n{self.model_file_name} \nboth in training/models and training/models_saved",
            )
            return

        with custom_object_scope(
            {
                "metric_new_idea": km_4.metric_new_idea,
                "metric_rmse_percent": km.metric_rmse_percent,
                "metric_abs_percent": km.metric_abs_percent,
                "metric_loss_comp_2": km_4.metric_loss_comp_2,
                "metric_win_percent": km_4.metric_win_percent,
                "metric_win_pred_capture_percent": km_4.metric_win_pred_capture_percent,
                "metric_pred_capture_percent": km_4.metric_pred_capture_percent,
                "metric_correct_win_trend_percent": km_4.metric_correct_win_trend_percent,
                "metric_win_checkpoint": km_4.metric_win_checkpoint,
            },
        ):
            model = load_model(file_path)
            # model.summary()

        self.y_pred: NDArray = model.predict(self.X_data)
        print("self.y_pred.shape", self.y_pred.shape)

        if self.x_type == BandType.BAND_2:
            x_close: np.ndarray = (self.X_data[:, -1, 0] + self.X_data[:, -1, 1]) / 2
        elif self.x_type in [BandType.BAND_4, BandType.BAND_5]:
            x_close: np.ndarray = self.X_data[:, -1, 3]

        x_close_real: np.ndarray = round_to_nearest_0_05(x_close * self.prev_close)

        # open, high, low, close

        # Y_data = self.transform_y_array(y_arr=self.Y_data)

        self.y_pred_new = self.truncated_y_pred(y_arr=self.y_pred)

        # temporary - not correcting the pred values
        # self.y_pred_new = self.correct_pred_values(self.y_pred_new)

        self.y_pred_real = round_to_nearest_0_05(
            self.y_pred_new * self.prev_close[:, np.newaxis, np.newaxis],
        )
        self.Y_data_real = round_to_nearest_0_05(
            self.Y_data * self.prev_close[:, np.newaxis, np.newaxis],
        )

        y_pred_real_untruncated = round_to_nearest_0_05(
            self.y_pred * self.prev_close[:, np.newaxis, np.newaxis],
        )

        self.function_make_win_graph(
            y_true=self.Y_data_real,
            y_pred=self.y_pred_real,
            x_close=x_close_real,
        )

        # self.function_error_132_graph(
        #     y_true=self.Y_data_real,
        #     y_pred=y_pred_real_untruncated,
        # )

        return

    def truncated_y_pred(self, y_arr: np.ndarray) -> np.ndarray:
        first_non_eliminated_element_index: int = int(
            km_4.SKIP_FIRST_PERCENTILE * y_arr.shape[1],
        )
        last_non_eliminated_element_index: int = y_arr.shape[1] - int(km_4.SKIP_LAST_PERCENTILE * y_arr.shape[1]) - 1

        last_skipped_elements: int = int(km_4.SKIP_LAST_PERCENTILE * y_arr.shape[1])

        res: np.ndarray = y_arr.copy()

        for i in range(first_non_eliminated_element_index):
            res[:, i, :] = y_arr[:, first_non_eliminated_element_index, :]

        for i in range(last_skipped_elements):
            res[:, -1 * i, :] = y_arr[:, last_non_eliminated_element_index, :]

        if self.safety_factor < 1:
            res[:, :, 1] = (res[:, :, 1] + res[:, :, 2]) / 2 + (res[:, :, 1] - res[:, :, 2]) / 2 * self.safety_factor
            res[:, :, 2] = (res[:, :, 1] + res[:, :, 2]) / 2 - (res[:, :, 1] - res[:, :, 2]) / 2 * self.safety_factor

        return res

    def correct_pred_values(self, y_arr: np.ndarray) -> np.ndarray:
        res = y_arr.copy()
        # step 1 - correct /exchange low/high values if needed., for each candle

        for i_day in range(res.shape[0]):
            for i_tick in range(res.shape[1]):
                if res[i_day, i_tick, 2] > res[i_day, i_tick, 1]:
                    # (low > high)
                    res[i_day, i_tick, 2], res[i_day, i_tick, 1] = (
                        res[i_day, i_tick, 1],
                        res[i_day, i_tick, 2],
                    )

        # comment - step 2, in current environment not giving better results
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

    def function_make_win_graph(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        x_close: np.ndarray,
    ):
        min_true: np.ndarray = np.min(y_true[:, :, 2], axis=1)
        max_true: np.ndarray = np.max(y_true[:, :, 1], axis=1)

        min_pred: np.ndarray = np.min(y_pred[:, :, 2], axis=1)
        max_pred: np.ndarray = np.max(y_pred[:, :, 1], axis=1)

        min_pred_index: np.ndarray = np.argmin(y_pred[:, :, 2], axis=1)
        max_pred_index: np.ndarray = np.argmax(y_pred[:, :, 1], axis=1)

        buy_order_pred: np.ndarray = np.all([max_pred_index > min_pred_index], axis=0)

        valid_pred: np.ndarray = np.all([max_pred > min_pred], axis=0)

        close_below_band: np.ndarray = np.all([x_close <= min_pred], axis=0)

        close_above_band: np.ndarray = np.all([x_close >= max_pred], axis=0)

        close_in_band: np.ndarray = np.all(
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

        simulation = Simulation(
            buy_price_arr=min_pred,
            sell_price_arr=max_pred,
            order_type_buy_arr=buy_order_pred,
            real_price_arr=y_true,
        )

        self.is_model_worth_saving = simulation.is_worth_saving
        self.is_model_worth_double_saving = simulation.is_worth_double_saving

        fraction_valid_pred: float = np.mean(valid_pred.astype(np.float32))

        fraction_max_inside: float = np.mean(max_inside.astype(np.float32))

        fraction_min_inside: float = np.mean(min_inside.astype(np.float32))

        fraction_average_in: float = np.mean(average_in.astype(np.float32))

        fraction_win: float = np.mean(wins.astype(np.float32))

        all_days_pro_arr: np.ndarray = (max_pred / min_pred) * wins.astype(np.float32)
        all_days_pro_arr_non_zero: np.ndarray = all_days_pro_arr[all_days_pro_arr != 0]

        all_days_pro_cummulative_val: float = np.prod(all_days_pro_arr_non_zero)

        pred_capture_arr: np.ndarray = (max_pred / min_pred - 1) * wins.astype(
            np.float32,
        )

        total_capture_possible_arr: np.ndarray = max_true / min_true - 1

        pred_capture_ratio: float = np.sum(pred_capture_arr) / np.sum(
            total_capture_possible_arr,
        )

        pred_capture_percent_str: str = "{:.2f}".format(pred_capture_ratio * 100)

        win_percent_str: str = "{:.2f}".format(fraction_win * 100)

        average_in_percent_str: str = "{:.2f}".format(fraction_average_in * 100)

        cdgr: float = pow(all_days_pro_cummulative_val, 1 / len(wins)) - 1

        pro_250: float = pow(cdgr + 1, 250) - 1
        pro_250_5: float = pow(cdgr * 5 + 1, 250) - 1
        pro_250_str: str = "{:.2f}".format(pro_250 * 100)
        pro_250_5_str: str = "{:.2f}".format(pro_250_5 * 100)

        self.win_250_days = round(pro_250 * 100, 2)

        print("\n\n")
        print("valid_pred\t", round(fraction_valid_pred * 100, 2), " %")
        print("max_inside\t", round(fraction_max_inside * 100, 2), " %")
        print("min_inside\t", round(fraction_min_inside * 100, 2), " %\n")
        print("average_in\t", average_in_percent_str, " %\n")

        print("win_days_perc\t", win_percent_str, " %")
        print("pred_capture\t", pred_capture_percent_str, " %")

        print("per_day\t\t", round(cdgr * 100, 4), " %")
        print("250 days:\t", pro_250_str)
        print("\nleverage:\t", pro_250_5_str)
        print("datetime:\t", self.now_datetime)

        # print("\n\nNUMBER_OF_NEURONS\t\t", km.NUMBER_OF_NEURONS)
        # print("NUMBER_OF_LAYERS\t\t", km.NUMBER_OF_LAYERS)
        # print("NUMBER_OF_EPOCHS\t\t", get_number_of_epochs())
        # print("INITIAL_DROPOUT\t\t\t", km.INITIAL_DROPOUT)

        print("folder_name\t\t", self.model_file_name)

        return

    def function_error_132_graph(self, y_true: np.ndarray, y_pred: np.ndarray):
        error_a = np.abs(y_pred - y_true) / y_true

        new_array = np.empty(shape=(0, 4))

        # average error np array
        for i_tick in range(error_a.shape[1]):
            open = error_a[:, i_tick, 0].sum()
            high = error_a[:, i_tick, 1].sum()
            low = error_a[:, i_tick, 2].sum()
            close = error_a[:, i_tick, 3].sum()

            to_add_array = np.array(
                [
                    open / error_a.shape[0],
                    high / error_a.shape[0],
                    low / error_a.shape[0],
                    close / error_a.shape[0],
                ],
            )

            new_array = np.concatenate((new_array, np.array([to_add_array])), axis=0)

        y_o = new_array[:, 0] * 100
        y_h = new_array[:, 1] * 100
        y_l = new_array[:, 2] * 100
        y_c = new_array[:, 3] * 100

        # Create x-axis values
        x = np.arange(len(new_array))

        fig = plt.figure(figsize=(16, 9))

        plt.plot(x, y_o, label="open Δ")
        plt.plot(x, y_h, label="high Δ")
        plt.plot(x, y_l, label="low Δ")
        plt.plot(x, y_c, label="close Δ")

        plt.title(
            f" name: {self.now_datetime}\n"
            + f"NUMBER_OF_NEURONS = {NUMBER_OF_NEURONS}  "
            + f"NUMBER_OF_LAYERS = {NUMBER_OF_LAYERS}\n"
            + f"NUMBER_OF_EPOCHS = {NUMBER_OF_EPOCHS} | "
            + f"INITIAL_DROPOUT = {INITIAL_DROPOUT_PERCENT}",
            fontsize=20,
        )

        # Set labels and title
        plt.xlabel("serial", fontsize=15)
        plt.ylabel("perc", fontsize=15)
        plt.legend(fontsize=15)

        filename = f"training/graphs/{self.y_type} - {self.now_datetime} - abs - sf={self.safety_factor} - model_{self.model_num}.png"
        if self.test_size == 0:
            filename = filename[:-4] + "- valid.png"

        plt.savefig(filename, dpi=300, bbox_inches="tight")

        # NOTE: when you want to see graphs, you need to uncomment the following line
        # plt.show()

        return

    def function_make_win_graph_old(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        x_close: np.ndarray,
    ):
        min_true: np.ndarray = np.min(y_true[:, :, 2], axis=1)
        max_true: np.ndarray = np.max(y_true[:, :, 1], axis=1)

        min_pred: np.ndarray = np.min(y_pred[:, :, 2], axis=1)
        max_pred: np.ndarray = np.max(y_pred[:, :, 1], axis=1)

        min_pred_index: np.ndarray = np.argmin(y_pred[:, :, 2], axis=1)
        max_pred_index: np.ndarray = np.argmax(y_pred[:, :, 1], axis=1)

        buy_order_pred: np.ndarray = np.all([max_pred_index > min_pred_index], axis=0)

        valid_actual: np.ndarray = np.all([max_true > min_true], axis=0)

        valid_pred: np.ndarray = np.all([max_pred > min_pred], axis=0)

        close_below_band: np.ndarray = np.all([x_close <= min_pred], axis=0)

        close_above_band: np.ndarray = np.all([x_close >= max_pred], axis=0)

        close_in_band: np.ndarray = np.all(
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

        simulation = Simulation(
            buy_price_arr=min_pred,
            sell_price_arr=max_pred,
            order_type_buy_arr=buy_order_pred,
            real_price_arr=y_true,
        )

        self.is_model_worth_saving = simulation.is_worth_saving

        fraction_valid_actual: float = np.mean(valid_actual.astype(np.float32))

        fraction_valid_pred: float = np.mean(valid_pred.astype(np.float32))

        fraction_max_inside: float = np.mean(max_inside.astype(np.float32))

        fraction_min_inside: float = np.mean(min_inside.astype(np.float32))

        fraction_average_in: float = np.mean(average_in.astype(np.float32))

        fraction_win: float = np.mean(wins.astype(np.float32))

        all_days_pro_arr: np.ndarray = (max_pred / min_pred) * wins.astype(np.float32)
        all_days_pro_arr_non_zero: np.ndarray = all_days_pro_arr[all_days_pro_arr != 0]

        all_days_pro_cummulative_val: float = np.prod(all_days_pro_arr_non_zero)

        pred_capture_arr: np.ndarray = (max_pred / min_pred - 1) * wins.astype(
            np.float32,
        )

        total_capture_possible_arr: np.ndarray = max_true / min_true - 1

        pred_capture_ratio: float = np.sum(pred_capture_arr) / np.sum(
            total_capture_possible_arr,
        )

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

        # if self.test_size != 0:
        #     plt.axvline(x=int(len(max_true) * (1 - self.test_size)) - 0.5, color="blue")

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

        ax.set_ylabel("price", fontsize=15)

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
        print("datetime:\t", self.now_datetime)

        ax.set_title(
            f" name: {self.now_datetime} \n\n"
            + f" wins: {win_percent_str}% || "
            + f" average_in: {average_in_percent_str}% || "
            + f" 250 days: {pro_250_str}",
            fontsize=20,
        )

        filename = f"training/graphs/{self.y_type} - {self.now_datetime} - Splot - sf={self.safety_factor}.png"
        if self.test_size == 0:
            filename = (
                f"training/graphs/{self.y_type} - {self.now_datetime} - Splot - sf={self.safety_factor} - valid.png"
            )

        plt.savefig(filename, dpi=300, bbox_inches="tight")

        print("\n\nNUMBER_OF_NEURONS\t\t", NUMBER_OF_NEURONS)
        print("NUMBER_OF_LAYERS\t\t", NUMBER_OF_LAYERS)
        print("NUMBER_OF_EPOCHS\t\t", NUMBER_OF_EPOCHS)
        print("INITIAL_DROPOUT\t\t\t", INITIAL_DROPOUT_PERCENT)

        print("folder_name\t", self.model_file_name)

        # plt.show()

        return

    def day_candle_pred_true(self, i_day: int):
        error = self.Y_data_real - self.y_pred_real

        y_o = error[i_day, :, 0]
        y_h = error[i_day, :, 1]
        y_l = error[i_day, :, 2]
        y_c = error[i_day, :, 3]

        x = np.arange(error.shape[1])

        plt.figure(figsize=(16, 9))
        plt.title(f"Day - {i_day}")

        plt.axhline(y=0, xmin=x[0], xmax=x[-1], color="blue")

        plt.plot(x, y_o, label="open Δ")
        plt.plot(x, y_h, label="high Δ")
        plt.plot(x, y_l, label="low Δ")
        plt.plot(x, y_c, label="close Δ")

        return None
