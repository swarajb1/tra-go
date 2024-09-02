import os

import band_2_1.keras_model as km_21_model
import band_2_1.model_metrics as km_21_metrics
import keras_model_tf as km_tf
import numpy as np
from core.evaluation import CoreEvaluation
from numpy.typing import NDArray
from tensorflow.keras.models import Model
from training_yf import round_to_nearest_0_05

from database.enums import BandType, ModelLocationType, TickerOne

SAFETY_FACTOR: float = float(os.getenv("SAFETY_FACTOR"))

NUMBER_OF_EPOCHS: int = int(os.getenv("NUMBER_OF_EPOCHS"))


class CustomEvaluation(CoreEvaluation):
    def __init__(
        self,
        ticker: TickerOne,
        X_data: NDArray[np.float64],
        Y_data: NDArray[np.float64],
        Y_data_real: NDArray[np.float64],
        prev_day_close: NDArray[np.float64],
        x_type: BandType,
        y_type: BandType,
        test_size: float,
        model_file_name: str,
        model_location_type: ModelLocationType,
    ):
        super().__init__(
            ticker=ticker,
            X_data=X_data,
            Y_data=Y_data,
            Y_data_real=Y_data_real,
            prev_day_close=prev_day_close,
            x_type=x_type,
            y_type=y_type,
            test_size=test_size,
            model_file_name=model_file_name,
            model_location_type=model_location_type,
        )

        self.y_pred_real: NDArray
        self.x_last_zone_close_real: NDArray

        self.simulation_250_days: float = 0

        self.custom_evaluate_safety_factor()

    def custom_evaluate_safety_factor(self):
        custom_scope = {
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
        }

        model: Model = self.load_model(custom_scope)

        self.y_pred: NDArray = model.predict(self.x_data)

        x_last_zone_close: NDArray

        if self.x_type == BandType.BAND_4:
            x_last_zone_close = self.x_data[:, -1, 3]
        elif self.x_type == BandType.BAND_2:
            x_last_zone_close = (self.x_data[:, -1, 0] + self.x_data[:, -1, 1]) / 2
        elif self.x_type == BandType.BAND_1_CLOSE:
            x_last_zone_close = self.x_data[:, -1, 0]

        self.x_last_zone_close_real: NDArray = round_to_nearest_0_05(x_last_zone_close * self.prev_day_close)

        # (low, high) = (0, 1)

        self.y_pred_real = self.y_pred
        self.y_pred_real[:, 0] = round_to_nearest_0_05(self.y_pred[:, 0] * self.prev_day_close)
        self.y_pred_real[:, 1] = round_to_nearest_0_05(self.y_pred[:, 1] * self.prev_day_close)

        self.correct_pred_values()

        # self._apply_safety_factor_on_pred()

        self._correct_pred_values_based_on_last_close()

        min_pred: NDArray = self.y_pred_real[:, 0]
        max_pred: NDArray = self.y_pred_real[:, 1]

        buy_order_pred: NDArray[np.bool_] = self.y_pred_real[:, 2].astype(bool)

        self.generate_win_graph(
            max_pred=max_pred,
            min_pred=min_pred,
            buy_order_pred=buy_order_pred,
        )

        return

    def _apply_safety_factor_on_pred(self) -> None:
        if SAFETY_FACTOR == 1:
            return

        y_pred_average: NDArray = (self.y_pred_real[:, 0] + self.y_pred_real[:, 1]) / 2

        y_band_height: NDArray = self.y_pred_real[:, 1] - self.y_pred_real[:, 0]

        new_y_pred_low: NDArray = y_pred_average - y_band_height / 2 / SAFETY_FACTOR
        new_y_pred_high: NDArray = y_pred_average + y_band_height / 2 / SAFETY_FACTOR

        self.y_pred_real[:, 0] = round_to_nearest_0_05(new_y_pred_low)
        self.y_pred_real[:, 1] = round_to_nearest_0_05(new_y_pred_high)

        return

    def correct_pred_values(self):
        # step 1 - correct /exchange low/high values if needed., for each candle

        for i_day in range(self.y_pred_real.shape[0]):
            if self.y_pred_real[i_day, 0] > self.y_pred_real[i_day, 1]:
                # (low > high)

                (
                    self.y_pred_real[i_day, 0],
                    self.y_pred_real[i_day, 1],
                ) = (
                    self.y_pred_real[i_day, 1],
                    self.y_pred_real[i_day, 0],
                )

        return

    def _correct_pred_values_based_on_last_close(self):
        min_pred: NDArray = self.y_pred_real[:, 0]
        max_pred: NDArray = self.y_pred_real[:, 1]

        valid_pred: NDArray[np.bool_] = np.all([max_pred > min_pred], axis=0)

        # step - using last close price of x data, and change min_pred and max_pred values, because of hypothesis that, the close price will be in the band
        close_below_band: NDArray = np.all([self.x_last_zone_close_real <= min_pred], axis=0)

        close_above_band: NDArray = np.all([self.x_last_zone_close_real >= max_pred], axis=0)

        close_in_band: NDArray = np.all(
            [
                self.x_last_zone_close_real >= min_pred,
                self.x_last_zone_close_real <= max_pred,
            ],
            axis=0,
        )

        for i_day in range(self.number_of_days):
            if not close_in_band[i_day] and valid_pred[i_day]:
                band_val: float = max_pred[i_day] - min_pred[i_day]

                if close_below_band[i_day]:
                    # buy_order_pred[i] = True
                    self.y_pred_real[i_day, 0] = self.x_last_zone_close_real[i_day]
                    self.y_pred_real[i_day, 1] = self.x_last_zone_close_real[i_day] + band_val

                elif close_above_band[i_day]:
                    # buy_order_pred[i] = False
                    self.y_pred_real[i_day, 0] = self.x_last_zone_close_real[i_day] - band_val
                    self.y_pred_real[i_day, 1] = self.x_last_zone_close_real[i_day]

                del band_val
        del i_day

        return
