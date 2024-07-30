import os

import numpy as np
from keras.models import load_model
from keras.utils import custom_object_scope
from numpy.typing import NDArray
from tensorflow.keras.models import Model

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
        assert Y_data_real.ndim == 3, "Y Real Data array must be 3-dimensional"

        self.ticker = ticker

        self.now_datetime = now_datetime
        self.model_num = model_num
        self.model_location_type = model_location_type

        self.x_data = X_data
        self.y_data = Y_data
        self.y_data_real = Y_data_real

        self.prev_close = prev_close.reshape(len(prev_close))

        self.x_type = x_type
        self.y_type = y_type

        self.test_size = test_size

        # --------------------------------------------------
        # other attributes declared here

        self.model_file_path: str

        self.safety_factor = SAFETY_FACTOR

        self.number_of_days = self.x_data.shape[0]

        self.is_model_worth_saving: bool = False
        self.is_model_worth_double_saving: bool = False

        self.win_250_days: float = 0

        self._print_start_of_evaluation_message()

        self._set_model_file_path()

        self._fill_gaps_in_y_real_data()

    def _print_start_of_evaluation_message(self) -> None:
        if self.test_size > 0:
            print("TRAINING data now ...")
        else:
            print("\n" * 2, "_" * 140, "\n" * 2, sep="")
            print("VALIDATION data now ...")

    def _set_model_file_path(self) -> None:
        self.model_file_path = f"model - {self.now_datetime} - {self.x_type.value} - {self.y_type.value} - {self.ticker.name} - modelCheckPoint-{self.model_num}.keras"

        file_path: str = os.path.join(self.model_location_type.value, self.model_file_path)

        # check this file exists or not
        if not os.path.exists(file_path):
            raise ValueError(f"WARNING: file not found at: \n{file_path}\n")

        self.model_file_path = file_path

        return

    def load_model(self, custom_objects: dict | None = None) -> Model:
        if custom_objects is None:
            custom_objects = {}

        with custom_object_scope(custom_objects):
            model = load_model(self.model_file_path)

        return model

    def _fill_gaps_in_y_real_data(self) -> None:
        # for given 2 adjacent ticks.
        # close of the previous tick must be inside the min/open and max/close of the next tick

        for day in range(self.y_data_real.shape[0]):
            for tick in range(self.y_data_real.shape[1]):
                if tick == 0:
                    continue

                self.y_data_real[day, tick, 0] = self.y_data_real[day, tick, 1] = self.prev_close[day]

                # check of gap exists
                prev_tick_close = self.y_data_real[day, tick - 1, 1]
                next_tick_min = self.y_data_real[day, tick, 2]
                next_tick_max = self.y_data_real[day, tick, 1]

                if next_tick_min <= prev_tick_close <= next_tick_max:
                    continue

                # ohlc (open, high, low, close)
                # gap up
                if prev_tick_close < next_tick_min:
                    # next tick low and open = prev tick close

                    self.y_data_real[day, tick, 0] = prev_tick_close
                    self.y_data_real[day, tick, 3] = prev_tick_close

                # gap down
                if prev_tick_close > next_tick_max:
                    # next tick high and open = prev tick close

                    self.y_data_real[day, tick, 0] = prev_tick_close
                    self.y_data_real[day, tick, 1] = prev_tick_close

        return


# task - check for all real data for tickers
# that all open and close are inside min and max
# and that max is greater than min
