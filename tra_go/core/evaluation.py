import os
from copy import deepcopy
from pathlib import Path

import numpy as np
from core.win_graph import WinGraph
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
        prev_day_close: NDArray[np.float64],
        x_type: BandType,
        y_type: BandType,
        test_size: float,
        model_file_name: str,
        model_location_type: ModelLocationType,
    ):
        assert Y_data_real.ndim == 3, "Y Real data array must be 3-dimensional"
        assert Y_data_real.shape[2] == 4, "Y Real data .shape[2] must be 4"
        assert prev_day_close.ndim == 1, "Prev Close data array must be 1-dimensional"

        self.ticker = ticker

        self.x_data = X_data
        self.y_data = Y_data
        self.y_data_real = Y_data_real

        self.x_type = x_type
        self.y_type = y_type

        self.test_size = test_size

        self.model_location_type = model_location_type
        self.model_file_name = model_file_name

        self.prev_day_close = prev_day_close

        # --------------------------------------------------
        # other properties declared here

        self.number_of_days: int = self.x_data.shape[0]

        self.is_model_worth_saving: bool = False
        self.is_model_worth_double_saving: bool = False

        self.win_250_days: float = 0
        self.win_pred_capture_percent: float = 0

        self._print_start_of_evaluation_message()

        self.model_file_path: Path
        self._set_model_file_path()

        self._fill_gaps_in_y_real_data()

    def _print_start_of_evaluation_message(self) -> None:
        if self.test_size > 0:
            print("TRAINING data now ...")
        else:
            print("\n" * 2, "_" * 140, "\n" * 2, sep="")
            print("VALIDATION data now ...")

    def _set_model_file_path(self) -> None:
        self.model_file_path: Path = Path(self.model_location_type.value) / self.model_file_name

        if not Path.exists(self.model_file_path):
            raise (f"WARNING: file not found at: \n{self.model_file_path}\n")

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

        deepcopy(self.y_data_real)

        print("\nstep: Filling gaps in Y Real data ...")
        # fill gaps of the previous tick based on forward tick.

        for day in range(self.y_data_real.shape[0]):
            for tick in range(self.y_data_real.shape[1]):
                if tick == 0:
                    # no adjustment for the first tick
                    continue

                # check of gap exists
                prev_tick_min = self.y_data_real[day, tick - 1, 2]
                prev_tick_max = self.y_data_real[day, tick - 1, 1]

                next_tick_min = self.y_data_real[day, tick, 2]
                next_tick_max = self.y_data_real[day, tick, 1]

                if next_tick_min <= prev_tick_min <= next_tick_max or next_tick_min <= prev_tick_max <= next_tick_max:
                    # no gap condition
                    continue

                # ohlc (open, high, low, close)

                # # type of gap - gap up
                # if prev_tick_close < next_tick_min:
                #     # next tick low and open = prev tick close

                #     self.y_data_real[day, tick, 0] = prev_tick_close
                #     self.y_data_real[day, tick, 3] = prev_tick_close

                # # type of gap - gap down
                # if prev_tick_close > next_tick_max:
                #     # next tick high and open = prev tick close

                #     self.y_data_real[day, tick, 0] = prev_tick_close
                #     self.y_data_real[day, tick, 1] = prev_tick_close

        return

    def generate_win_graph(self, max_pred, min_pred, buy_order_pred) -> None:
        win_graph = WinGraph(
            max_pred=max_pred,
            min_pred=min_pred,
            order_type_buy=buy_order_pred,
            y_real=self.y_data_real,
        )

        # self.win_250_days, self.win_pred_capture_percent = win_graph.get_win_values()
        # self.is_model_worth_saving, self.is_model_worth_double_saving = win_graph.get_model_worthiness()

        copy_attributes: list[str] = [
            "is_model_worth_saving",
            "is_model_worth_double_saving",
            "win_250_days",
            "win_pred_capture_percent",
            "simulation_250_days",
        ]

        for attr in copy_attributes:
            setattr(self, attr, getattr(win_graph, attr))

        print("file name:\t\t\t", self.model_file_name)

        return


# task - check for all real data for tickers
# that all open and close are inside min and max
# and that max is greater than min
