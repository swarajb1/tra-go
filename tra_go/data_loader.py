import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from training_zero import (
    NUMBER_OF_POINTS_IN_ZONE_1_ST,
    NUMBER_OF_POINTS_IN_ZONE_2_ND,
    NUMBER_OF_POINTS_IN_ZONE_DAY,
    TOTAL_POINTS_IN_ONE_DAY,
)

from database.enums import BandType, IntervalType, IODataType, TickerDataType, TickerOne

PROJECT_FOLDER_NAME: str = "tra-go"


class DataLoader:
    def __init__(
        self,
        ticker: TickerOne,
        interval: IntervalType,
        x_type: BandType,
        y_type: BandType,
        test_size: float = 0.2,
    ):
        self.ticker = ticker
        self.interval = interval
        self.x_type = x_type
        self.y_type = y_type
        self.test_size = test_size

        # ----------------------------------------

        self.train_y_real_data: NDArray[np.float64]
        self.test_y_real_data: NDArray[np.float64]

        self.load_data_y_real()

        self.train_x_data: NDArray[np.float64]
        self.train_y_data: NDArray[np.float64]

        self.test_x_data: NDArray[np.float64]
        self.test_y_data: NDArray[np.float64]

        self.load_data_y_for_training()

        self.train_prev_close: NDArray[np.float64]
        self.test_prev_close: NDArray[np.float64]

        self.load_data_prev_close()

    def _get_columns(self, ticker_data_type: TickerDataType) -> list[str]:
        columns: list[str]

        base_columns: list[str] = ["open", "high", "low", "close", "volume"]

        if ticker_data_type == TickerDataType.TRAINING:
            columns = base_columns + ["real_close"]
        elif ticker_data_type == TickerDataType.REAL_AND_CLEANED:
            columns = base_columns

        return columns

    def _get_full_data_df(self, ticker_data_type: TickerDataType) -> pd.DataFrame:
        # file_path: str = f"./data_cleaned/{self.interval.value}/{self.ticker.value} - {self.interval.value}.csv"
        file_path: str

        if ticker_data_type == TickerDataType.TRAINING:
            file_path = os.path.join(
                os.getcwd(),
                f"data_{ticker_data_type.value}",
                f"{self.ticker.value} - {self.interval.value}.csv",
            )

        elif ticker_data_type == TickerDataType.REAL_AND_CLEANED:
            file_path = os.path.join(
                os.getcwd(),
                f"data_{ticker_data_type.value}",
                self.interval.value,
                f"{self.ticker.value} - {self.interval.value}.csv",
            )

        df = pd.read_csv(file_path)

        return df

    def data_inside_zone(self, ticker_data_type: TickerDataType) -> pd.DataFrame:
        df = self._get_full_data_df(ticker_data_type)

        res_df = pd.DataFrame()

        # # when taking from 915 (165, 165)
        # initial_index_offset: int = 0

        # when taking from 1000  (132, 132)
        # initial_index_offset: int = 47

        # when taking from 951 (150, 150)
        # initial_index_offset: int = 36

        initial_index_offset: int

        if NUMBER_OF_POINTS_IN_ZONE_2_ND == 132:
            initial_index_offset = 47
        elif NUMBER_OF_POINTS_IN_ZONE_2_ND == 150:
            initial_index_offset = 36

        number_of_days: int = len(df) // TOTAL_POINTS_IN_ONE_DAY

        assert (
            len(df) % TOTAL_POINTS_IN_ONE_DAY == 0
        ), f"Full {TickerDataType.key} Data length is not divisible by TOTAL_POINTS_IN_ONE_DAY"

        for day in range(number_of_days):
            start_index: int = day * TOTAL_POINTS_IN_ONE_DAY + initial_index_offset
            end_index: int = start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

            res_df = pd.concat([res_df, df.iloc[start_index : end_index + 1]])

        res_df.reset_index(drop=True, inplace=True)

        # columns: list[str] = self._get_columns(ticker_data_type)

        # return res_df[columns]
        return res_df

    def data_split_x_y(self, df: pd.DataFrame, ticker_data_type: TickerDataType | None) -> pd.DataFrame:
        """Splits the data into input, output dataframe."""

        # assumptions:
        # 1. df is inside zone data

        df_i = pd.DataFrame()
        df_o = pd.DataFrame()

        number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

        assert (
            len(df) % NUMBER_OF_POINTS_IN_ZONE_DAY == 0
        ), "Full Zone Data length is not divisible by NUMBER_OF_POINTS_IN_ZONE_DAY"

        for day in range(number_of_days):
            day_start_index: int = int(day * NUMBER_OF_POINTS_IN_ZONE_DAY)
            day_end_index: int = day_start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

            first_2_nd_zone_index: int = int(day_start_index + NUMBER_OF_POINTS_IN_ZONE_1_ST)

            df_i = pd.concat([df_i, df.iloc[day_start_index:first_2_nd_zone_index]])
            df_o = pd.concat([df_o, df.iloc[first_2_nd_zone_index : day_end_index + 1]])

        df_i.reset_index(drop=True, inplace=True)
        df_o.reset_index(drop=True, inplace=True)

        df_i["close_average"] = (df_i["close"] + df_i["open"] + df_i["high"] + df_i["low"]) / 4
        df_o["close_average"] = (df_o["close"] + df_o["open"] + df_o["high"] + df_o["low"]) / 4

        # df_i["close_average"] = (df_i["high"] + df_i["low"] + df_i["close"]) / 3
        # df_o["close_average"] = (df_o["high"] + df_o["low"] + df_o["close"]) / 3

        assert (
            len(df_i) % NUMBER_OF_POINTS_IN_ZONE_1_ST == 0
        ), "Input Dataframe length is not divisible by NUMBER_OF_POINTS_IN_ZONE_1_ST"

        assert (
            len(df_o) % NUMBER_OF_POINTS_IN_ZONE_2_ND == 0
        ), "Output Dataframe length is not divisible by NUMBER_OF_POINTS_IN_ZONE_2_ND"

        columns_x: list[str]
        columns_y: list[str]

        band_columns: dict[BandType, list[str]] = {
            BandType.BAND_1_CLOSE: ["close_average"],
            BandType.BAND_1_1: ["close_average"],
            BandType.BAND_2: ["low", "high"],
            BandType.BAND_2_1: ["low", "high"],
            BandType.BAND_4: ["open", "high", "low", "close"],
            BandType.BAND_5: ["open", "high", "low", "close", "volume"],
        }

        if ticker_data_type == TickerDataType.REAL_AND_CLEANED:
            columns_x = band_columns[BandType.BAND_4]
            columns_y = band_columns[BandType.BAND_4]
        else:
            columns_x = band_columns[self.x_type]
            columns_y = band_columns[self.y_type]

        return df_i[columns_x], df_o[columns_y]

    def load_data_y_real(self) -> None:
        ticker_data_type = TickerDataType.REAL_AND_CLEANED

        df_train, df_test = self.data_split_train_test(ticker_data_type=ticker_data_type)

        df_train_x, df_train_y = self.data_split_x_y(df=df_train, ticker_data_type=ticker_data_type)

        df_test_x, df_test_y = self.data_split_x_y(df=df_test, ticker_data_type=ticker_data_type)

        train_y = by_date_df_array(df_train_y, band_type=BandType.BAND_4, io_type=IODataType.OUTPUT_DATA)
        test_y = by_date_df_array(df_test_y, band_type=BandType.BAND_4, io_type=IODataType.OUTPUT_DATA)

        self.train_y_real_data, self.test_y_real_data = train_y, test_y

        return

    def get_real_y_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # print("\ntrain_data - y real")
        # check_gaps(self.train_y_real_data)

        # print("test_data - y real")
        # check_gaps(self.test_y_real_data)

        return self.train_y_real_data, self.test_y_real_data

    def data_split_train_test(self, ticker_data_type: TickerDataType) -> tuple[pd.DataFrame, pd.DataFrame]:
        # split into train and test

        df = self.data_inside_zone(ticker_data_type)

        assert (
            len(df) % NUMBER_OF_POINTS_IN_ZONE_DAY == 0
        ), "Full Zone Data length is not divisible by NUMBER_OF_POINTS_IN_ZONE_DAY"

        number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

        train_days: int = int(number_of_days * (1 - self.test_size))

        first_test_index: int = train_days * NUMBER_OF_POINTS_IN_ZONE_DAY

        train_df = df.iloc[:first_test_index]
        test_df = df.iloc[first_test_index:]

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        columns: list[str] = self._get_columns(ticker_data_type)

        return train_df[columns], test_df[columns]

    def load_data_y_for_training(self) -> None:
        ticker_data_type = TickerDataType.TRAINING

        df_train, df_test = self.data_split_train_test(ticker_data_type=ticker_data_type)

        df_train_x, df_train_y = self.data_split_x_y(df=df_train, ticker_data_type=ticker_data_type)

        df_test_x, df_test_y = self.data_split_x_y(df=df_test, ticker_data_type=ticker_data_type)

        train_x = by_date_df_array(df_train_x, band_type=self.x_type, io_type=IODataType.INPUT_DATA)
        test_x = by_date_df_array(df_test_x, band_type=self.x_type, io_type=IODataType.INPUT_DATA)

        train_y = by_date_df_array(df_train_y, band_type=self.y_type, io_type=IODataType.OUTPUT_DATA)
        test_y = by_date_df_array(df_test_y, band_type=self.y_type, io_type=IODataType.OUTPUT_DATA)

        if self.y_type == BandType.BAND_2_1 or self.y_type == BandType.BAND_1_1:
            train_y = self.df_data_into_3_feature_array(train_y)
            test_y = self.df_data_into_3_feature_array(test_y)

        self.train_x_data = train_x
        self.train_y_data = train_y

        self.test_x_data = test_x
        self.test_y_data = test_y

        return

    def load_data_prev_close(self) -> None:
        ticker_data_type = TickerDataType.TRAINING

        df = self.data_inside_zone(ticker_data_type)

        prev_close: list[float] = []

        number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

        for day in range(number_of_days):
            day_start_index: int = int(day * NUMBER_OF_POINTS_IN_ZONE_DAY)

            prev_close.append(df.iloc[day_start_index, df.columns.get_loc("real_close")])

        number_of_days_train: int = int(number_of_days * (1 - self.test_size))

        self.train_prev_close = np.array(prev_close[:number_of_days_train])
        self.test_prev_close = np.array(prev_close[number_of_days_train:])

        assert (
            self.train_prev_close.shape[0] == self.train_x_data.shape[0]
        ), "train_prev_close length and train_x_data.shape[0] are not equal"

        assert (
            self.test_prev_close.shape[0] == self.test_x_data.shape[0]
        ), "test_prev_close length and test_x_data.shape[0] are not equal"

        return

    def get_train_test_split_data(
        self,
    ) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]]:
        assert (
            self.train_x_data.shape[0] == self.train_y_data.shape[0]
        ), "train_x_data and train_y_data  .shape[0] (days) are not equal"

        assert (
            self.test_x_data.shape[0] == self.test_y_data.shape[0]
        ), "test_x_data and test_y_data  .shape[0] (days) are not equal"

        return (
            (self.train_x_data, self.train_y_data),
            (self.test_x_data, self.test_y_data),
        )

    def get_prev_close_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        assert self.train_prev_close.ndim == 1, "train_prev_close is not 1D"
        assert self.test_prev_close.ndim == 1, "test_prev_close is not 1D"

        return (self.train_prev_close, self.test_prev_close)

    def df_data_into_3_feature_array(self, arr: NDArray) -> NDArray:
        res = np.zeros((arr.shape[0], 3))

        min_values: NDArray
        max_values: NDArray
        is_buy_trend: NDArray

        if self.y_type == BandType.BAND_2_1:
            assert arr.shape[2] == 2, "Array.shape[2] is not equal to 2, the data is not coming in the form of BAND_2"

            min_values = np.min(arr[:, :, 0], axis=1)
            max_values = np.max(arr[:, :, 1], axis=1)

            # buy trend when max comes after min
            is_buy_trend = np.argmax(arr[:, :, 1], axis=1) > np.argmin(arr[:, :, 0], axis=1)

        elif self.y_type == BandType.BAND_1_1:
            assert arr.shape[2] == 1, "Array.shape[1] is not equal to 1, the data is not coming in the form of BAND_1"

            min_values = np.min(arr[:, :, 0], axis=1)
            max_values = np.max(arr[:, :, 0], axis=1)

            # buy trend when max comes after min
            is_buy_trend = np.argmax(arr[:, :, 0], axis=1) > np.argmin(arr[:, :, 0], axis=1)

        res[:, 0] = min_values
        res[:, 1] = max_values
        res[:, 2] = is_buy_trend.astype(int)

        return res


def check_gaps(data: NDArray[np.float64]) -> None:
    count_gaps_train = 0

    for i_day in range(data.shape[0]):
        for i_tick in range(data.shape[1] - 1):
            close = data[i_day, i_tick, 3]
            next_max = data[i_day, i_tick + 1, 1]
            next_min = data[i_day, i_tick + 1, 2]

            if not (next_min <= close <= next_max):
                count_gaps_train += 1

    print("Count Gaps:\t\t\t", count_gaps_train)
    print(f"Count Gaps Percentage:\t\t {count_gaps_train / (data.shape[0] * data.shape[1]) * 100:.2f} %\n")


def by_date_df_array(df: pd.DataFrame, band_type: BandType, io_type: IODataType) -> NDArray:
    array = df.values

    points_in_each_day: int

    if io_type == IODataType.INPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_1_ST
    elif io_type == IODataType.OUTPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_2_ND

    assert len(array) % points_in_each_day == 0, "Array length is not divisible by points_in_each_day"

    if band_type == BandType.BAND_4:
        assert array.shape[1] == 4, "Array.shape[1] is not equal to 4"

    elif band_type in [BandType.BAND_2, BandType.BAND_2_1]:
        assert array.shape[1] == 2, "Array.shape[1] is not equal to 2"

    elif band_type == BandType.BAND_5:
        assert array.shape[1] == 5, "Array.shape[1] is not equal to 5"

    elif band_type == BandType.BAND_1_1:
        assert array.shape[1] == 1, "Array.shape[1] is not equal to 1"

    num_feature: int = array.shape[1]

    res = array.reshape(len(array) // points_in_each_day, points_in_each_day, num_feature)

    return res
