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

from database.enums import (
    BandType,
    IntervalType,
    IODataType,
    RequiredDataType,
    TickerDataType,
    TickerOne,
)


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
        self.train_prev_close: NDArray[np.float64]

        self.test_x_data: NDArray[np.float64]
        self.test_y_data: NDArray[np.float64]
        self.test_prev_close: NDArray[np.float64]

        self.load_data_train_test_close()

    def load_data_y_real(self) -> None:
        df = self._get_data_df(ticker_data_type=TickerDataType.CLEANED)

        df = self.data_inside_zone(df=df, data_type=RequiredDataType.REAL)

        df_train, df_test = self.data_split_train_test(df=df, required_data_type=RequiredDataType.REAL)

        df_train_x, df_train_y = self.data_split_x_y(df=df_train, required_data_type=RequiredDataType.REAL)

        df_test_x, df_test_y = self.data_split_x_y(df=df_test, required_data_type=RequiredDataType.REAL)

        train_y = by_date_df_array(df_train_y, band_type=BandType.BAND_4, io_type=IODataType.OUTPUT_DATA)
        test_y = by_date_df_array(df_test_y, band_type=BandType.BAND_4, io_type=IODataType.OUTPUT_DATA)

        self.train_y_real_data, self.test_y_real_data = train_y, test_y

        return

    def get_real_y_data(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        print("\ntrain_data - y real")
        check_gaps(self.train_y_real_data)

        print("test_data - y real")
        check_gaps(self.test_y_real_data)

        return self.train_y_real_data, self.test_y_real_data

    def _get_data_df(self, ticker_data_type: TickerDataType) -> pd.DataFrame:
        # file_path: str = f"./data_cleaned/{self.interval.value}/{self.ticker.value} - {self.interval.value}.csv"

        file_path = os.path.join(
            ".",
            ticker_data_type.value,
            self.interval.value,
            f"{self.ticker.value} - {self.interval.value}.csv",
        )

        df = pd.read_csv(file_path)

        return df

    def data_split_x_y(self, df: pd.DataFrame, required_data_type: RequiredDataType | None) -> pd.DataFrame:
        """Splits the data into input, output dataframe."""

        df_i = pd.DataFrame()
        df_o = pd.DataFrame()

        number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

        for day in range(number_of_days):
            day_start_index: int = int(day * NUMBER_OF_POINTS_IN_ZONE_DAY)
            day_end_index: int = day_start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

            first_2_nd_zone_index: int = int(day_start_index + NUMBER_OF_POINTS_IN_ZONE_1_ST)

            df_i = pd.concat([df_i, df.iloc[day_start_index:first_2_nd_zone_index]])
            df_o = pd.concat([df_o, df.iloc[first_2_nd_zone_index : day_end_index + 1]])

        df_i.reset_index(drop=True, inplace=True)
        df_o.reset_index(drop=True, inplace=True)

        columns_x: list[str]
        columns_y: list[str]

        band_columns: dict[BandType, list[str]] = {
            BandType.BAND_2: ["low", "high"],
            BandType.BAND_2_1: ["low", "high"],
            BandType.BAND_4: ["open", "high", "low", "close"],
            BandType.BAND_5: ["open", "high", "low", "close", "volume"],
        }

        if required_data_type == RequiredDataType.REAL:
            columns_x = band_columns[BandType.BAND_4]
            columns_y = band_columns[BandType.BAND_4]
        else:
            columns_x = band_columns[self.x_type]
            columns_y = band_columns[self.y_type]

        return df_i[columns_x], df_o[columns_y]

    def data_inside_zone(self, df: pd.DataFrame, data_type: RequiredDataType) -> pd.DataFrame:
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
        ), "Full Cleaned Data length is not divisible by TOTAL_POINTS_IN_ONE_DAY"

        for day in range(number_of_days):
            start_index: int = day * TOTAL_POINTS_IN_ONE_DAY + initial_index_offset
            end_index: int = start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

            res_df = pd.concat([res_df, df.iloc[start_index : end_index + 1]])

        res_df.reset_index(drop=True, inplace=True)

        columns: list[str] = self.get_columns(data_type)

        return res_df[columns]

    def get_columns(self, required_data_type: RequiredDataType) -> list[str]:
        columns: list[str]

        base_columns: list[str] = ["open", "high", "low", "close", "volume"]

        if required_data_type == RequiredDataType.TRAINING:
            columns = base_columns + ["real_close"]
        elif required_data_type == RequiredDataType.REAL:
            columns = base_columns

        return columns

    def data_split_train_test(
        self,
        df: pd.DataFrame,
        required_data_type: RequiredDataType,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # split into train and test

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

        columns: list[str] = self.get_columns(required_data_type)

        return train_df[columns], test_df[columns]

    def get_train_test_split(
        self,
    ) -> tuple[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]:
        return (
            (self.train_x_data, self.train_y_data, self.train_prev_close),
            (self.test_x_data, self.test_y_data, self.test_prev_close),
        )

    def load_data_train_test_close(self, df: pd.DataFrame) -> pd.DataFrame:
        """Splits the data into input, output dataframe, and the previous close
        price dataframe."""

        df = self.data_inside_zone(df=df, data_type=RequiredDataType.TRAINING)

        df_i = pd.DataFrame()
        df_o = pd.DataFrame()

        prev_close = pd.DataFrame()

        number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

        for day in range(number_of_days):
            day_start_index: int = int(day * NUMBER_OF_POINTS_IN_ZONE_DAY)
            day_end_index: int = day_start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

            first_2_nd_zone_index: int = int(day_start_index + NUMBER_OF_POINTS_IN_ZONE_1_ST)

            df_i = pd.concat([df_i, df.iloc[day_start_index:first_2_nd_zone_index]])
            df_o = pd.concat([df_o, df.iloc[first_2_nd_zone_index : day_end_index + 1]])

            dict_1 = {"real_close": df.iloc[day_start_index, df.columns.get_loc("real_close")]}

            prev_close = pd.concat([prev_close, pd.DataFrame(dict_1, index=[0])], ignore_index=True)

        df_i.reset_index(drop=True, inplace=True)
        df_o.reset_index(drop=True, inplace=True)

        columns_x: list[str]
        columns_y: list[str]

        if self.x_type == BandType.BAND_4:
            columns_x = ["open", "high", "low", "close"]
        elif self.x_type == BandType.BAND_2:
            columns_x = ["low", "high"]
        elif self.x_type == BandType.BAND_5:
            columns_x = ["open", "high", "low", "close", "volume"]

        if self.y_type == BandType.BAND_4:
            columns_y = ["open", "high", "low", "close"]
        elif self.y_type in [BandType.BAND_2, BandType.BAND_2_1]:
            columns_y = ["low", "high"]
        elif self.y_type == BandType.BAND_5:
            columns_y = ["open", "high", "low", "close", "volume"]

        return (
            df_i[columns_x],
            df_o[columns_y],
            prev_close[["real_close"]],
        )


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
    print(
        "Count Gaps Percentage:\t\t",
        "{:.2f}".format(count_gaps_train / (data.shape[0] * data.shape[1]) * 100),
        " %\n",
    )


def by_date_df_array(df: pd.DataFrame, band_type: BandType, io_type: IODataType) -> NDArray:
    array = df.values

    points_in_each_day: int

    if io_type == IODataType.INPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_1_ST
    elif io_type == IODataType.OUTPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_2_ND

    if band_type == BandType.BAND_4:
        assert len(array) % (points_in_each_day * 4) == 0, "Array length is not divisible by 4 * points_in_each_day"

        res = array.reshape(len(array) // points_in_each_day, points_in_each_day, 4)

    elif band_type in [BandType.BAND_2, BandType.BAND_2_1]:
        assert len(array) % (points_in_each_day * 2) == 0, "Array length is not divisible by 2 * points_in_each_day"

        res = array.reshape(len(array) // points_in_each_day, points_in_each_day, 2)

    elif band_type == BandType.BAND_5:
        assert len(array) % (points_in_each_day * 5) == 0, "Array length is not divisible by 5 * points_in_each_day"

        res = array.reshape(len(array) // points_in_each_day, points_in_each_day, 5)

    return res
