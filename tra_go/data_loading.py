import pandas as pd
from numpy.typing import NDArray
from training_zero import (
    NUMBER_OF_POINTS_IN_ZONE_1_ST,
    NUMBER_OF_POINTS_IN_ZONE_2_ND,
    NUMBER_OF_POINTS_IN_ZONE_DAY,
    TOTAL_POINTS_IN_ONE_DAY,
    by_date_df_array,
    check_gaps,
    data_split_train_test,
)

from database.enums import (
    BandType,
    IntervalType,
    IODataType,
    RequiredDataType,
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

        self.train_y_real_data: NDArray

        self.test_y_real_data: NDArray

        self.load_real_y_data()

    def load_real_y_data(self):
        df = self.get_data_df_real()

        df = self.data_inside_zone(df=df, data_type=RequiredDataType.REAL)

        df_train, df_test = data_split_train_test(
            df=df,
            test_size=self.test_size,
            data_required=RequiredDataType.REAL,
        )

        df_train_x, df_train_y = self.data_split_x_y(df=df_train, data_type=RequiredDataType.REAL)

        df_test_x, df_test_y = self.data_split_x_y(df=df_test, data_type=RequiredDataType.REAL)

        train_y = by_date_df_array(df_train_y, band_type=BandType.BAND_4, io_type=IODataType.OUTPUT_DATA)
        test_y = by_date_df_array(df_test_y, band_type=BandType.BAND_4, io_type=IODataType.OUTPUT_DATA)

        self.train_y_real_data, self.test_y_real_data = train_y, test_y

    def get_real_y_data(self) -> tuple[NDArray, NDArray]:
        print("\ntrain_data - y real")
        check_gaps(self.train_y_real_data)

        print("test_data - y real")
        check_gaps(self.test_y_real_data)

        return self.train_y_real_data, self.test_y_real_data

    def get_data_df_real(self) -> pd.DataFrame:
        file_path: str = f"./data_cleaned/{self.interval.value}/{self.ticker.value} - {self.interval.value}.csv"

        df = pd.read_csv(file_path)

        return df

    def data_split_x_y(self, df: pd.DataFrame, data_type: RequiredDataType | None) -> pd.DataFrame:
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

        band_2_columns: list[str] = ["low", "high"]
        band_4_columns: list[str] = ["open", "high", "low", "close"]
        band_5_columns: list[str] = ["open", "high", "low", "close", "volume"]

        if self.x_type == BandType.BAND_4 or data_type == RequiredDataType.REAL:
            columns_x = band_4_columns

        elif self.x_type == BandType.BAND_2:
            columns_x = band_2_columns

        elif self.x_type == BandType.BAND_5:
            columns_x = band_5_columns

        if self.y_type == BandType.BAND_4 or data_type == RequiredDataType.REAL:
            columns_y = band_4_columns

        elif self.y_type in [BandType.BAND_2, BandType.BAND_2_1]:
            columns_y = band_2_columns

        elif self.y_type == BandType.BAND_5:
            columns_y = band_5_columns

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

        number_of_days = len(df) // TOTAL_POINTS_IN_ONE_DAY

        for day in range(number_of_days):
            start_index: int = day * TOTAL_POINTS_IN_ONE_DAY + initial_index_offset
            end_index: int = start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

            res_df = pd.concat([res_df, df.iloc[start_index : end_index + 1]])

        res_df.reset_index(drop=True, inplace=True)

        columns: list[str]

        if data_type == RequiredDataType.TRAINING:
            columns = ["open", "high", "low", "close", "real_close", "volume"]
        elif data_type == RequiredDataType.REAL:
            columns = ["open", "high", "low", "close", "volume"]

        return res_df[columns]
