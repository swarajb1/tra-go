import numpy as np
import pandas as pd

from database.enums import BandType, IODataType, TickerOne

TOTAL_POINTS_IN_ONE_DAY: int = 375

NUMBER_OF_POINTS_IN_ZONE_1_ST: int = 132
NUMBER_OF_POINTS_IN_ZONE_2_ND: int = 132

NUMBER_OF_POINTS_IN_ZONE_DAY: int = NUMBER_OF_POINTS_IN_ZONE_1_ST + NUMBER_OF_POINTS_IN_ZONE_2_ND


def get_data_all_df(ticker: TickerOne, interval) -> pd.DataFrame:
    df = pd.read_csv(get_csv_file_path(ticker.value, interval))

    return df


def get_csv_file_path(ticker, interval) -> str:
    file_path = f"./data_training/{ticker} - {interval}.csv"
    return file_path


def data_split_train_test(df: pd.DataFrame, test_size) -> pd.DataFrame:
    # split into train and test

    # 1 days points inside zone = 264

    number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

    train_days: int = int(number_of_days * (1 - test_size))

    first_test_index: int = train_days * NUMBER_OF_POINTS_IN_ZONE_DAY

    train_df = df.iloc[:first_test_index]
    test_df = df.iloc[first_test_index:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return (
        train_df[["open", "high", "low", "close", "volume", "real_close"]],
        test_df[["open", "high", "low", "close", "volume", "real_close"]],
    )


def data_split_x_y_close(df: pd.DataFrame, interval: str, x_type: BandType, y_type: BandType) -> pd.DataFrame:
    """Splits the data into input, output dataframe, and the previous close
    price dataframe."""

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

    if x_type == BandType.BAND_4:
        columns_x = ["open", "high", "low", "close"]
    elif x_type == BandType.BAND_2:
        columns_x = ["low", "high"]
    elif x_type == BandType.BAND_5:
        columns_x = ["open", "high", "low", "close", "volume"]

    if y_type == BandType.BAND_4:
        columns_y = ["open", "high", "low", "close"]
    elif y_type == BandType.BAND_2:
        columns_y = ["low", "high"]
    elif x_type == BandType.BAND_5:
        columns_y = ["open", "high", "low", "close", "volume"]

    return (
        df_i[columns_x],
        df_o[columns_y],
        prev_close[["real_close"]],
    )


def data_inside_zone(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    res_df = pd.DataFrame()

    # when taking from full starting
    INITIAL_INDEX_OFFSET: int = 0

    # when taking from 1000
    # INITIAL_OFFSET: int = 47

    number_of_days = len(df) // TOTAL_POINTS_IN_ONE_DAY

    for day in range(number_of_days):
        start_index: int = day * TOTAL_POINTS_IN_ONE_DAY + INITIAL_INDEX_OFFSET
        end_index: int = day * TOTAL_POINTS_IN_ONE_DAY + (INITIAL_INDEX_OFFSET + NUMBER_OF_POINTS_IN_ZONE_DAY)

        res_df = pd.concat([res_df, df.iloc[start_index : end_index + 1]])

    res_df.reset_index(drop=True, inplace=True)

    return res_df[["open", "high", "low", "close", "real_close", "volume"]]


def by_date_df_array(df: pd.DataFrame, band_type: BandType, io_type: IODataType) -> np.ndarray:
    array = df.values

    points_in_each_day: int

    if io_type == IODataType.INPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_1_ST
    elif io_type == IODataType.OUTPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_2_ND

    if band_type == BandType.BAND_4:
        res = array.reshape(len(array) // points_in_each_day, points_in_each_day, 4)

    elif band_type == BandType.BAND_2:
        res = array.reshape(len(array) // points_in_each_day, points_in_each_day, 2)

    elif band_type == BandType.BAND_5:
        res = array.reshape(len(array) // points_in_each_day, points_in_each_day, 5)

    return res


def train_test_split(data_df, interval, x_type: BandType, y_type: BandType, test_size=0.2) -> pd.DataFrame:
    # separate into 178, 132 entries df. for train and test df.
    # # separate into 132, 132 entries df. for train and test df.

    # divide the price data of that day by the closing price of the previous day.
    # for the very first day of the dataset - divide the prices by the opening price.

    df = data_df.copy(deep=True)

    df = data_inside_zone(df=df, interval=interval)

    df_train, df_test = data_split_train_test(df=df, test_size=test_size)

    df_train_x, df_train_y, df_train_close = data_split_x_y_close(
        df=df_train,
        interval=interval,
        x_type=x_type,
        y_type=y_type,
    )

    df_test_x, df_test_y, df_test_close = data_split_x_y_close(
        df=df_test,
        interval=interval,
        x_type=x_type,
        y_type=y_type,
    )

    train_x = by_date_df_array(df_train_x, band_type=x_type, io_type=IODataType.INPUT_DATA)
    test_x = by_date_df_array(df_test_x, band_type=x_type, io_type=IODataType.INPUT_DATA)

    train_y = by_date_df_array(df_train_y, band_type=y_type, io_type=IODataType.OUTPUT_DATA)
    test_y = by_date_df_array(df_test_y, band_type=y_type, io_type=IODataType.OUTPUT_DATA)

    train_prev_close = df_train_close.values
    test_prev_close = df_test_close.values

    return (
        (train_x, train_y, train_prev_close),
        (test_x, test_y, test_prev_close),
    )
