import numpy as np
import pandas as pd

from database.enums import BandType, TickerOne

ONE_DAY_ZONE_POINTS = 264


def get_data_all_df(ticker: TickerOne, interval) -> pd.DataFrame:
    df = pd.read_csv(get_csv_file_path(ticker.value, interval))

    return df


def get_csv_file_path(ticker, interval) -> str:
    file_path = f"./data_training/{ticker} - {interval}.csv"
    return file_path


def data_split_train_test(df: pd.DataFrame, test_size) -> pd.DataFrame:
    # split into train and test

    # 1 days points inside zone = 264

    num_days: int = len(df) // ONE_DAY_ZONE_POINTS

    train_days: int = int(num_days * (1 - test_size))

    first_test_index: int = train_days * ONE_DAY_ZONE_POINTS

    train_df = df.iloc[:first_test_index]
    test_df = df.iloc[first_test_index:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return (
        train_df[["open", "high", "low", "close", "volume", "real_close"]],
        test_df[["open", "high", "low", "close", "volume", "real_close"]],
    )


def data_split_x_y_close(df: pd.DataFrame, interval: str, x_type: BandType, y_type: BandType) -> pd.DataFrame:
    df_i = pd.DataFrame()
    df_o = pd.DataFrame()

    prev_close = pd.DataFrame()

    for day in range(len(df) // ONE_DAY_ZONE_POINTS):
        day_start_index: int = int(day * ONE_DAY_ZONE_POINTS)
        day_end_index: int = day_start_index + ONE_DAY_ZONE_POINTS - 1

        first_2_nd_zone_index: int = int(day_start_index + ONE_DAY_ZONE_POINTS / 2)

        df_i = pd.concat([df_i, df.iloc[day_start_index:first_2_nd_zone_index]])
        df_o = pd.concat([df_o, df.iloc[first_2_nd_zone_index : day_end_index + 1]])

        dict_1 = {
            "real_close": df.iloc[day_start_index, df.columns.get_loc("real_close")],
        }

        prev_close = pd.concat(
            [prev_close, pd.DataFrame(dict_1, index=[0])],
            ignore_index=True,
        )

    df_i.reset_index(drop=True, inplace=True)
    df_o.reset_index(drop=True, inplace=True)

    columns_x: list[str] = []
    columns_y: list[str] = []

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

    # in every 375 rows for every day, zone is from 47th to 310th row

    for day in range(len(df) // 375):
        start_index = day * 375 + 47
        end_index = day * 375 + 310

        res_df = pd.concat([res_df, df.iloc[start_index : end_index + 1]])

    res_df.reset_index(drop=True, inplace=True)

    return res_df[["open", "high", "low", "close", "real_close", "volume"]]


def by_date_df_array(df: pd.DataFrame, band_type: BandType) -> np.ndarray:
    array = df.values

    if band_type == BandType.BAND_4:
        res = array.reshape(len(array) // 132, 132, 4)

    elif band_type == BandType.BAND_2:
        res = array.reshape(len(array) // 132, 132, 2)

    elif band_type == BandType.BAND_5:
        res = array.reshape(len(array) // 132, 132, 5)

    return res


def train_test_split(data_df, interval, x_type: BandType, y_type: BandType, test_size=0.2) -> pd.DataFrame:
    # separate into 132, 132 entries df. for train and test df.

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
    # 23x(132,4)

    df_test_x, df_test_y, df_test_close = data_split_x_y_close(
        df=df_test,
        interval=interval,
        x_type=x_type,
        y_type=y_type,
    )
    # 6x(132,4)

    train_x = by_date_df_array(df_train_x, band_type=x_type)
    test_x = by_date_df_array(df_test_x, band_type=x_type)

    train_y = by_date_df_array(df_train_y, band_type=y_type)
    test_y = by_date_df_array(df_test_y, band_type=y_type)

    train_prev_close = df_train_close.values
    test_prev_close = df_test_close.values

    return (
        (train_x, train_y, train_prev_close),
        (test_x, test_y, test_prev_close),
    )
