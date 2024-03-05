import numpy as np
import pandas as pd

NUM_ONE_DAY_ZONE_POINTS = 264


def get_data_all_df(ticker, interval) -> pd.DataFrame:
    if ticker == "CCI":
        ticker = "ICICIBANK"

    df = pd.read_csv(get_csv_file_path(ticker, interval))

    return df


def get_csv_file_path(ticker, interval) -> str:
    file_path = f"./data_training/{ticker} - {interval}.csv"
    return file_path


def data_split_train_test(df: pd.DataFrame, test_size) -> pd.DataFrame:
    # split into train and test

    # 1 days points inside zone = 264

    num_days: int = len(df) // NUM_ONE_DAY_ZONE_POINTS

    train_days: int = int(num_days * (1 - test_size))

    first_test_index: int = train_days * NUM_ONE_DAY_ZONE_POINTS

    train_df = df.iloc[:first_test_index]
    test_df = df.iloc[first_test_index:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return (
        train_df[["open", "close", "high", "low", "real_close"]],
        test_df[["open", "close", "high", "low", "real_close"]],
    )


def data_split_x_y_close(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df_i = pd.DataFrame()
    df_o = pd.DataFrame()

    prev_close = pd.DataFrame()

    for day in range(len(df) // NUM_ONE_DAY_ZONE_POINTS):
        day_start_index: int = int(day * NUM_ONE_DAY_ZONE_POINTS)
        day_end_index: int = day_start_index + NUM_ONE_DAY_ZONE_POINTS - 1

        first_2_nd_zone_index: int = int(day_start_index + NUM_ONE_DAY_ZONE_POINTS / 2)

        df_i = pd.concat([df_i, df.iloc[day_start_index:first_2_nd_zone_index]])
        df_o = pd.concat([df_o, df.iloc[first_2_nd_zone_index : day_end_index + 1]])

        dict_1 = {"real_close": df.iloc[day_start_index, df.columns.get_loc("real_close")]}

        prev_close = pd.concat([prev_close, pd.DataFrame(dict_1, index=[0])], ignore_index=True)

    df_i.reset_index(drop=True, inplace=True)
    df_o.reset_index(drop=True, inplace=True)

    return (
        df_i[["open", "close", "high", "low"]],
        df_o[["open", "close", "high", "low"]],
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

    return res_df[["open", "close", "high", "low", "real_close"]]


def by_date_df_array(df: pd.DataFrame) -> np.ndarray:
    array = df.values
    res = array.reshape(len(array) // 132, 132, 4)

    return res


def train_test_split(data_df, interval, y_type, test_size=0.2) -> pd.DataFrame:
    # separate into 132, 132 entries df. for train and test df.

    # divide the price data of that day by the closing price of the previous day.
    # for the very first day of the dataset - divide the prices by the opening price.

    df = data_df.copy(deep=True)

    df = data_inside_zone(df=df, interval=interval)

    df_train, df_test = data_split_train_test(df=df, test_size=test_size)

    df_train_x, df_train_y, df_train_close = data_split_x_y_close(df=df_train, interval=interval)
    # 23x(132,4)

    df_test_x, df_test_y, df_test_close = data_split_x_y_close(df=df_test, interval=interval)
    # 6x(132,4)

    train_x = by_date_df_array(df_train_x)
    test_x = by_date_df_array(df_test_x)

    train_y = by_date_df_array(df_train_y)
    test_y = by_date_df_array(df_test_y)

    train_prev_close = df_train_close.values
    test_prev_close = df_test_close.values

    return (
        (train_x, train_y, train_prev_close),
        (test_x, test_y, test_prev_close),
    )
