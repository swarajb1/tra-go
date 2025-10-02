from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from numpy.typing import NDArray


def is_in_half(check_datetime, which_half: int, interval: str) -> bool:
    # which_half = 0, means 1st half - input data
    # which_half = 1, means 2nd half - predict data

    datetime_1 = datetime.strptime("2000-01-01 10:00:00+0530", "%Y-%m-%d %H:%M:%S%z")
    time_1 = (datetime_1 + timedelta(minutes=132 * which_half)).time()
    time_check = datetime.strptime(check_datetime, "%Y-%m-%d %H:%M:%S%z").time()

    if interval == "1m":
        time_2 = (datetime_1 + timedelta(minutes=132 * (which_half + 1) + 1)).time()

    elif interval == "2m":
        time_2 = (datetime_1 + timedelta(minutes=132 * (which_half + 1) + 2)).time()

    elif interval == "5m":
        time_2 = (datetime_1 + timedelta(minutes=132 * (which_half + 1) + 5)).time()

    if time_check > time_1 and time_check < time_2:
        return True

    return False


def to_date(datetime_1) -> str:
    return datetime.strptime(datetime_1, "%Y-%m-%d %H:%M:%S%z").date()


def to_date_str(datetime_1) -> str:
    return to_date(datetime_1).strftime("%Y-%m-%d")


def is_same_date(datetime_1, check_datetime):
    return to_date(datetime_1) == to_date(check_datetime)


def is_same_date_2(date_1, list_check_date_str):
    for d_1 in list_check_date_str:
        if d_1 == date_1:
            return True
    return False


def is_in_zone(check_datetime, interval) -> bool:
    datetime_1 = datetime.strptime("2000-01-01 10:00:00+0530", "%Y-%m-%d %H:%M:%S%z")
    time_1 = datetime_1.time()
    time_check = datetime.strptime(check_datetime, "%Y-%m-%d %H:%M:%S%z").time()

    if interval == "1m":
        time_2 = (datetime_1 + timedelta(minutes=132 * 2 + 1)).time()

    elif interval == "2m":
        time_2 = (datetime_1 + timedelta(minutes=132 * 2 + 2)).time()

    elif interval == "5m":
        time_2 = (datetime_1 + timedelta(minutes=132 * 2 + 5)).time()

    else:
        raise ValueError("Interval not supported")

    if time_check > time_1 and time_check < time_2:
        return True

    return False


def is_in_first_half(check_datetime, interval):
    return is_in_half(check_datetime, which_half=0, interval=interval)


def is_in_second_half(check_datetime, interval):
    return is_in_half(check_datetime, which_half=1, interval=interval)


def get_data_df(ticker, interval, which_half: str) -> pd.DataFrame:
    df = pd.read_csv(get_csv_file_path(ticker, interval))

    if which_half == "full_zone":
        df["to_add"] = df["Datetime"].apply(lambda x: is_in_zone(x, interval=interval))
    elif which_half == "first_half":
        df["to_add"] = df["Datetime"].apply(lambda x: is_in_first_half(x))
    elif which_half == "second_half":
        df["to_add"] = df["Datetime"].apply(lambda x: is_in_second_half(x))

    df["open"] = df["Open"].apply(lambda x: round(number=x, ndigits=2))
    df["close"] = df["Close"].apply(lambda x: round(number=x, ndigits=2))
    df["high"] = df["High"].apply(lambda x: round(number=x, ndigits=2))
    df["low"] = df["Low"].apply(lambda x: round(number=x, ndigits=2))

    new_2 = df[df["to_add"]].copy(deep=True)
    new_2.rename(columns={"Datetime": "datetime"}, inplace=True)

    return new_2[
        [
            "datetime",
            "open",
            "close",
            "high",
            "low",
        ]
    ]


def get_data_all_df(ticker, interval) -> pd.DataFrame:
    df = pd.read_csv(get_csv_file_path(ticker, interval))

    df["open"] = df["Open"].apply(lambda x: round(number=x, ndigits=2))
    df["close"] = df["Close"].apply(lambda x: round(number=x, ndigits=2))
    df["high"] = df["High"].apply(lambda x: round(number=x, ndigits=2))
    df["low"] = df["Low"].apply(lambda x: round(number=x, ndigits=2))

    df.rename(columns={"Datetime": "datetime"}, inplace=True)

    print(df.columns)

    return df[
        [
            "datetime",
            "open",
            "close",
            "high",
            "low",
        ]
    ]


def get_csv_file_path(ticker, interval) -> str:
    file_path = f"./data_yf/nse/{interval}_data/{ticker} - {interval}.csv"
    return file_path


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # start time = 0915
    # last time = 1529
    # total minutes = 375

    df = df.sort_values(by="datetime", ascending=True)

    df["date"] = df["datetime"].apply(lambda x: to_date_str(x))
    all_dates = df["date"].unique()

    all_datetimes_required = []
    timezone = pytz.timezone("Asia/Kolkata")
    for date in all_dates:
        that_date = datetime.strptime(date, "%Y-%m-%d")
        date_obj = datetime(
            year=that_date.year,
            month=that_date.month,
            day=that_date.day,
            hour=9,
            minute=15,
            second=0,
        )
        first_datetime = timezone.localize(date_obj)
        for i in range(375):
            all_datetimes_required.append(
                (first_datetime + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S%z"),
            )

    all_datetimes_in_data = []
    for index, row in df.iterrows():
        all_datetimes_in_data.append(
            datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S%z").strftime(
                "%Y-%m-%d %H:%M:%S%z",
            ),
        )

    # make a set of all datetime to be there
    # and what datetime are actually there
    # finding missing ones, using set
    # add them as zeros in correct datetimes, sort df
    # put previous values in them in palce of zeros

    all_datetimes_required_set = set(all_datetimes_required)
    all_datetimes_in_data_set = set(all_datetimes_in_data)

    missing_datetimes = all_datetimes_required_set - all_datetimes_in_data_set

    add_df_rows = []
    for d in missing_datetimes:
        dict_1 = {
            "datetime": d,
            "open": 0,
            "close": 0,
            "high": 0,
            "low": 0,
        }
        add_df_rows.append(deepcopy(dict_1))
        dict_1.clear()

    new_df = pd.DataFrame(add_df_rows)
    df = pd.concat([df, new_df], ignore_index=True)

    df = df.sort_values(by="datetime", ascending=True)
    df.reset_index(drop=True, inplace=True)

    df.drop("date", axis=1, inplace=True)

    missing_indexes = []
    for index, row in df.iterrows():
        if row["open"] == 0:
            missing_indexes.append(index)

    missing_rows = len(missing_indexes)
    while missing_rows != 0:
        for index in missing_indexes:
            if df.at[index, "open"] == 0:
                if index > 0:
                    ref_index = index - 1
                else:
                    ref_index = index + 1

                if df.at[ref_index, "open"] != 0:
                    df.at[index, "open"] = df.at[ref_index, "open"]
                    df.at[index, "close"] = df.at[ref_index, "close"]
                    df.at[index, "high"] = df.at[ref_index, "high"]
                    df.at[index, "low"] = df.at[ref_index, "low"]
                    missing_rows -= 1

    # then divide accordingly into 2 zones
    # send only full zone df, not full date.

    return df[
        [
            "datetime",
            "open",
            "close",
            "high",
            "low",
        ]
    ]


def data_scaling(df: pd.DataFrame) -> pd.DataFrame:
    # fixded 375 rows for all days.
    # just divide by the close value of previous row,
    # for the very first day, use the open value

    one_day = 375
    num_days = len(df) // one_day

    # for other days
    for j in range(num_days - 1, 0, -1):
        prev_close = df.loc[j * one_day - 1, "close"]
        for i in range(one_day):
            df.loc[j * one_day + i, "close"] /= prev_close
            df.loc[j * one_day + i, "high"] /= prev_close
            df.loc[j * one_day + i, "low"] /= prev_close
            df.loc[j * one_day + i, "open"] /= prev_close

    # for 1st day
    open_val = df.at[0, "open"]
    for i in range(one_day):
        df.at[i, "close"] /= open_val
        df.at[i, "high"] /= open_val
        df.at[i, "low"] /= open_val
        df.at[i, "open"] /= open_val

    return df


def get_prev_close(df: pd.DataFrame) -> NDArray:
    one_day = 375
    num_days = len(df) // one_day

    res: NDArray = np.zeros(num_days)

    # for other days
    for j in range(num_days - 1, 0, -1):
        prev_close = df.loc[j * one_day - 1, "close"]
        res[j] = prev_close

    # for 1st day
    open_val = df.at[0, "open"]
    res[0] = open_val

    return res


def data_my_zone(df: pd.DataFrame, interval) -> pd.DataFrame:
    df["to_add"] = df["datetime"].apply(lambda x: is_in_zone(x, interval=interval))

    new_2 = df[df["to_add"]].copy(deep=True)
    new_2.reset_index(drop=True, inplace=True)

    return new_2[
        [
            "datetime",
            "open",
            "close",
            "high",
            "low",
        ]
    ]


def data_split_train_test(df: pd.DataFrame, test_size) -> pd.DataFrame:
    # split into train and test
    # not 4 parts, with x and y
    # now both x, y in one df

    df["date"] = df["datetime"].apply(lambda x: to_date_str(x))
    all_dates = df["date"].unique()
    num_days = len(all_dates)

    training_dates = all_dates[: int(num_days * (1 - test_size))]
    testing_dates = all_dates[int(num_days * (1 - test_size)) :]

    df["is_train"] = df["date"].apply(lambda x: is_same_date_2(x, training_dates))

    df["is_test"] = df["date"].apply(lambda x: is_same_date_2(x, testing_dates))

    train_df = df[df["is_train"]].copy(deep=True)
    test_df = df[df["is_test"]].copy(deep=True)

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    return (
        train_df[["datetime", "open", "close", "high", "low", "date"]],
        test_df[["datetime", "open", "close", "high", "low", "date"]],
    )


def data_split_x_y(df, interval) -> pd.DataFrame:
    df["is_input"] = df["datetime"].apply(
        lambda x: is_in_half(x, which_half=0, interval=interval),
    )
    df["is_output"] = df["datetime"].apply(
        lambda x: is_in_half(x, which_half=1, interval=interval),
    )

    df_i = df[df["is_input"]].copy(deep=True)
    df_o = df[df["is_output"]].copy(deep=True)

    df_i.reset_index(drop=True, inplace=True)
    df_o.reset_index(drop=True, inplace=True)

    return (
        df_i[["datetime", "open", "close", "high", "low", "date"]],
        df_o[["datetime", "open", "close", "high", "low", "date"]],
    )


def data_split_x_y_new(df, interval) -> pd.DataFrame:
    # split the whole dataframe into 2 parts = input and output

    df["date"] = df["datetime"].apply(lambda x: to_date_str(x))

    df["is_input"] = df["datetime"].apply(
        lambda x: is_in_half(x, which_half=0, interval=interval),
    )
    df["is_output"] = df["datetime"].apply(
        lambda x: is_in_half(x, which_half=1, interval=interval),
    )

    df_i = df[df["is_input"]].copy(deep=True)
    df_o = df[df["is_output"]].copy(deep=True)

    df_i.reset_index(drop=True, inplace=True)
    df_o.reset_index(drop=True, inplace=True)

    return (
        df_i[["datetime", "open", "close", "high", "low", "date"]],
        df_o[["datetime", "open", "close", "high", "low", "date"]],
    )


def train_test_split(
    data_df,
    interval,
    y_type,
    test_size=0.2,
) -> [pd.DataFrame]:
    # separate into 132, 132 entries df. for train and test df.

    # divide the price data of that day by the closing price of the previous day.
    # for the very first day of the dataset - divide the prices by the opening price.

    df = data_cleaning(data_df)

    df_cleaned = df.copy(deep=True)

    # getting clean and my zone data
    df = data_scaling(df)
    # getting scaled data according to previous day closing price, in percentages terms
    df = data_my_zone(df, interval=interval)
    # getting data is inside the full zone.

    df_train, df_test = data_split_train_test(df=df, test_size=test_size)

    df_train_x, df_train_y = data_split_x_y(df=df_train, interval=interval)
    # 23x(132,4)

    df_test_x, df_test_y = data_split_x_y(df=df_test, interval=interval)
    # 6x(132,4)

    df_x, df_y = data_split_x_y(df=df, interval=interval)

    selected_columns_1 = ["low", "high", "open", "close"]
    selected_columns_2 = ["low", "high"]

    train_x = by_date_df_array(df_train_x[selected_columns_1])
    test_x = by_date_df_array(df_test_x[selected_columns_1])

    if y_type == "hl":
        train_y = points_hl(df_train_y[selected_columns_2])
        test_y = points_hl(df_test_y[selected_columns_2])

    elif y_type == "band":
        train_y = by_date_df_array(df_train_y[selected_columns_2])
        test_y = by_date_df_array(df_test_y[selected_columns_2])

    elif y_type == "band_2":
        # all x and y is transformed to avg_price, band_height
        train_y_temp = by_date_df_array(df_train_y[selected_columns_2])
        test_y_temp = by_date_df_array(df_test_y[selected_columns_2])

        train_y = array_transform_avg_band(train_y_temp)
        test_y = array_transform_avg_band(test_y_temp)

        x_close = last_close_value(df_x)

        return ((train_x, train_y), (test_x, test_y), x_close)

    elif y_type == "band_4":
        train_y = by_date_df_array(df_train_y[selected_columns_1])
        test_y = by_date_df_array(df_test_y[selected_columns_1])

        df_train_c, df_test_c = data_split_train_test(
            df=df_cleaned,
            test_size=test_size,
        )

        train_prev_close = get_prev_close(df_train_c)
        test_prev_close = get_prev_close(df_test_c)

        return (
            (train_x, train_y, train_prev_close),
            (test_x, test_y, test_prev_close),
        )

    return ((train_x, train_y), (test_x, test_y))


def train_test_split_5(
    data_df,
    interval,
    y_type,
    test_size=0.2,
) -> NDArray:
    # separate into 132, 132 entries df. for train and test df.

    # divide the price data of that day by the closing price of the previous day.
    # for the very first day of the dataset - divide the prices by the opening price.

    df = data_cleaning(data_df)

    df_cleaned = df.copy(deep=True)

    # getting clean and my zone data
    df = data_scaling(df)
    # getting scaled data according to previous day closing price, in percentages terms
    df = data_my_zone(df, interval=interval)
    # getting data is inside the full zone.

    df_train, df_test = data_split_train_test(df=df, test_size=test_size)

    df_train_x, df_train_y = data_split_x_y(df=df_train, interval=interval)

    df_test_x, df_test_y = data_split_x_y(df=df_test, interval=interval)

    selected_columns_1 = ["low", "high", "open", "close"]

    train_x = by_date_df_array(df_train_x[selected_columns_1])
    test_x = by_date_df_array(df_test_x[selected_columns_1])

    train_y = by_date_df_array(df_train_y[selected_columns_1])
    test_y = by_date_df_array(df_test_y[selected_columns_1])

    train_y = add_trend_parameter(train_y)
    test_y = add_trend_parameter(test_y)

    df_train_c, df_test_c = data_split_train_test(df=df_cleaned, test_size=test_size)

    train_prev_close = get_prev_close(df_train_c)
    test_prev_close = get_prev_close(df_test_c)

    return (
        (train_x, train_y, train_prev_close),
        (test_x, test_y, test_prev_close),
    )


def get_day_trend(day_arr: NDArray) -> int:
    min_index: NDArray = np.argmin(day_arr[:, 0])
    max_index: NDArray = np.argmax(day_arr[:, 1])

    return int(max_index > min_index)


def add_trend_parameter(arr: NDArray) -> NDArray:
    res = np.zeros((arr.shape[0], arr.shape[1], 5))

    res[:, :, :4] = arr

    for day in range(arr.shape[0]):
        res[day, :, 4] = get_day_trend(arr[day])

    return res


def last_close_value(df: pd.DataFrame) -> NDArray:
    res = np.array([])

    res = np.append(res, df.iloc[0]["open"])

    for i in range(1, len(df) // 132):
        res = np.append(res, df.iloc[(i - 1) * 132 + 131]["close"])

    return res


def train_test_split_2_mods(
    data_df,
    interval,
    y_type,
    test_size=0.2,
) -> NDArray:
    # y_type = "2_mods"

    # separate into 132, 132 entries df. for train and test df.

    # divide the price data of that day by the closing price of the previous day.
    # for the very first day of the dataset - divide the prices by the opening price.

    df = data_cleaning(data_df)
    # getting clean and my zone data
    df = data_scaling(df)
    # getting scaled data according to previous day closing price, in percentages terms
    df = data_my_zone(df, interval=interval)
    # getting data is inside the full zone.

    df_train, df_test = data_split_train_test(df=df, test_size=test_size)

    df_train_x, df_train_y = data_split_x_y(df=df_train, interval=interval)
    # 23x(132,4)

    df_test_x, df_test_y = data_split_x_y(df=df_test, interval=interval)
    # 6x(132,4)

    res = {}
    for column in ["high", "low"]:
        train_x = by_date_np_array(df_train_x[column].values)
        test_x = by_date_np_array(df_test_x[column].values)
        train_y = by_date_np_array(df_train_y[column].values)
        test_y = by_date_np_array(df_test_y[column].values)

        res[column] = {}
        res[column]["train_x"] = train_x
        res[column]["test_x"] = test_x
        res[column]["train_y"] = train_y
        res[column]["test_y"] = test_y
    return res


def points_hl(df):
    res = np.array([])
    full_rows = []
    for index, row in df.iterrows():
        x = row.values.tolist()
        full_rows.append(x)

    for i in range(len(full_rows) // 132):
        high = max([x[0] for x in full_rows[i * 132 : (i + 1) * 132]])
        low = min([x[1] for x in full_rows[i * 132 : (i + 1) * 132]])

        if res.size == 0:
            res = np.array([np.array([high, low])])
        else:
            res = np.append(res, [np.array([high, low])], axis=0)

    return res


def by_date_df_array(df: pd.DataFrame) -> NDArray:
    res = []
    full_rows = []
    for index, row in df.iterrows():
        x = row.values.tolist()
        full_rows.append(x)

    for i in range(len(full_rows) // 132):
        res.append(deepcopy(full_rows[i * 132 : (i + 1) * 132]))

    res_1 = np.array([np.array(xi) for xi in res])

    return res_1


def array_transform_avg_band(array: NDArray) -> NDArray:
    res = np.zeros_like(array)
    # index 0 is low, 1 is high

    # average price
    res[:, :, 0] = (array[:, :, 0] + array[:, :, 1]) / 2

    # band height
    res[:, :, 1] = array[:, :, 1] - array[:, :, 0]

    return res


def by_date_np_array(np_array) -> list[pd.DataFrame]:
    res = []
    full_rows = np_array.tolist()

    for i in range(len(full_rows) // 132):
        res.append(deepcopy(full_rows[i * 132 : (i + 1) * 132]))

    res_1 = np.array([np.array(xi) for xi in res])

    res_3d = np.expand_dims(res_1, axis=-1)

    return res_3d


def split_by_date(data_df: pd.DataFrame, interval: str, columns: list[str]):
    """Assumes that the number of data points is divisible by 132."""

    if interval == "1m":
        number_of_points: int = 132

    res: NDArray = np.empty((0, number_of_points, len(columns)))

    temp = np.empty((0, len(columns)))
    for index, row in data_df.iterrows():
        z = row.loc[columns].values
        temp = np.append(temp, [z], axis=0)

        if (index + 1) % number_of_points == 0:
            res = np.append(res, np.array([temp]), axis=0)
            temp = np.empty((0, len(columns)))

    return res


def get_x_y_individual_data(
    data_df: pd.DataFrame,
    interval: str,
    columns: list[str],
) -> NDArray:
    df = data_cleaning(data_df)
    # getting clean and my zone data
    df = data_scaling(df)
    # getting scaled data according to previous day closing price, in percentages terms
    df = data_my_zone(df, interval=interval)

    data_df_x, data_df_y = data_split_x_y_new(df=df, interval=interval)

    arr_x = split_by_date(data_df=data_df_x, interval=interval, columns=columns)
    arr_y = split_by_date(data_df=data_df_y, interval=interval, columns=columns)

    arr_x = arr_x.astype(np.float64)
    arr_y = arr_y.astype(np.float64)

    return arr_x, arr_y


def append_test_train_arr(
    X_train,
    Y_train,
    X_test,
    Y_test,
    train_prev_close,
    test_prev_close,
):
    X_arr = np.append(X_train, X_test, axis=0)
    Y_arr = np.append(Y_train, Y_test, axis=0)
    prev_arr = np.append(train_prev_close, test_prev_close, axis=0)

    return X_arr, Y_arr, prev_arr


def round_to_nearest_0_05(value):
    return np.round(value * 20) / 20
