from copy import deepcopy
from datetime import datetime, timedelta

import keras_model as km
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from keras.utils import custom_object_scope
from main import NUMBER_OF_EPOCHS
from tensorflow import keras


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

    new_2 = df[df["to_add"] is True].copy(deep=True)
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
    if ticker == "CCI":
        ticker = "ICICIBANK.NS"

    df = pd.read_csv(get_csv_file_path(ticker, interval))

    df["open"] = df["Open"].apply(lambda x: round(number=x, ndigits=2))
    df["close"] = df["Close"].apply(lambda x: round(number=x, ndigits=2))
    df["high"] = df["High"].apply(lambda x: round(number=x, ndigits=2))
    df["low"] = df["Low"].apply(lambda x: round(number=x, ndigits=2))

    df.rename(columns={"Datetime": "datetime"}, inplace=True)

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
    file_path = f"./data_stock_price_yf/{interval} data/{ticker} - {interval}.csv"
    return file_path


def data_cleaning(df) -> pd.DataFrame:
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
        # all x and y is transformed to avg_price ,band_height
        train_y_temp = by_date_df_array(df_train_y[selected_columns_2])
        test_y_temp = by_date_df_array(df_test_y[selected_columns_2])

        train_y = array_transform_avg_band(train_y_temp)
        test_y = array_transform_avg_band(test_y_temp)

        x_close = last_close_value(df_x)

    return ((train_x, train_y), (test_x, test_y), x_close)


def last_close_value(df: pd.DataFrame) -> np.array:
    res = np.array([])

    for i in range(len(df) // 132):
        res = np.append(res, df.iloc[i * 132 + 131]["close"])

    return res


def train_test_split_2_mods(
    data_df,
    interval,
    y_type,
    test_size=0.2,
) -> [pd.DataFrame]:
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

        # print(round((high / low - 1) * 100, 2))

    return res


def by_date_df_array(df: pd.DataFrame) -> np.array:
    res = []
    full_rows = []
    for index, row in df.iterrows():
        x = row.values.tolist()
        full_rows.append(x)

    for i in range(len(full_rows) // 132):
        res.append(deepcopy(full_rows[i * 132 : (i + 1) * 132]))

    res_1 = np.array([np.array(xi) for xi in res])

    return res_1


def array_transform_avg_band(array: np.ndarray) -> np.array:
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

    res: np.ndarray = np.empty((0, number_of_points, len(columns)))

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
) -> np.array:
    df = data_cleaning(data_df)
    # getting clean and my zone data
    df = data_scaling(df)
    # getting scaled data according to previous day closing price, in percentages terms
    df = data_my_zone(df, interval=interval)

    data_df_x, data_df_y = data_split_x_y_new(df=df, interval=interval)

    arr_x = split_by_date(data_df=data_df_x, interval=interval, columns=columns)
    arr_y = split_by_date(data_df=data_df_y, interval=interval, columns=columns)

    arr_x = arr_x.astype(np.float32)
    arr_y = arr_y.astype(np.float32)

    return arr_x, arr_y


def custom_evaluate_safety_factor_band_2_3(
    X_test,
    Y_test,
    x_close: np.ndarray,
    y_type: str,
    testsize: float = 0,
    now_datetime: str = "2020-01-01 00-00",
):
    # convert y_test to same format as y_pred
    with custom_object_scope(
        {
            "metric_new_idea_2": km.metric_new_idea_2,
            "metric_band_base_percent": km.metric_band_base_percent,
            "metric_loss_comp_2": km.metric_loss_comp_2,
            "metric_band_hl_wrongs_percent": km.metric_band_hl_wrongs_percent,
            "metric_band_avg_correction_percent": km.metric_band_avg_correction_percent,
            "metric_band_average_percent": km.metric_band_average_percent,
            "metric_band_height_percent": km.metric_band_height_percent,
            "metric_win_percent": km.metric_win_percent,
            "metric_pred_capture_percent": km.metric_pred_capture_percent,
            "metric_win_pred_capture_percent": km.metric_win_pred_capture_percent,
        },
    ):
        model = keras.models.load_model(
            f"training/models/model - {now_datetime} - {y_type}",
        )
        model.summary()

    y_pred = model.predict(X_test)

    # only 2 columns are needed
    Y_test = Y_test[:, :, :2]

    SKIP_FIRST_PERCENTILE = 0.2

    # now both y arrays transformed to (l,h) type
    # sf = 0.4, 0.5, 0.6
    y_pred = transform_y_array(
        y_pred,
        safety_factor=0.8,
        skip_first_percentile=SKIP_FIRST_PERCENTILE,
    )

    Y_test = transform_y_array(
        Y_test,
        safety_factor=1,
        skip_first_percentile=SKIP_FIRST_PERCENTILE,
    )

    function_make_win_graph_2(
        y_true=Y_test,
        x_close=x_close,
        y_pred=y_pred,
        testsize=testsize,
        y_type=y_type,
        now_datetime=now_datetime,
    )

    # function_error_132_graph(y_pred=y_pred, y_test=Y_test, now_datetime=now_datetime, y_type=y_type)

    return


def transform_y_array(
    y_arr: np.ndarray,
    use_band_height: bool = True,
    safety_factor: float = 1,
    skip_first_percentile: float = 0,
) -> np.ndarray:
    first_non_eiminated_element: int = int(skip_first_percentile * y_arr.shape[1])

    res: np.ndarray = np.zeros(y_arr.shape)

    if use_band_height:
        res[:, :, 0] = y_arr[:, :, 0] - y_arr[:, :, 1] * safety_factor / 2
        res[:, :, 1] = y_arr[:, :, 0] + y_arr[:, :, 1] * safety_factor / 2
    else:
        res[:, :, 0] = y_arr[:, :, 0]
        res[:, :, 1] = y_arr[:, :, 0]

    for i in range(first_non_eiminated_element):
        res[:, i, :] = y_arr[:, first_non_eiminated_element, :]

    return res


def function_error_132_graph(y_pred, y_test, now_datetime, y_type):
    error_a = np.abs(y_pred - y_test)

    new_array = np.empty((0, 2))

    # average error np array
    for i in range(error_a.shape[1]):
        low = error_a[:, i, 0].sum()
        high = error_a[:, i, 1].sum()

        to_add_array = np.array([low / error_a.shape[0], high / error_a.shape[0]])

        new_array = np.concatenate((new_array, np.array([to_add_array])), axis=0)

    y1 = new_array[:, 0] * 100
    y2 = new_array[:, 1] * 100

    # Create x-axis values
    x = np.arange(len(new_array))

    fig = plt.figure(figsize=(16, 9))

    plt.plot(x, y1, label="low Δ")
    plt.plot(x, y2, label="high Δ")

    plt.title(
        f" name: {now_datetime}\n"
        + f"NUMBER_OF_NEURONS = {km.NUMBER_OF_NEURONS}  "
        + f"NUMBER_OF_LAYERS = {km.NUMBER_OF_LAYERS}\n"
        + f"NUMBER_OF_EPOCHS = {NUMBER_OF_EPOCHS} | "
        + f"INITIAL_DROPOUT = {km.INITIAL_DROPOUT} | "
        + f"WEIGHT_FOR_MEA = {km.WEIGHT_FOR_MEA}",
        fontsize=20,
    )

    # Set labels and title
    plt.xlabel("serial", fontsize=15)
    plt.ylabel("perc", fontsize=15)
    plt.legend(fontsize=15)
    filename = f"training/graphs/{y_type} - {now_datetime} - band_2 abs.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
    return


def function_make_win_graph_2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_close: np.ndarray,
    y_type: str,
    now_datetime: str,
    testsize: float,
):
    min_pred: np.ndarray = np.min(y_pred[:, :, 0], axis=1)
    max_pred: np.ndarray = np.max(y_pred[:, :, 1], axis=1)

    min_true: np.ndarray = np.min(y_true[:, :, 0], axis=1)
    max_true: np.ndarray = np.max(y_true[:, :, 1], axis=1)

    min_pred_index: np.ndarray = np.argmin(y_pred[:, :, 0], axis=1)
    max_pred_index: np.ndarray = np.argmax(y_pred[:, :, 1], axis=1)

    buy_order_pred: np.ndarray = np.all([max_pred_index > min_pred_index], axis=0)

    min_true_index: np.ndarray = np.argmin(y_true[:, :, 0], axis=1)
    max_true_index: np.ndarray = np.argmax(y_true[:, :, 1], axis=1)

    buy_order_true: np.ndarray = np.all([max_true_index > min_true_index], axis=0)

    valid_actual: np.ndarray = np.all([max_true > min_true], axis=0)

    valid_pred: np.ndarray = np.all([max_pred > min_pred], axis=0)

    close_below_band: np.ndarray = np.all(
        [
            x_close < min_pred,
            valid_pred,
        ],
        axis=0,
    )

    close_above_band: np.ndarray = np.all(
        [
            x_close > max_pred,
            valid_pred,
        ],
        axis=0,
    )

    close_in_band: np.ndarray = np.all(
        [
            x_close > min_pred,
            x_close < max_pred,
            valid_pred,
        ],
        axis=0,
    )

    for i in range(len(min_pred)):
        if not close_in_band[i]:
            band_val: float = max_pred[i] - min_pred[i]

            if close_below_band[i]:
                # print(i, "below")
                buy_order_pred[i] = True
                min_pred[i] = x_close[i]
                max_pred[i] = x_close[i] + band_val

            elif close_above_band[i]:
                # print(i)
                buy_order_pred[i] = False
                max_pred[i] = x_close[i]
                min_pred[i] = x_close[i] - band_val

    pred_average: np.ndarray = (max_pred + min_pred) / 2

    valid_min: np.ndarray = np.all([min_pred > min_true], axis=0)

    valid_max: np.ndarray = np.all([max_true > max_pred], axis=0)

    min_inside: np.ndarray = np.all(
        [
            max_true > min_pred,
            valid_min,
        ],
        axis=0,
    )

    max_inside: np.ndarray = np.all(
        [
            max_pred > min_true,
            valid_max,
        ],
        axis=0,
    )

    wins: np.ndarray = np.all(
        [
            min_inside,
            max_inside,
            valid_pred,
        ],
        axis=0,
    )

    real_wins: np.ndarray = np.all(
        [
            wins,
            buy_order_pred == buy_order_true,
        ],
        axis=0,
    )

    correct_orders: np.ndarray = np.all(
        [
            buy_order_pred == buy_order_true,
        ],
        axis=0,
    )

    average_in: np.ndarray = np.all(
        [
            max_true > pred_average,
            pred_average > min_true,
        ],
        axis=0,
    )

    simulation(min_pred, max_pred, buy_order_pred, y_true)

    # fraction_correct_orders = np.mean(correct_orders.astype(np.float32))

    # fraction_valid_actual = np.mean(valid_actual.astype(np.float32))

    # fraction_valid_pred = np.mean(valid_pred.astype(np.float32))

    # fraction_valid_max = np.mean(valid_max.astype(np.float32))

    # fraction_valid_min = np.mean(valid_min.astype(np.float32))

    # fraction_max_inside = np.mean(max_inside.astype(np.float32))

    # fraction_min_inside = np.mean(min_inside.astype(np.float32))

    # fraction_average_in = np.mean(average_in.astype(np.float32))

    # fraction_win = np.mean(wins.astype(np.float32))

    # fraction_real_win = np.mean(real_wins.astype(np.float32))

    # all_days_pro_arr: np.ndarray = (max_pred / min_pred) * wins.astype(np.float32)
    # all_days_pro_arr_non_zero = all_days_pro_arr[all_days_pro_arr != 0]

    # all_days_pro_cummulative_val: float = np.prod(all_days_pro_arr_non_zero)

    # all_days_pro_arr_real: np.ndarray = (max_pred / min_pred) * real_wins.astype(np.float32)
    # all_days_pro_arr_non_zero_real = all_days_pro_arr[all_days_pro_arr_real != 0]

    # all_days_pro_cummulative_val_real: float = np.prod(all_days_pro_arr_non_zero_real)

    # pred_capture_arr: np.ndarray = (max_pred / min_pred - 1) * wins.astype(np.float32)

    # total_capture_possible_arr: np.ndarray = max_true / min_true - 1

    # pred_capture_ratio: float = np.sum(pred_capture_arr) / np.sum(total_capture_possible_arr)

    # pred_capture_percent_str: str = "{:.2f}".format(pred_capture_ratio * 100)

    # win_percent_str: str = "{:.2f}".format(fraction_win * 100)

    # real_win_percent_str: str = "{:.2f}".format(fraction_real_win * 100)

    # average_in_percent_str: str = "{:.2f}".format(fraction_average_in * 100)

    # cdgr: float = pow(all_days_pro_cummulative_val, 1 / len(wins)) - 1

    # pro_250: float = pow(cdgr + 1, 250) - 1
    # pro_250_5: float = pow(cdgr * 5 + 1, 250) - 1
    # pro_250_str: str = "{:.2f}".format(pro_250 * 100)
    # pro_250_5_str: str = "{:.2f}".format(pro_250_5 * 100)

    # cdgr_real: float = pow(all_days_pro_cummulative_val_real, 1 / len(real_wins)) - 1
    # pro_250_real: float = pow(cdgr_real + 1, 250) - 1
    # pro_250_str_real: str = "{:.2f}".format(pro_250_real * 100)

    # y_min = min(np.min(min_pred), np.min(min_true))
    # y_max = max(np.max(max_pred), np.max(max_true))

    # x: list[int] = [i + 1 for i in range(len(max_pred))]

    # fig = plt.figure(figsize=(16, 9))

    # ax = fig.add_subplot(111)

    # plt.axvline(x=int(len(max_true) * (1 - testsize)) - 0.5, color="blue")

    # plt.fill_between(x, min_true, max_true, color="yellow")

    # # plt.scatter(x, list_min_actual, color="orange", s=50)
    # # plt.scatter(x, list_max_actual, color="orange", s=50)

    # plt.plot(x, pred_average, linestyle="dashed", c="red")

    # for i in range(len(wins)):
    #     if wins[i]:
    #         plt.scatter(
    #             x=x[i],
    #             y=y_min - (y_max - y_min) / 100,
    #             c="yellow",
    #             linewidths=2,
    #             marker="^",
    #             edgecolor="red",
    #             s=125,
    #         )

    #     if real_wins[i]:
    #         plt.scatter(
    #             x=x[i],
    #             y=y_min - (y_max - y_min) / 20,
    #             c="green",
    #             linewidths=2,
    #             marker="^",
    #             edgecolor="green",
    #             s=125,
    #         )

    #     if correct_orders[i]:
    #         plt.scatter(
    #             x=x[i],
    #             y=y_min - (y_max - y_min) / 10,
    #             c="blue",
    #             linewidths=2,
    #             marker="^",
    #             edgecolor="blue",
    #             s=125,
    #         )

    #     if valid_pred[i]:
    #         plt.vlines(
    #             x=x[i],
    #             ymin=min_pred[i],
    #             ymax=max_pred[i],
    #             colors="green",
    #         )
    #         ax.set_xlabel("days", fontsize=15)

    # ax.set_ylabel("fraction of prev close", fontsize=15)

    # print("\n\n")
    # print("valid_act\t", round(fraction_valid_actual * 100, 2), " %")
    # print("valid_pred\t", round(fraction_valid_pred * 100, 2), " %")
    # print("max_inside\t", round(fraction_max_inside * 100, 2), " %")
    # print("min_inside\t", round(fraction_min_inside * 100, 2), " %\n")
    # print("average_in\t", average_in_percent_str, " %\n")

    # print("win_days_perc\t", win_percent_str, " %")
    # print("real_win_days_perc\t", real_win_percent_str, " %")
    # print("pred_capture\t", pred_capture_percent_str, " %")

    # print("per_day\t\t", round(cdgr * 100, 4), " %")
    # print("250 days:\t", pro_250_str)
    # print("\nleverage:\t", pro_250_5_str)
    # print("datetime:\t", now_datetime)

    # ax.set_title(
    #     f" name: {now_datetime} \n\n"
    #     + f" real_wins: {real_win_percent_str}% || "
    #     + f" wins: {win_percent_str}% || "
    #     + f" average_in: {average_in_percent_str}% || "
    #     + f" 250 days: {pro_250_str} ||"
    #     + f" 250 days_real: {pro_250_str_real}",
    #     fontsize=20,
    # )

    # filename = f"training/graphs/{y_type} - {now_datetime} - Splot.png"

    # plt.savefig(filename, dpi=300, bbox_inches="tight")

    # print("\n\nNUMBER_OF_NEURONS\t\t", km.NUMBER_OF_NEURONS)
    # print("NUMBER_OF_LAYERS\t\t", km.NUMBER_OF_LAYERS)
    # print("NUMBER_OF_EPOCHS\t\t", NUMBER_OF_EPOCHS)
    # print("INITIAL_DROPOUT\t\t\t", km.INITIAL_DROPOUT)
    # print("WEIGHT_FOR_MEA\t\t\t", km.WEIGHT_FOR_MEA)

    # plt.show()

    return


def simulation(
    min_pred: np.ndarray,
    max_pred: np.ndarray,
    buy_order_pred: list[bool],
    y_true: np.ndarray,
) -> None:
    REWARD_TO_RISK_RATIO = 5

    # 3 order are placed when the similation starts
    #   buy order
    #   sell order
    #   stop_loss_order based on what type of whole order this is - buy/sell
    #       or trend whether max comes first or the min.
    #
    #   when the last tick happends. any pending order is executed that that time.
    #       it will be either partial reward or partial stop_loss
    #
    #

    count_trade_taken: int = 0
    count_trade_taken_and_out: int = 0
    count_stop_loss_hit: int = 0
    count_completed_at_closing: int = 0
    count_expected_trades: int = 0

    total_winings: float = 0

    number_of_days: int = y_true.shape[0]

    for i_day in range(y_true.shape[0]):
        trade_taken: bool = False
        trade_taken_and_out: bool = False
        stop_loss_hit: bool = False

        is_trade_type_buy: bool = buy_order_pred[i_day]

        buy_price: float = min_pred[i_day]
        sell_price: float = max_pred[i_day]
        stop_loss: float = 0

        expected_reward: float = sell_price - buy_price

        net_day_reward: float = 0

        if is_trade_type_buy:
            # pred is up
            stop_loss = buy_price - expected_reward / REWARD_TO_RISK_RATIO
        else:
            # pred is down
            stop_loss = sell_price + expected_reward / REWARD_TO_RISK_RATIO

        for i_tick in range(y_true.shape[1]):
            if is_trade_type_buy:
                # buy trade
                if not trade_taken:
                    if (
                        y_true[i_day, i_tick, 0] < buy_price
                        and buy_price < y_true[i_day, i_tick, 1]
                    ):
                        trade_taken = True

                if trade_taken and not trade_taken_and_out:
                    if (
                        y_true[i_day, i_tick, 0] < sell_price
                        and sell_price < y_true[i_day, i_tick, 1]
                    ):
                        trade_taken_and_out = True
                        net_day_reward = expected_reward

                    elif (
                        y_true[i_day, i_tick, 0] < stop_loss
                        and stop_loss < y_true[i_day, i_tick, 1]
                    ):
                        trade_taken_and_out = True
                        stop_loss_hit = True

                        net_day_reward = -expected_reward / REWARD_TO_RISK_RATIO

            else:
                # sell trade
                if not trade_taken:
                    if (
                        y_true[i_day, i_tick, 0] < sell_price
                        and sell_price < y_true[i_day, i_tick, 1]
                    ):
                        trade_taken = True

                if trade_taken and not trade_taken_and_out:
                    if (
                        y_true[i_day, i_tick, 0] < buy_price
                        and buy_price < y_true[i_day, i_tick, 1]
                    ):
                        trade_taken_and_out = True
                        net_day_reward = expected_reward

                    elif (
                        y_true[i_day, i_tick, 0] < stop_loss
                        and stop_loss < y_true[i_day, i_tick, 1]
                    ):
                        trade_taken_and_out = True
                        stop_loss_hit = True

                        net_day_reward = -expected_reward / REWARD_TO_RISK_RATIO

        if not trade_taken_and_out:
            if is_trade_type_buy:
                # buy trade
                avg_close = (y_true[i_day, -1, 0] + y_true[i_day, -1, 1]) / 2
                net_day_reward = avg_close - buy_price

            else:
                # sell trade
                avg_close = (y_true[i_day, -1, 0] + y_true[i_day, -1, 1]) / 2
                net_day_reward = sell_price - avg_close

        if trade_taken:
            count_trade_taken += 1

        if trade_taken_and_out:
            count_trade_taken_and_out += 1

        if stop_loss_hit:
            count_stop_loss_hit += 1

        if trade_taken and not trade_taken_and_out:
            count_completed_at_closing += 1

        if trade_taken_and_out and not stop_loss_hit:
            count_expected_trades += 1

        total_winings += net_day_reward

        print("{:.6f}".format(net_day_reward))

    print("\n\n")
    print("-" * 30)
    print("\n\n")

    print("number_of_days\t\t", number_of_days, "\n")

    print(
        "percent_trade_taken\t\t",
        "{:.2f}".format(count_trade_taken / number_of_days * 100),
        " %",
    )
    print(
        "percent_trade_taken_and_out\t",
        "{:.2f}".format(count_trade_taken_and_out / number_of_days * 100),
        " %",
    )
    print(
        "percent_stop_loss_hit\t\t",
        "{:.2f}".format(count_stop_loss_hit / number_of_days * 100),
        " %",
    )
    print(
        "percent_completed_at_closing\t",
        "{:.2f}".format(count_completed_at_closing / number_of_days * 100),
        " %",
    )

    print(
        "percent_expected_trades\t\t",
        "{:.2f}".format(count_expected_trades / number_of_days * 100),
        " %",
    )

    total_winings_per_day: float = total_winings / number_of_days
    print("\n\ntotal_winings_per_day\t\t", round(total_winings_per_day, 6))
    print("total_winings_per_day_leverage\t", round(total_winings_per_day * 5, 6))

    print("\n\n250_days\t\t\t", round(pow(1 + total_winings_per_day, 250) - 1, 6))
    print(
        "250_days_leverage\t\t",
        round(pow(1 + total_winings_per_day * 5, 250) - 1, 6),
    )

    print("\n\n")

    return
