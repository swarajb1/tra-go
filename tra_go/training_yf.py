from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from copy import deepcopy
import matplotlib.pyplot as plt


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
            all_datetimes_required.append((first_datetime + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S%z"))

    all_datetimes_in_data = []
    for index, row in df.iterrows():
        all_datetimes_in_data.append(
            datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S%z").strftime("%Y-%m-%d %H:%M:%S%z")
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
    df["is_input"] = df["datetime"].apply(lambda x: is_in_half(x, which_half=0, interval=interval))
    df["is_output"] = df["datetime"].apply(lambda x: is_in_half(x, which_half=1, interval=interval))

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

    selected_columns_1 = ["open", "close", "high", "low"]
    selected_columns_2 = ["high", "low"]

    train_x = by_date_df_array(df_train_x[selected_columns_1])
    test_x = by_date_df_array(df_test_x[selected_columns_1])

    if y_type == "hl":
        train_y = points_hl(df_train_y[selected_columns_2])
        test_y = points_hl(df_test_y[selected_columns_2])

    elif y_type == "band":
        train_y = by_date_df_array_hl(df_train_y[selected_columns_2])
        test_y = by_date_df_array_hl(df_test_y[selected_columns_2])

    return (
        (train_x, train_y),
        (test_x, test_y),
    )


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


def by_date_df_array(df) -> [pd.DataFrame]:
    res = []
    full_rows = []
    for index, row in df.iterrows():
        x = row.values.tolist()
        full_rows.append(x)

    for i in range(len(full_rows) // 132):
        res.append(deepcopy(full_rows[i * 132 : (i + 1) * 132]))

    res_1 = np.array([np.array(xi) for xi in res])

    return res_1


def by_date_df_array_hl(df) -> [pd.DataFrame]:
    res = []
    full_rows = []
    for index, row in df.iterrows():
        x = row[["high", "low"]].values.tolist()
        full_rows.append(x)

    for i in range(len(full_rows) // 132):
        res.append(deepcopy(full_rows[i * 132 : (i + 1) * 132]))

    res_1 = np.array([np.array(xi) for xi in res])

    return res_1


def evaluate(self) -> float:
    # TODOO:
    # 0 - direct values comparision with prediction 132x4, to 132x4
    # 1 - predicted high low is inside the actual high low
    # 2 - predicted high low, with safety_factor is the actual high low
    return 0


def custom_evaluate_full_envelope(model, X_test, Y_test):
    y_pred = model.predict(X_test)

    num_days = y_pred.shape[0]
    # high is 3rd column, low is 4th column

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(num_days):
        # i  -> day
        # for 1st day

        min_pred = y_pred[i, 0, 1]
        max_pred = y_pred[i, 0, 0]

        min_actual = Y_test[i, 0, 1]
        max_actual = Y_test[i, 0, 0]

        for j in range(y_pred.shape[1]):
            # j -> time
            min_pred = min(min_pred, y_pred[i, j, 1])
            max_pred = max(max_pred, y_pred[i, j, 0])

            min_actual = min(min_actual, Y_test[i, j, 1])
            max_actual = max(max_actual, Y_test[i, j, 0])

        list_min_pred.append(min_pred)
        list_max_pred.append(max_pred)
        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

    error_low = mean_absolute_error(list_min_pred, list_min_actual)
    error_high = mean_absolute_error(list_max_pred, list_max_actual)
    print("mean_absolute_error")
    print(round(error_high, 6), round(error_low, 6))

    return


def custom_evaluate_safety_factor(
    model,
    X_test,
    Y_test,
    now_datetime: str,
    y_type: str,
    safety_factor: float,
):
    if y_type == "hl":
        return custom_evaluate_safety_factor_hl(
            model=model,
            X_test=X_test,
            Y_test=Y_test,
            y_type=y_type,
            now_datetime=now_datetime,
            safety_factor=safety_factor,
        )

    elif y_type == "band":
        return custom_evaluate_safety_factor_band(
            model=model,
            X_test=X_test,
            Y_test=Y_test,
            y_type=y_type,
            now_datetime=now_datetime,
            safety_factor=safety_factor,
        )


def custom_evaluate_safety_factor_band(
    model,
    X_test,
    Y_test,
    y_type,
    now_datetime,
    safety_factor,
):
    y_pred = model.predict(X_test)

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(y_pred.shape[0]):
        # i  -> day
        # for 1st day
        min_pred = y_pred[i, 0, 1]
        max_pred = y_pred[i, 0, 0]

        min_actual = Y_test[i, 0, 1]
        max_actual = Y_test[i, 0, 0]

        for j in range(int(y_pred.shape[1] * 0.25), int(y_pred.shape[1] * 0.75)):
            # j -> time
            min_pred = min(min_pred, y_pred[i, j, 1])
            max_pred = max(max_pred, y_pred[i, j, 0])

            min_actual = min(min_actual, Y_test[i, j, 1])
            max_actual = max(max_actual, Y_test[i, j, 0])

        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

        average_pred = (min_pred + max_pred) / 2
        min_t = average_pred + (min_pred - average_pred) * safety_factor
        max_t = average_pred + (max_pred - average_pred) * safety_factor

        list_min_pred.append(min_t)
        list_max_pred.append(max_t)

    win_percent = function_rest(
        list_max_actual=list_max_actual,
        list_min_actual=list_min_actual,
        list_max_pred=list_max_pred,
        list_min_pred=list_min_pred,
        y_type=y_type,
        safety_factor=safety_factor,
        now_datetime=now_datetime,
    )

    function_2(y_pred=y_pred, y_test=Y_test)

    return


def function_2(y_pred, y_test):
    error_a = np.abs(y_pred - y_test)

    new_array = np.array([])

    # average error np array
    for i in range(error_a.shape[1]):
        high = 0
        low = 0
        for j in range(error_a.shape[0]):
            high += error_a[j, i, 0]
            low += error_a[j, i, 1]

        to_add_array = np.array([high / error_a.shape[0], low / error_a.shape[0]])

        if new_array.size == 0:
            new_array = np.array([to_add_array])
        else:
            new_array = np.append(new_array, [to_add_array], axis=0)

    y1 = new_array[:, 0]
    y2 = new_array[:, 1]

    # Create x-axis values
    x = np.arange(len(new_array))

    plt.figure(figsize=(16, 9))

    # Plot y1 and y2 against x
    plt.plot(x, y1, label="high")
    plt.plot(x, y2, label="low")

    # Set labels and title
    plt.xlabel("serial", fontsize=15)
    plt.ylabel("perc", fontsize=15)
    plt.legend(fontsize=15)

    # plt.plot(new_array)
    plt.show()

    return


def custom_evaluate_safety_factor_hl(
    model,
    X_test,
    Y_test,
    y_type,
    now_datetime,
    safety_factor,
):
    y_pred = model.predict(X_test)

    list_min_pred = []
    list_max_pred = []
    list_min_actual = []
    list_max_actual = []

    for i in range(y_pred.shape[0]):
        # i  -> day
        # for 1st day
        min_pred = y_pred[i, 1]
        max_pred = y_pred[i, 0]

        min_actual = Y_test[i, 1]
        max_actual = Y_test[i, 0]

        list_min_actual.append(min_actual)
        list_max_actual.append(max_actual)

        average_pred = (min_pred + max_pred) / 2
        min_t = average_pred + (min_pred - average_pred) * safety_factor
        max_t = average_pred + (max_pred - average_pred) * safety_factor

        list_min_pred.append(min_t)
        list_max_pred.append(max_t)

    win_percent = function_rest(
        list_max_actual=list_max_actual,
        list_min_actual=list_min_actual,
        list_max_pred=list_max_pred,
        list_min_pred=list_min_pred,
        y_type=y_type,
        safety_factor=safety_factor,
        now_datetime=now_datetime,
    )

    return


def function_rest(
    list_max_actual,
    list_min_actual,
    list_max_pred,
    list_min_pred,
    y_type,
    safety_factor,
    now_datetime,
):
    list_pred_avg = []

    res = []
    valid_pred = []
    valid_act = []
    valid_max = []
    valid_min = []
    is_average_in = []

    for i in range(len(list_max_actual)):
        min_pred = list_min_pred[i]
        max_pred = list_max_pred[i]
        min_actual = list_min_actual[i]
        max_actual = list_max_actual[i]

        average_pred = (min_pred + max_pred) / 2

        list_pred_avg.append(average_pred)

        win = max_pred < max_actual and min_pred > min_actual and max_pred > min_pred

        valid_pred.append(max_pred > min_pred)
        valid_act.append(max_actual > min_actual)
        valid_max.append(max_pred < max_actual)
        valid_min.append(min_pred > min_actual)
        is_average_in.append(average_pred < max_actual and average_pred > min_actual)

        res.append(win)

    pred_num = 0
    for i in valid_pred:
        if i:
            pred_num += 1

    act_num = 0
    for i in valid_act:
        if i:
            act_num += 1

    max_num = 0
    for i in valid_max:
        if i:
            max_num += 1

    min_num = 0
    for i in valid_min:
        if i:
            min_num += 1

    average_in_num = 0
    for i in is_average_in:
        if i:
            average_in_num += 1

    print("valid_act\t", round(act_num / len(valid_act) * 100, 2), " %")
    print("valid_pred\t", round(pred_num / len(valid_pred) * 100, 2), " %")
    print("max_bound\t", round(max_num / len(valid_max) * 100, 2), " %")
    print("min_bound\t", round(min_num / len(valid_min) * 100, 2), " %\n")
    print("average_in\t", round(average_in_num / len(valid_min) * 100, 2), " %\n")

    x = [i + 1 for i in range(len(list_max_actual))]

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(111)

    plt.fill_between(x, list_min_actual, list_max_actual, color="yellow")

    plt.plot(x, list_pred_avg, linestyle="dashed", c="red")

    for i in range(len(list_min_pred)):
        plt.vlines(
            x=x[i],
            ymin=list_min_pred[i],
            ymax=list_max_pred[i],
            colors="green",
        )

    ax.set_xlabel("days", fontsize=15)
    ax.set_ylabel("fraction of prev close", fontsize=15)
    y_min = min(min(list_min_actual), min(list_min_pred))
    y_max = max(max(list_max_actual), max(list_max_pred))

    wins = 0
    total_capture = 0
    pred_capture = 0
    all_days_pro = 1
    for i in range(len(res)):
        total_capture += list_max_actual[i] / list_min_actual[i] - 1
        if res[i]:
            all_days_pro *= list_max_pred[i] / list_min_pred[i]
            pred_capture += list_max_pred[i] / list_min_pred[i] - 1

            wins += 1
            plt.scatter(
                x[i],
                y_min - (y_max - y_min) / 100,
                c="yellow",
                linewidths=2,
                marker="^",
                edgecolor="red",
                s=400,
            )

    win_percent = round((wins / len(res)) * 100, 2)
    cdgr = (pow(all_days_pro, 1 / len(res)) - 1) * 100

    pred_capture_percent = round((pred_capture / total_capture) * 100, 2)

    avg_captured = 0
    if wins != 0:
        avg_captured = "{:.4f}".format(pred_capture / wins * 100)
    pro_250 = "{:.4f}".format(pow(cdgr / 100 + 1, 250) - 1)
    pro_250_5 = "{:.4f}".format(pow(cdgr * 5 / 100 + 1, 250) - 1)

    print("win_days_perc\t", win_percent, " %")
    print("pred capture\t", pred_capture_percent, " %")
    print("per_day\t\t", avg_captured, " %")
    print("250 days:\t", pro_250)
    print("leverage:\t", pro_250_5)
    print("datetime:\t", now_datetime)

    ax.set_title(
        f" name: {now_datetime} \n\n win percent: {win_percent}%  || per_day = {avg_captured}% || pred_capture: {pred_capture_percent}%",
        fontsize=20,
    )

    filename = f"graphs/{y_type} - {now_datetime} - Splot.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    # plt.show()

    return win_percent
