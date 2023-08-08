import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from time import time
import pytz
from copy import deepcopy


class MyANN:
    def __init__(self, ticker: str, interval: str) -> None:
        self.ticker = ticker
        self.interval = interval

    def is_in_half(self, check_datetime, which_half: int) -> bool:
        # which_half = 0, means 1st half - input data
        # which_half = 1, means 2nd half - predict data
        datetime_1 = datetime.strptime(
            "2000-01-01 10:00:00+0530", "%Y-%m-%d %H:%M:%S%z"
        )
        time_1 = (datetime_1 + timedelta(minutes=132 * which_half)).time()
        time_check = datetime.strptime(check_datetime, "%Y-%m-%d %H:%M:%S%z").time()

        if self.interval == "1m":
            time_2 = (datetime_1 + timedelta(minutes=132 * (which_half + 1) + 1)).time()
        elif self.interval == "2m":
            time_2 = (datetime_1 + timedelta(minutes=132 * (which_half + 1) + 2)).time()
        elif self.interval == "5m":
            time_2 = (datetime_1 + timedelta(minutes=132 * (which_half + 1) + 5)).time()
        if time_check > time_1 and time_check < time_2:
            return True
        return False

    def to_date(self, datetime_1) -> str:
        return datetime.strptime(datetime_1, "%Y-%m-%d %H:%M:%S%z").date()

    def to_date_str(self, datetime_1) -> str:
        return self.to_date(datetime_1).strftime("%Y-%m-%d")

    def is_same_date(self, datetime_1, check_datetime):
        return self.to_date(datetime_1) == self.to_date(check_datetime)

    def is_in_zone(self, check_datetime) -> bool:
        datetime_1 = datetime.strptime(
            "2000-01-01 10:00:00+0530", "%Y-%m-%d %H:%M:%S%z"
        )
        time_1 = datetime_1.time()
        time_check = datetime.strptime(check_datetime, "%Y-%m-%d %H:%M:%S%z").time()

        if self.interval == "1m":
            time_2 = (datetime_1 + timedelta(minutes=132 * 2 + 1)).time()
        elif self.interval == "2m":
            time_2 = (datetime_1 + timedelta(minutes=132 * 2 + 2)).time()
        elif self.interval == "5m":
            time_2 = (datetime_1 + timedelta(minutes=132 * 2 + 5)).time()

        if time_check > time_1 and time_check < time_2:
            return True
        return False

    def is_in_first_half(self, check_datetime):
        return self.is_in_half(check_datetime, which_half=0)

    def is_in_second_half(self, check_datetime):
        return self.is_in_half(check_datetime, which_half=1)

    def round_decimals_2(self, num):
        return round(num, 2)

    def get_data_df(self, which_half: str) -> pd.DataFrame:
        df = pd.read_csv(self.get_csv_file_path())

        if which_half == "full_zone":
            df["to_add"] = df["Datetime"].apply(lambda x: self.is_in_zone(x))
        elif which_half == "first_half":
            df["to_add"] = df["Datetime"].apply(lambda x: self.is_in_first_half(x))
        elif which_half == "second_half":
            df["to_add"] = df["Datetime"].apply(lambda x: self.is_in_second_half(x))

        df["open"] = df["Open"].apply(lambda x: self.round_decimals_2(x))
        df["close"] = df["Close"].apply(lambda x: self.round_decimals_2(x))
        df["high"] = df["High"].apply(lambda x: self.round_decimals_2(x))
        df["low"] = df["Low"].apply(lambda x: self.round_decimals_2(x))

        new_2 = df[df["to_add"] == True].copy(deep=True)

        return new_2[
            [
                "Datetime",
                "open",
                "close",
                "high",
                "low",
            ]
        ]

    def get_data_all_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.get_csv_file_path())

        df["open"] = df["Open"].apply(lambda x: self.round_decimals_2(x))
        df["close"] = df["Close"].apply(lambda x: self.round_decimals_2(x))
        df["high"] = df["High"].apply(lambda x: self.round_decimals_2(x))
        df["low"] = df["Low"].apply(lambda x: self.round_decimals_2(x))

        return df[
            [
                "Datetime",
                "open",
                "close",
                "high",
                "low",
            ]
        ]

    def get_csv_file_path(self) -> str:
        file_path = f"./././data_stock_price_yf/{self.interval} data/{self.ticker} - {self.interval}.csv"
        return file_path

    def data_clean(
        self,
    ) -> pd.DataFrame:
        # start time = 0915
        # last time = 1529
        # total minutes = 375

        df = self.get_data_all_df()
        df = df.sort_values(by="Datetime", ascending=True)

        df["Dates"] = df["Datetime"].apply(lambda x: self.to_date_str(x))
        all_dates = df["Dates"].unique()

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
                    (first_datetime + timedelta(minutes=i)).strftime(
                        "%Y-%m-%d %H:%M:%S%z"
                    )
                )

        all_datetimes_in_data = []
        for index, row in df.iterrows():
            all_datetimes_in_data.append(
                datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S%z").strftime(
                    "%Y-%m-%d %H:%M:%S%z"
                )
            )

        # make a set of all datetime to be there
        # and what datetime are actually there
        # finding missing ones
        # add them as zeros in correct datetimes, sort df
        # put previous values in them if not there.

        all_datetimes_required_set = set(all_datetimes_required)
        all_datetimes_in_data_set = set(all_datetimes_in_data)

        missing_datetimes = all_datetimes_required_set - all_datetimes_in_data_set

        add_df_rows = []
        for d in missing_datetimes:
            dict_1 = {
                "Datetime": d,
                "open": 0,
                "close": 0,
                "high": 0,
                "low": 0,
            }
            add_df_rows.append(deepcopy(dict_1))
            dict_1.clear()

        new_df = pd.DataFrame(add_df_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values(by="Datetime", ascending=True)
        print(df.columns)
        df.drop("Dates", axis=1, inplace=True)
        print(df.columns)

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
                        print(df.at[index, "Datetime"])

        # then divide accordingly into 2 zones

        return df

    def train_test_split(
        self,
        test_size_proportion: float,
    ) -> [pd.DataFrame]:
        # separate into 132, 132 entries df. for train and test df.

        # divide the price data of that day by the cloding price of the previous day.
        # for the very first day of the dataset - divide the prices by the opening price.

        list_df = self.data_clean()

        if test_size_proportion:
            test_size_proportion = 0.2

        test_num = round(len(list_df) * test_size_proportion)

        return (
            list_df[:test_num],
            list_df[test_num:],
            list_df[:test_num],
            list_df[test_num:],
        )


def main():
    obj = MyANN(ticker="ADANIPORTS.NS", interval="1m")

    X_train, X_test, Y_train, Y_test = obj.train_test_split(test_size_proportion=0.2)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     df[["age", "affordibility"]], df.bought_insurance, test_size=0.2, random_state=25
    # )

    # X_train_scaled = X_train.copy()
    # X_train_scaled["age"] = X_train_scaled["age"] / 100

    # X_test_scaled = X_test.copy()
    # X_test_scaled["age"] = X_test_scaled["age"] / 100

    # model = keras.Sequential(
    #     [
    #         keras.layers.Dense(
    #             1,
    #             input_shape=(2,),
    #             activation="relu",
    #             kernel_initializer="ones",
    #             bias_initializer="zeros",
    #         )
    #     ]
    # )

    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # model.fit(X_train_scaled, y_train, epochs=5000)

    # model.evaluate(X_test_scaled, y_test)

    # model.predict(X_test_scaled)

    # coef, intercept = model.get_weights()


if __name__ == "__main__":
    time_1 = time()
    main()
    print(f"time taken = {round(time() - time_1, 2)} sec")
