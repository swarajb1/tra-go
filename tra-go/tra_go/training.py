import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from time import time


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

    def get_data_all_df(self, which_half: str) -> pd.DataFrame:
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
    ) -> list:
        # list containing dict key=data, data= day's dataframe.

        # check all dates in the df
        # then clean - if a datetime entry is missing or the value in 4 price columns is not there, put average values.
        # full zone
        # separate

        # res = [{"2000-01-01": df_that_day}]

        res = []
        df = self.get_data_all_df(which_half="full_zone")
        print(df)

        df_sorted = df.sort_values(by="Datetime", ascending=True)

        datetime_prev = datetime.strptime(
            df_sorted.loc[0]["Datetime"], "%Y-%m-%d %H:%M:%S%z"
        )

        all_dates = []
        for index, row in df_sorted.iterrows():
            date_new = datetime.strptime(
                row["Datetime"], "%Y-%m-%d %H:%M:%S%z"
            ).strftime("%Y-%m-%d")
            if date_new not in all_dates:
                all_dates.append(date_new)

        all_datetimes = []
        # start time = 0915
        # lasat time = 1529
        # total minutes = 375
        for date in all_dates:
            datetime_new = datetime.strptime(date, "%Y-%m-%d")
            datetime_1 = datetime_1 + timedelta(minutes=1)

        # NOTE: assuming that 915 datetime exists for all days for all stocks.
        # for index, row in df_sorted.iterrows():
        #     first_minute = (
        #         datetime.strptime(row["Datetime"], "%Y-%m-%d %H:%M:%S%z")
        #         .time()
        #         .strftime("%H:%M:%S")
        #     )
        #     if first_minute != "09:15:00":
        #         datetime_prev_plus_1 = datetime_prev + timedelta(minutes=1)
        #         if not self.is_same_date(
        #             row["Datetime"],
        #             datetime_prev_plus_1.strftime("%Y-%m-%d %H:%M:%S%z"),
        #         ):
        #             print(row["Datetime"])
        #             df.loc[df_sorted.index[0]] = df_sorted.iloc[0]
        #             print(df_sorted.iloc[0])
        #     else:

        # df_sorted = df.sort_values(by="Datetime", ascending=True)

        # # Sort the DataFrame based on the 'Age' column in ascending order
        # df_sorted = df.sort_values(by='Age', ascending=True)

        # print(df_sorted)

        # for datetime in df[["Datetime"]].values:
        #     if self.is_in_zone(datetime[0]):
        #         date_now = self.to_date(datetime[0])
        #         if date_now not in data.keys():
        #             data[date_now] = 0
        #         data[date_now] += 1

        # for i in data.keys():
        #     print(i, "\t", data[i] - 264)

        return res

    def train_test_split(
        self,
        test_size_proportion: float,
    ) -> [pd.DataFrame]:
        # separate into 132, 132 entries df. for train and test df.

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
