import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from time import time


class MyANN:
    def __init__(self, ticker: str, interval: str) -> None:
        self.ticker = ticker
        self.interval = interval

    def if_in_half(self, check_datetime, which_half: int) -> bool:
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

    def if_in_zone(self, check_datetime) -> bool:
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

    def if_in_first_half(self, check_datetime):
        return self.if_in_half(check_datetime, which_half=0)

    def if_in_second_half(self, check_datetime):
        return self.if_in_half(check_datetime, which_half=1)

    def round_decimals_2(self, num):
        return round(num, 2)

    def get_data_df(self, which_half: str) -> pd.DataFrame:
        df = pd.read_csv(self.get_csv_file_path())

        if which_half == "full_zone":
            df["to_add"] = df["Datetime"].apply(lambda x: self.if_in_zone(x))
        elif which_half == "first_half":
            df["to_add"] = df["Datetime"].apply(lambda x: self.if_in_first_half(x))
        elif which_half == "second_half":
            df["to_add"] = df["Datetime"].apply(lambda x: self.if_in_second_half(x))

        df["high"] = df["High"].apply(lambda x: self.round_decimals_2(x))
        df["low"] = df["Low"].apply(lambda x: self.round_decimals_2(x))
        df["open"] = df["Open"].apply(lambda x: self.round_decimals_2(x))
        df["close"] = df["Close"].apply(lambda x: self.round_decimals_2(x))

        new_2 = df[df["to_add"] == True].copy()

        return new_2[
            [
                "Datetime",
                "high",
                "low",
                "open",
                "close",
            ]
        ]

    def get_csv_file_path(self) -> str:
        file_path = f"./././data_stock_price_yf/{self.interval} data/{self.ticker} - {self.interval}.csv"
        return file_path

    def train_test_split(
        self,
        df_whole: pd.DataFrame,
        test_size_proportion: float,
    ) -> list(pd.DataFrame):
        # steps 1:
        # check number of days in the data.
        # find the date where training data becomes, test data.
        # split the data into 2 list of data frames
        #   each item in that list will be a dataframe, that data frame will only contain data about a single data only.
        #    where one is fully training data and other is fully testing data.

        pass


def main():
    obj = MyANN(ticker="ADANIPORTS.NS", interval="1m")

    df = pd.read_csv(obj.get_csv_file_path())

    print(df.head())

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
