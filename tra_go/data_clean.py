from copy import deepcopy
from datetime import datetime, time, timedelta

import pandas as pd
import pytz


class DataCleanerZero:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval

        self.data_raw = self.get_data_all_df()

        self.data_cleaned = self.data_cleaning()

        self.save_cleaned_data()

    def get_csv_file_path(self) -> str:
        file_path = f"./data_z/nse/{self.interval}/{self.symbol} - {self.interval}.csv"
        return file_path

    def get_data_all_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.get_csv_file_path())

        df["open"] = df["open"].apply(lambda x: round(number=x, ndigits=2))
        df["close"] = df["close"].apply(lambda x: round(number=x, ndigits=2))
        df["high"] = df["high"].apply(lambda x: round(number=x, ndigits=2))
        df["low"] = df["low"].apply(lambda x: round(number=x, ndigits=2))

        df.rename(columns={"date": "datetime"}, inplace=True)

        return df[
            [
                "datetime",
                "open",
                "close",
                "high",
                "low",
            ]
        ]

    def data_cleaning(self) -> pd.DataFrame:
        # start time = 0915
        # last time = 1529
        # total minutes = 375

        df = self.data_raw.sort_values(by="datetime", ascending=True)

        # remove diwali murath trading rows
        df["is_regular"] = df["datetime"].apply(lambda x: self.is_in_regular_hours(x))
        df = df[df["is_regular"] == True].copy(deep=True)

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
            all_datetimes_required.extend(
                (first_datetime + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S%z") for i in range(375)
            )

        all_datetimes_in_data = []
        for index, row in df.iterrows():
            all_datetimes_in_data.append(
                datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S%z").strftime("%Y-%m-%d %H:%M:%S%z"),
            )

        # make a set of all datetime to be there
        # and what datetime are actually there
        # finding missing ones, using set
        # add them as zeros in correct datetimes, sort df
        # put previous values in them in palce of zeros

        missing_datetimes = set(all_datetimes_required) - set(all_datetimes_in_data)

        add_df_rows = []
        for d in missing_datetimes:
            dict_1 = {"datetime": d, "open": 0, "close": 0, "high": 0, "low": 0}

            add_df_rows.append(deepcopy(dict_1))
            dict_1.clear()

        new_df = pd.DataFrame(add_df_rows)
        df = pd.concat([df, new_df], ignore_index=True)

        df = df.sort_values(by="datetime", ascending=True)
        df.reset_index(drop=True, inplace=True)

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

        missing_indexes = []
        for index, row in df.iterrows():
            if row["open"] == 0:
                missing_indexes.append(index)

        df["date"] = df["datetime"].apply(lambda x: to_date_str(x))

        return df[["datetime", "open", "close", "high", "low"]]

    def save_cleaned_data(self) -> None:
        self.data_cleaned.to_csv(
            f"./data_cleaned/{self.interval}/{self.symbol} - {self.interval}.csv",
            index=False,
        )

    def is_in_regular_hours(self, check_datetime) -> bool:
        time_open = time(9, 15, 0)
        time_close = time(15, 29, 0)

        time_check = datetime.strptime(check_datetime, "%Y-%m-%d %H:%M:%S%z").time()

        if time_open <= time_check and time_check <= time_close:
            return True
        return False


def to_date_str(datetime_1) -> str:
    date_obj = datetime.strptime(datetime_1, "%Y-%m-%d %H:%M:%S%z").date()

    return date_obj.strftime("%Y-%m-%d")


if __name__ == "__main__":
    symbol = "ICICIBANK"
    interval = "1m"

    DataCleanerZero(symbol=symbol, interval=interval)

    print("Data Cleaning Done.")
