from copy import deepcopy
from datetime import datetime, time, timedelta

import pandas as pd
import pytz

from script_2 import nifty50_symbols


class DataCleanerZero:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval

        self.data_raw = self.get_data_all_df()

        self.data_regular = self.data_clean_1()

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

        return df[["datetime", "open", "close", "high", "low"]]

    def data_clean_1(self) -> pd.DataFrame:
        # step 1: only regular data

        df = self.data_raw.sort_values(by="datetime", ascending=True)

        # remove diwali murath trading rows
        df["is_regular_hours"] = df["datetime"].apply(lambda x: self.is_in_regular_hours(x))
        df = df[df["is_regular_hours"] == True].copy(deep=True)

        df["is_weekday"] = df["datetime"].apply(lambda x: is_weekday_datetime_str(x))
        df = df[df["is_weekday"] == True].copy(deep=True)

        # step 2: datetimes with second != 0, putting them to be zero, if second=0 does not exist

        df["datetime_obj"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S%z")
        non_zero_second_indexes = df[df["datetime_obj"].dt.second != 0].index.tolist()

        for index in non_zero_second_indexes:
            datetime_obj = datetime.strptime(df.at[index, "datetime"], "%Y-%m-%d %H:%M:%S%z")
            new_datetime = datetime_obj.replace(second=0)
            new_datetime_str = new_datetime.strftime("%Y-%m-%d %H:%M:%S%z")

            if df["datetime"].isin([new_datetime_str]).any():
                dict_1 = {
                    "datetime": new_datetime_str,
                    "open": df.at[index, "open"],
                    "close": df.at[index, "close"],
                    "high": df.at[index, "high"],
                    "low": df.at[index, "low"],
                }

                df = pd.concat([df, pd.DataFrame(dict_1, index=[0])], ignore_index=True)

            df.drop(index, inplace=True)

        df = df.sort_values(by="datetime", ascending=True)
        df.reset_index(drop=True, inplace=True)

        # df["datetime_obj"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S%z")
        # duplicate_non_zero_second_indexes = df[df["datetime_obj"].dt.second != 0].index.tolist()

        return df[["datetime", "open", "close", "high", "low"]]

    def data_cleaning(self) -> pd.DataFrame:
        # start time = 0915
        # last time = 1529
        # total minutes = 375

        df = self.data_regular.copy(deep=True)

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

        missing_indexes.sort()

        missing_rows = len(missing_indexes)
        for index in missing_indexes:
            ref_index = self.get_non_zero_index(index, missing_indexes, len(df) - 1)

            df.at[index, "open"] = df.at[ref_index, "open"]
            df.at[index, "close"] = df.at[ref_index, "close"]
            df.at[index, "high"] = df.at[ref_index, "high"]
            df.at[index, "low"] = df.at[ref_index, "low"]

            missing_rows -= 1

        print("missing_indexes \t= ", len(missing_indexes))

        df["date"] = df["datetime"].apply(lambda x: to_date_str(x))

        return df[["datetime", "open", "close", "high", "low"]]

    def get_non_zero_index(self, index: int, missing_indexes: list[int], max_index: int) -> int:
        left_index: int = index - 1
        right_index: int = index + 1

        while not (
            (left_index not in missing_indexes and left_index >= 0)
            or (right_index not in missing_indexes and right_index <= max_index)
        ):
            # print(left_index, right_index)
            left_index -= 1
            right_index += 1

        if left_index not in missing_indexes and left_index >= 0:
            return left_index
        if right_index not in missing_indexes and right_index <= max_index:
            return right_index

    def save_cleaned_data(self) -> None:
        print(self.symbol, "\t= ", len(self.data_cleaned) / 375)

        self.data_cleaned.to_csv(f"./data_cleaned/{self.interval}/{self.symbol} - {self.interval}.csv", index=False)

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


def is_weekday_datetime_str(datetime_str: str):
    date = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S%z")
    # weekday() returns the day of the week as an integer (Monday is 0, Sunday is 6)
    return date.weekday() < 5


class DataScalerZero:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval

        self.data_cleaned = pd.read_csv(self.get_csv_file_path())

        self.data_scaled = self.data_scaling()

        self.save_scaled_data()

    def get_csv_file_path(self) -> str:
        file_path = f"./data_cleaned/{self.interval}/{self.symbol} - {self.interval}.csv"

        return file_path

    def data_scaling(self) -> pd.DataFrame:
        df = self.data_cleaned.copy(deep=True)

        df["real_close"] = df["close"] + 1

        return df[["open", "close", "high", "low", "real_close"]]

    def save_scaled_data(self) -> None:
        self.data_scaled.to_csv(f"./data_traning/{self.symbol} - {self.interval}.csv", index=False)


if __name__ == "__main__":
    interval = "1m"

    def func(symbol: str):
        print("\n" * 2)
        print(symbol)

        DataCleanerZero(symbol=symbol, interval=interval)
        print("Data Cleaning Done.")

        # DataScalerZero(symbol=symbol, interval=interval)
        # print("Data Scaling Done.")

    for symbol in nifty50_symbols:
        func(symbol=symbol)

    # arr = np.array(nifty50_symbols)
    # split_arr = np.array_split(arr, 3)

    # def func_1():
    #     for symbol in split_arr[0]:
    #         func(symbol)

    # def func_2():
    #     for symbol in split_arr[1]:
    #         func(symbol)

    # def func_3():
    #     for symbol in split_arr[2]:
    #         func(symbol)

    # t1 = threading.Thread(target=func_1)
    # t2 = threading.Thread(target=func_2)
    # t3 = threading.Thread(target=func_3)

    # # Start threads
    # t1.start()
    # t2.start()
    # t3.start()

    # # Wait for both threads to finish
    # t1.join()
    # t2.join()
    # t3.join()
