import multiprocessing as mp
import os
from copy import deepcopy
from datetime import datetime, time, timedelta
from functools import partial

import pandas as pd
import pytz
from sklearn.preprocessing import MinMaxScaler
from utils.functions import min_max_scaler

from scripts.script_2 import nifty50_symbols

SAFETY_FACTOR_CPU_CORES_USAGE: int = 2


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
        df["high"] = df["high"].apply(lambda x: round(number=x, ndigits=2))
        df["low"] = df["low"].apply(lambda x: round(number=x, ndigits=2))
        df["close"] = df["close"].apply(lambda x: round(number=x, ndigits=2))

        df.rename(columns={"date": "datetime"}, inplace=True)

        return df[["datetime", "open", "high", "low", "close", "volume"]]

    def data_clean_1(self) -> pd.DataFrame:
        # step 1: only regular data

        df = self.data_raw.sort_values(by="datetime", ascending=True)

        # remove diwali murath trading rows
        df["is_regular_hours"] = df["datetime"].apply(
            lambda x: self.is_in_regular_hours(x),
        )
        df = df[df["is_regular_hours"]].copy(deep=True)

        df["is_weekday"] = df["datetime"].apply(lambda x: is_weekday_datetime_str(x))
        df = df[df["is_weekday"]].copy(deep=True)

        # step 2: datetimes with second != 0, putting them to be zero, if second=0 does not exist

        df["datetime_obj"] = pd.to_datetime(
            df["datetime"],
            format="%Y-%m-%d %H:%M:%S%z",
        )
        non_zero_second_indexes = df[df["datetime_obj"].dt.second != 0].index.tolist()

        for index in non_zero_second_indexes:
            datetime_obj = datetime.strptime(
                df.at[index, "datetime"],
                "%Y-%m-%d %H:%M:%S%z",
            )
            new_datetime = datetime_obj.replace(second=0)
            new_datetime_str = new_datetime.strftime("%Y-%m-%d %H:%M:%S%z")

            if df["datetime"].isin([new_datetime_str]).any():
                dict_1 = {
                    "datetime": new_datetime_str,
                    "open": df.at[index, "open"],
                    "high": df.at[index, "high"],
                    "low": df.at[index, "low"],
                    "close": df.at[index, "close"],
                    "volume": 0,
                }

                df = pd.concat([df, pd.DataFrame(dict_1, index=[0])], ignore_index=True)

            df.drop(index, inplace=True)

        df = df.sort_values(by="datetime", ascending=True)
        df.reset_index(drop=True, inplace=True)

        # df["datetime_obj"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S%z")
        # duplicate_non_zero_second_indexes = df[df["datetime_obj"].dt.second != 0].index.tolist()

        return df[["datetime", "open", "high", "low", "close", "volume"]]

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
                datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S%z").strftime(
                    "%Y-%m-%d %H:%M:%S%z",
                ),
            )

        # make a set of all datetime to be there
        # and what datetime are actually there
        # finding missing ones, using set
        # add them as zeros in correct datetimes, sort df
        # put previous values in them in palce of zeros

        missing_datetimes = set(all_datetimes_required) - set(all_datetimes_in_data)

        add_df_rows = []
        for d in missing_datetimes:
            dict_1 = {
                "datetime": d,
                "open": 0,
                "close": 0,
                "high": 0,
                "low": 0,
                "volume": 0,
            }

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
            df.at[index, "high"] = df.at[ref_index, "high"]
            df.at[index, "low"] = df.at[ref_index, "low"]
            df.at[index, "close"] = df.at[ref_index, "close"]
            df.at[index, "volume"] = 0

            missing_rows -= 1

        print(f"{self.symbol} \t- , missing_indexes \t= ", len(missing_indexes))

        df["date"] = df["datetime"].apply(lambda x: to_date_str(x))

        return df[["datetime", "open", "high", "low", "close", "volume"]]

    def get_non_zero_index(
        self,
        index: int,
        missing_indexes: list[int],
        max_index: int,
    ) -> int:
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

        # This should never happen, but added for type safety
        return 0

    def save_cleaned_data(self) -> None:
        print(self.symbol, "\t= ", len(self.data_cleaned) / 375)

        os.makedirs(f"./data_cleaned/{self.interval}", exist_ok=True)

        file_path: str = os.path.join(
            "./data_cleaned",
            self.interval,
            f"{self.symbol} - {self.interval}.csv",
        )

        self.data_cleaned.to_csv(file_path, index=False)

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
    """A class for scaling and saving data. This is the data that is used for
    model training.

    Attributes:
        symbol (str): The symbol of the data.
        interval (str): The interval of the data.

    Methods:
        __init__(symbol: str, interval: str): Initializes the DataScalerZero object.
        get_csv_file_path() -> str: Returns the file path of the CSV file.
        data_scaling() -> pd.DataFrame: Scales the data and returns a DataFrame.
        save_scaled_data() -> None: Saves the scaled data to a CSV file.
    """

    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval

        self.data_cleaned = pd.read_csv(self.get_csv_file_path())

        self.data_scaled = self.data_scaling()

        self.volume_scaler = MinMaxScaler()

        self.save_scaled_data()

    def get_csv_file_path(self) -> str:
        file_path = f"./data_cleaned/{self.interval}/{self.symbol} - {self.interval}.csv"

        return file_path

    def data_scaling(self) -> pd.DataFrame:
        df = self.data_cleaned.copy(deep=True)

        df["real_close"] = df["close"]

        df["volume_day_max"] = 0
        df["volume_day_min"] = 0

        # for 1st day
        # real close if the open of that day itself, as there is no previous day

        first_day_last_index: int = 375

        df.iloc[:first_day_last_index, df.columns.get_loc("real_close")] = df.iloc[
            0,
            df.columns.get_loc("open"),
        ]

        df.iloc[:first_day_last_index, df.columns.get_loc("volume_day_max")] = max(
            df.iloc[:first_day_last_index, df.columns.get_loc("volume")].values,
        )

        df.iloc[:first_day_last_index, df.columns.get_loc("volume_day_min")] = min(
            df.iloc[:first_day_last_index, df.columns.get_loc("volume")].values,
        )

        for day in range(1, len(df) // 375):
            start_index: int = day * 375
            end_index: int = start_index + 375

            prev_close: float = df.iloc[start_index - 1, df.columns.get_loc("close")]

            max_volume: int = max(
                df.iloc[start_index:end_index, df.columns.get_loc("volume")].values,
            )
            min_volume: int = min(
                df.iloc[start_index:end_index, df.columns.get_loc("volume")].values,
            )

            df.iloc[
                start_index:end_index,
                df.columns.get_loc("real_close"),
            ] = prev_close

            df.iloc[
                start_index:end_index,
                df.columns.get_loc("volume_day_max"),
            ] = max_volume

            df.iloc[
                start_index:end_index,
                df.columns.get_loc("volume_day_min"),
            ] = min_volume

        df["open"] = df["open"] / df["real_close"]
        df["high"] = df["high"] / df["real_close"]
        df["low"] = df["low"] / df["real_close"]
        df["close"] = df["close"] / df["real_close"]

        df["volume"] = df.apply(
            lambda row: min_max_scaler(
                row["volume"],
                row["volume_day_min"],
                row["volume_day_max"],
            ),
            axis=1,
        )

        return df[["open", "high", "low", "close", "real_close", "volume"]]

    def save_scaled_data(self) -> None:
        os.makedirs(f"./data_training/{self.interval}", exist_ok=True)

        self.data_scaled.to_csv(
            f"./data_training/{self.symbol} - {self.interval}.csv",
            index=True,
        )


def process_symbol(symbol: str, interval: str) -> None:
    """Process a single symbol with data cleaning and scaling."""
    try:
        print(f"\n{symbol} \t- Starting processing")

        # Data cleaning
        DataCleanerZero(symbol=symbol, interval=interval)
        print(f"{symbol} \t- Data Cleaning Done.")

        # Data scaling
        DataScalerZero(symbol=symbol, interval=interval)
        print(f"{symbol} \t- Data Scaling Done.")

    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")


def process_symbols_parallel(symbols: list[str], interval: str) -> None:
    """Process multiple symbols in parallel using all available CPU cores."""

    # Get the number of CPU cores
    num_cores = mp.cpu_count() // SAFETY_FACTOR_CPU_CORES_USAGE
    print(f"Using {num_cores} CPU cores for parallel processing out of total {mp.cpu_count()}")

    # Create a partial function with interval pre-filled
    process_func = partial(process_symbol, interval=interval)

    # Use multiprocessing Pool to process symbols in parallel
    with mp.Pool(processes=num_cores) as pool:
        pool.map(process_func, symbols)


if __name__ == "__main__":
    interval = "1m"

    print(f"Processing {len(nifty50_symbols)} symbols using parallel processing...")
    process_symbols_parallel(nifty50_symbols, interval)
    print("\nAll symbols processed successfully!")
