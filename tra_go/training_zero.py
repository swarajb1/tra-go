"""

This module contains constants and functions for training the model.
It handles data loading, splitting, and preparation for machine learning.
"""

# Standard library imports
import pandas as pd  # type: ignore
from numpy.typing import NDArray

# Local imports
from utils.others import get_initial_index_offset

from database.enums import BandType, IODataType, RequiredDataType, TickerOne

# Constants
TOTAL_POINTS_IN_ONE_DAY: int = 375
NUMBER_OF_POINTS_IN_ZONE_1: int = 150  # Number of points in zone 1
NUMBER_OF_POINTS_IN_ZONE_2: int = 150  # Number of points in zone 2
NUMBER_OF_POINTS_IN_ZONE_DAY: int = NUMBER_OF_POINTS_IN_ZONE_1 + NUMBER_OF_POINTS_IN_ZONE_2


def get_csv_file_path(ticker: str, interval: str) -> str:
    """Construct the file path for the CSV file based on ticker and interval."""

    return f"./data_training/{ticker} - {interval}.csv"


def get_data_all_df(ticker: TickerOne, interval: str) -> pd.DataFrame:
    """Load data from a CSV file for the given ticker and interval."""

    df = pd.read_csv(get_csv_file_path(ticker.value, interval))
    return df


def data_split_train_test(
    df: pd.DataFrame,
    test_size: float,
    data_required: RequiredDataType,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets based on the specified test size."""

    number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY
    train_days: int = int(number_of_days * (1 - test_size))
    first_test_index: int = train_days * NUMBER_OF_POINTS_IN_ZONE_DAY

    train_df = df.iloc[:first_test_index]
    test_df = df.iloc[first_test_index:]

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    columns: list[str]
    if data_required == RequiredDataType.TRAINING:
        columns = ["open", "high", "low", "close", "volume", "real_close"]
    elif data_required == RequiredDataType.REAL:
        columns = ["open", "high", "low", "close", "volume"]

    return train_df[columns], test_df[columns]


def data_split_x_y_close(
    df: pd.DataFrame,
    interval: str,
    x_type: BandType,
    y_type: BandType,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into input (X), output (Y), and previous close price dataframes."""

    df_i = pd.DataFrame()
    df_o = pd.DataFrame()
    prev_close = pd.DataFrame()

    number_of_days: int = len(df) // NUMBER_OF_POINTS_IN_ZONE_DAY

    for day in range(number_of_days):
        day_start_index: int = day * NUMBER_OF_POINTS_IN_ZONE_DAY
        first_2_nd_zone_index: int = day_start_index + NUMBER_OF_POINTS_IN_ZONE_1
        day_end_index: int = day_start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1

        df_i = pd.concat([df_i, df.iloc[day_start_index:first_2_nd_zone_index]])
        df_o = pd.concat([df_o, df.iloc[first_2_nd_zone_index : day_end_index + 1]])

        dict_1 = {"real_close": df.iloc[day_start_index, df.columns.get_loc("real_close")]}
        prev_close = pd.concat([prev_close, pd.DataFrame(dict_1, index=[0])], ignore_index=True)

    df_i.reset_index(drop=True, inplace=True)
    df_o.reset_index(drop=True, inplace=True)

    # Determine columns based on band type
    columns_x: list[str] = _get_columns_for_band_type(x_type)
    columns_y: list[str] = _get_columns_for_band_type(y_type)

    return df_i[columns_x], df_o[columns_y], prev_close[["real_close"]]


def _get_columns_for_band_type(band_type: BandType) -> list[str]:
    """Helper function to get the columns for a specific band type."""

    if band_type == BandType.BAND_4:
        return ["open", "high", "low", "close"]
    elif band_type == BandType.BAND_2 or band_type == BandType.BAND_2_1:
        return ["low", "high"]
    elif band_type == BandType.BAND_5:
        return ["open", "high", "low", "close", "volume"]
    else:
        raise ValueError(f"Unsupported band type: {band_type}")


def data_inside_zone(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Extract data within the specified zone from the dataframe."""

    res_df = pd.DataFrame()
    initial_index_offset: int = get_initial_index_offset(NUMBER_OF_POINTS_IN_ZONE_2)
    number_of_days = len(df) // TOTAL_POINTS_IN_ONE_DAY

    for day in range(number_of_days):
        start_index: int = day * TOTAL_POINTS_IN_ONE_DAY + initial_index_offset
        end_index: int = start_index + NUMBER_OF_POINTS_IN_ZONE_DAY - 1
        res_df = pd.concat([res_df, df.iloc[start_index : end_index + 1]])

    res_df.reset_index(drop=True, inplace=True)
    return res_df[["open", "high", "low", "close", "real_close", "volume"]]


def by_date_df_array(df: pd.DataFrame, band_type: BandType, io_type: IODataType) -> NDArray:
    """Convert the dataframe into a 3D array based on the specified band type and I/O type."""

    array = df.values

    # Determine points per day based on IO type
    if io_type == IODataType.INPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_1
    elif io_type == IODataType.OUTPUT_DATA:
        points_in_each_day = NUMBER_OF_POINTS_IN_ZONE_2
    else:
        raise ValueError(f"Unsupported IO type: {io_type}")

    # Map band types to their shapes
    band_shapes: dict[BandType, int] = {
        BandType.BAND_4: 4,
        BandType.BAND_2: 2,
        BandType.BAND_2_1: 2,
        BandType.BAND_5: 5,
    }

    if band_type not in band_shapes:
        raise ValueError(f"Unsupported band type: {band_type}")

    # Reshape the array
    res = array.reshape(len(array) // points_in_each_day, points_in_each_day, band_shapes[band_type])
    return res


def train_test_split(
    data_df: pd.DataFrame,
    interval: str,
    x_type: BandType,
    y_type: BandType,
    test_size: float = 0.2,
) -> tuple[tuple[NDArray, NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
    """Split the data into training and testing sets for model training."""

    # Create a deep copy to avoid modifying the original dataframe
    df = data_df.copy(deep=True)

    # Process the data
    df = data_inside_zone(df=df, interval=interval)
    df_train, df_test = data_split_train_test(df=df, test_size=test_size, data_required=RequiredDataType.TRAINING)

    # Split into X, Y, and close price for both training and testing sets
    df_train_x, df_train_y, df_train_close = data_split_x_y_close(
        df=df_train,
        interval=interval,
        x_type=x_type,
        y_type=y_type,
    )

    df_test_x, df_test_y, df_test_close = data_split_x_y_close(
        df=df_test,
        interval=interval,
        x_type=x_type,
        y_type=y_type,
    )

    # Convert to arrays
    train_x = by_date_df_array(df_train_x, band_type=x_type, io_type=IODataType.INPUT_DATA)
    test_x = by_date_df_array(df_test_x, band_type=x_type, io_type=IODataType.INPUT_DATA)

    train_y = by_date_df_array(df_train_y, band_type=y_type, io_type=IODataType.OUTPUT_DATA)
    test_y = by_date_df_array(df_test_y, band_type=y_type, io_type=IODataType.OUTPUT_DATA)

    train_prev_close = df_train_close.values
    test_prev_close = df_test_close.values

    return (
        (train_x, train_y, train_prev_close),
        (test_x, test_y, test_prev_close),
    )
