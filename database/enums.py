from enum import Enum
from typing import Final


class BandType(Enum):
    BAND_1_CLOSE = "band_1_close"
    BAND_1_1 = "band_1_1"
    BAND_2 = "band_2"
    BAND_2_1 = "band_2_1"
    BAND_4 = "band_4"
    BAND_5 = "band_5"


ModelInputDataTypes: Final[list[BandType]] = [
    BandType.BAND_1_CLOSE,
    BandType.BAND_2,
    BandType.BAND_4,
    BandType.BAND_5,
]


ModelOutputDataTypes: Final[list[BandType]] = [
    BandType.BAND_1_1,
    BandType.BAND_2_1,
    BandType.BAND_2,
]


class IntervalType(Enum):
    MIN_1 = "1m"
    MIN_2 = "2m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class TickerOne(Enum):
    ADANIPORTS = "ADANIPORTS"
    APOLLOHOSP = "APOLLOHOSP"
    ASIANPAINT = "ASIANPAINT"
    AXISBANK = "AXISBANK"
    BAJAJ_AUTO = "BAJAJ-AUTO"
    BAJFINANCE = "BAJFINANCE"
    BAJAJFINSV = "BAJAJFINSV"
    BHARTIARTL = "BHARTIARTL"
    BPCL = "BPCL"
    BRITANNIA = "BRITANNIA"
    CIPLA = "CIPLA"
    COALINDIA = "COALINDIA"
    DIVISLAB = "DIVISLAB"
    DRREDDY = "DRREDDY"
    EICHERMOT = "EICHERMOT"
    GAIL = "GAIL"
    GRASIM = "GRASIM"
    HCLTECH = "HCLTECH"
    LTIM = "LTIM"
    HDFCBANK = "HDFCBANK"
    HDFCLIFE = "HDFCLIFE"
    HEROMOTOCO = "HEROMOTOCO"
    HINDALCO = "HINDALCO"
    HINDUNILVR = "HINDUNILVR"
    ICICIBANK = "ICICIBANK"
    INDUSINDBK = "INDUSINDBK"
    INFY = "INFY"
    IOC = "IOC"
    ITC = "ITC"
    JSWSTEEL = "JSWSTEEL"
    KOTAKBANK = "KOTAKBANK"
    LT = "LT"
    M_M = "M&M"
    MARUTI = "MARUTI"
    NESTLEIND = "NESTLEIND"
    NTPC = "NTPC"
    ONGC = "ONGC"
    POWERGRID = "POWERGRID"
    RELIANCE = "RELIANCE"
    SBILIFE = "SBILIFE"
    SBIN = "SBIN"
    SUNPHARMA = "SUNPHARMA"
    TATAMOTORS = "TATAMOTORS"
    TATASTEEL = "TATASTEEL"
    TCS = "TCS"
    TECHM = "TECHM"
    TITAN = "TITAN"
    ULTRACEMCO = "ULTRACEMCO"
    UPL = "UPL"
    WIPRO = "WIPRO"
    NIFTY_50 = "NIFTY 50"
    NIFTY_BANK = "NIFTY BANK"


class ModelLocationType(Enum):
    TRAINED_NEW = "training/models"
    SAVED = "training/models_saved"
    SAVED_DOUBLE = "training/models_saved_double"
    SAVED_TRIPLE = "training/models_saved_triple"

    OLD = "training/models_z_old"
    DISCARDED = "training/models_zz_discarded"


class IODataType(Enum):
    INPUT_DATA = "input_data"
    OUTPUT_DATA = "output_data"


class RequiredDataType(Enum):
    TRAINING = "training"
    REAL = "real_and_cleaned"


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"


class ProcessedDataType(Enum):
    REAL = "real"
    EXPECTED_REWARD = "expected_reward"
    REAL_FULL_REWARD = "real_full_reward"


class TickerDataType(Enum):
    TRAINING = "training"
    REAL_AND_CLEANED = "cleaned"


class ApproachType(Enum):
    MIN_MAX_TREND = "min_max_trend"
    CLOSING_ONE_POINT = "closing_one_point"


class InputNumberOfDataPoints(Enum):
    I_132 = 132
    I_150 = 150


class OutputNumberOfDataPoints(Enum):
    O_132 = 132
    O_150 = 150
