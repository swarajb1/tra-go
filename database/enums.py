from enum import Enum


class BandType(Enum):
    BAND_2 = "band_2"
    BAND_4 = "band_4"
    BAND_5 = "band_5"


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


class XDataType(Enum):
    PART_DATA = "part_data"
    FULL_DATA = "full_data"


class IODataType(Enum):
    INPUT_DATA = "input_data"
    OUTPUT_DATA = "ouput_data"
