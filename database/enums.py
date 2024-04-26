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
