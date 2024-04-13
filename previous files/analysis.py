import pandas as pd
import csv
from datetime import datetime, timedelta

from time import time
import os
import pytz


nifty50_tickers = [
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GAIL.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFC.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "IOC.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "UPL.NS",
    "WIPRO.NS",
]

# todoo: - graph of profit from high and lows during zone. green dot if low is first, and red dot if high is first., without any safety factor. percent vs date.day


class AnalysisStocks:
    def __init__(self, ticker_1: str, interval: str) -> None:
        self.ticker_1 = ticker_1
        self.interval = interval

    def if_in_half(self, check_datetime, which_half: int) -> bool:
        # which_half = 0, means 1st half - input data
        # which_half = 1, means 2nd half - predict data
        datetime_1 = datetime.strptime("2000-01-01 10:00:00+0530", "%Y-%m-%d %H:%M:%S%z")
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
        datetime_1 = datetime.strptime("2000-01-01 10:00:00+0530", "%Y-%m-%d %H:%M:%S%z")
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

    def get_data_df(self, ticker: str, which_half: str) -> pd.DataFrame:
        file_path = f"./{self.interval} data/{ticker} - {self.interval}.csv"
        df = pd.read_csv(file_path)
        # new_1 = df[["Adj Close", "Datetime"]].copy()

        if which_half == "full_zone":
            df["to_add"] = df["Datetime"].apply(lambda x: self.if_in_zone(x))
        elif which_half == "first_half":
            df["to_add"] = df["Datetime"].apply(lambda x: self.if_in_first_half(x))
        elif which_half == "second_half":
            df["to_add"] = df["Datetime"].apply(lambda x: self.if_in_second_half(x))

        # df["adj_close"] = df["Adj Close"].apply(lambda x: self.round_decimals_2(x))
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

    def inner_join(self, df_1, df_2, key):
        res_def = pd.merge(df_1, df_2, on=key, how="inner", indicator=True)
        # not_matched_df1 = res_def[res_def["_merge"] == "left_only"]
        # not_matched_df2 = res_def[res_def["_merge"] == "right_only"]
        return res_def

    def find_high_lows(self, dataframe: pd.DataFrame):
        dataframe["Datetime"] = pd.to_datetime(dataframe["Datetime"], format="%Y-%m-%d %H:%M:%S%z")

        res = []

        unique_dates = dataframe["Datetime"].dt.date.unique()
        timezone_offset = timedelta(hours=5, minutes=30)
        for date in unique_dates:
            data_df = dataframe[dataframe["Datetime"].dt.date == date]

            high = data_df["high"].max()
            low = data_df["low"].min()

            high_datetime = str(dataframe.loc[dataframe["high"] == high, "Datetime"].values[0])
            high_datetime_2 = datetime.fromisoformat(high_datetime) + timezone_offset
            time_h = high_datetime_2.time()

            low_datetime = str(dataframe.loc[dataframe["low"] == low, "Datetime"].values[0])
            low_datetime_2 = datetime.fromisoformat(low_datetime) + timezone_offset
            time_l = low_datetime_2.time()

            # high occured after low.
            res_colour = "green"
            if time_l > time_h:
                res_colour = "red"

            res.append([str(date), high, low, res_colour])

        return res

    def full_zone_cumulative(self, interval_5m=False, safety_factor=100):
        # highs and lows, full potential, with or withour 5 minute interval between them.
        # safety factor applied on both sides at highs and lows.

        stock_1_df = self.get_data_df(self.ticker_1, which_half="full_zone")
        all_high_lows = self.find_high_lows(stock_1_df)
        num_days = len(all_high_lows)

        total_percent = 100
        for ls in all_high_lows:
            total_percent *= ls[1] / ls[2]
        daily_cgr = round((pow(total_percent / 100, 1 / num_days) - 1) * 100, 3)
        total_profit_percent = round(total_percent - 100, 3)

        # _2 is second half
        stock_1_df_2nd_half = self.get_data_df(self.ticker_1, which_half="second_half")
        all_high_lows_2nd_half = self.find_high_lows(stock_1_df_2nd_half)

        total_percent_2 = 100
        total_percent_sf = 100
        for ls in all_high_lows_2nd_half:
            total_percent_2 *= ls[1] / ls[2]
            # safety factor on difference of high and low side.
            total_percent_sf *= 1 + ((ls[1] - ls[2]) * (safety_factor / 100)) / ls[2]

        daily_cgr_2 = round((pow(total_percent_2 / 100, 1 / num_days) - 1) * 100, 3)
        total_profit_percent_2 = round(total_percent_2 - 100, 2)

        daily_cgr_sf = round((pow(total_percent_sf / 100, 1 / num_days) - 1) * 100, 3)
        total_profit_percent_sf = round(total_percent_sf - 100, 2)
        return {
            "total_percent_sf": total_profit_percent_sf,
            "daily_cgr_sf": daily_cgr_sf,
            "num_days": num_days,
            "total_percent_2nd_half": total_profit_percent_2,
            "daily_cgr_2nd_half": daily_cgr_2,
            "total_percent": total_profit_percent,
            "daily_cgr": daily_cgr,
            "sf_safety_factor": safety_factor,
        }

    # stock name, total percent, daily, number of days, red/green

    def calculate_correlation(self, ticker_2: str) -> float:
        stock_1_df = self.get_data_df(self.ticker_1, which_half="full_zone")
        stock_1_df.rename(columns={"adj_close": self.ticker_1}, inplace=True)

        stock_2_df = self.get_data_df(ticker_2, which_half="full_zone")
        stock_2_df.rename(columns={"adj_close": self.ticker_2}, inplace=True)

        new_df = self.inner_join(stock_1_df, stock_2_df, "Datetime")

        stock_1_prices = new_df[self.ticker_1].tolist()
        stock_2_prices = new_df[ticker_2].tolist()

        df = pd.DataFrame({"Stock1": stock_1_prices, "Stock2": stock_2_prices})
        correlation = df["Stock1"].corr(df["Stock2"])

        return correlation


def main_func(interval, n, work):
    INTERVAL = interval
    WORK = work

    rows = []

    if WORK == "coorelation":
        for i in range(n - 1):
            for j in range(i + 1, n):
                ticker_1 = nifty50_tickers[i]
                ticker_2 = nifty50_tickers[j]

                obj = AnalysisStocks(ticker_1=ticker_1, interval=INTERVAL)
                z = round(obj.calculate_correlation(ticker_2), 6) * 100
                print(f"\nCorrelation: {z}\n")

                rows.append({"stock_1": ticker_1, "stock_2": ticker_2, "correlation_in_zone": z})

        filename = f"./work - {WORK}/data - {INTERVAL} - {datetime.today().date()}.csv"

        fieldnames = [
            "stock_1",
            "stock_2",
            "correlation_in_zone",
            f"file_date= {datetime.today().date()}",
        ]

    elif WORK == "cumulative":
        for i in range(n):
            ticker_1 = nifty50_tickers[i]

            obj = AnalysisStocks(ticker_1=ticker_1, interval=INTERVAL)
            z = obj.full_zone_cumulative(safety_factor=80)
            print(f"\nfull_zone_cumulative: {ticker_1}, {z}\n")

            rows.append(
                {
                    "stock_name": ticker_1,
                    "total_percent_sf": z["total_percent_sf"],
                    "daily_cgr_sf": z["daily_cgr_sf"],
                    "total_percent_2nd_half": z["total_percent_2nd_half"],
                    "daily_cgr_2nd_half": z["daily_cgr_2nd_half"],
                    "number_of_days": z["num_days"],
                    "total_percent_full_zone": z["total_percent"],
                    "daily_cgr_full_zone": z["daily_cgr"],
                    "sf_safety_factor": z["sf_safety_factor"],
                }
            )

        folder_name = f"./work - {WORK}"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        filename = os.path.join(folder_name, f"{INTERVAL}.csv")

        fieldnames = [
            "stock_name",
            "total_percent_sf",
            "daily_cgr_sf",
            "number_of_days",
            "total_percent_2nd_half",
            "daily_cgr_2nd_half",
            f"file_date= {datetime.today().date()}",
            "total_percent_full_zone",
            "daily_cgr_full_zone",
            "sf_safety_factor",
        ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    work_options = {
        1: "coorelation",
        2: "cumulative",
        3: "work_3",
    }
    interval = "1m"
    num_stocks = 50
    work = 2
    main_func(interval=interval, n=num_stocks, work=work_options[work])


if __name__ == "__main__":
    time_1 = time()
    main()
    print(f"time taken = {round(time() - time_1, 2)} sec")
