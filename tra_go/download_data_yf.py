# yfinance.
# NOTE:
# Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

import datetime
import os
import time

import pandas as pd
import yfinance as yf

other_config = {
    "1m": {"number of days": 30, "batch days": 7},
    "2m": {"number of days": 60, "batch days": 60},
    "5m": {"number of days": 60, "batch days": 60},
    "15m": {"number of days": 60, "batch days": 60},
    "30m": {"number of days": 60, "batch days": 60},
    "1h": {"number of days": 730, "batch days": 730},
    "1d": {"number of days": 9000, "batch days": 9000},
}

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
    "LTIM.NS",
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


for INTERVAL in [
    "1m",
    "2m",
    "5m",
    "1h",
    "1d",
]:
    NUMBER_OF_DAYS = other_config[INTERVAL]["number of days"]
    BATCH_DAYS = other_config[INTERVAL]["batch days"]
    print(BATCH_DAYS)
    print(NUMBER_OF_DAYS)

    for ticker in nifty50_tickers:
        print(ticker)
        last_date = datetime.date.today()
        first_date = last_date - datetime.timedelta(days=NUMBER_OF_DAYS - 1)

        print("last_date = ", last_date)
        print("first_date = ", first_date)

        end_date = last_date
        start_date = end_date - datetime.timedelta(days=BATCH_DAYS - 1)

        all_data = pd.DataFrame()

        count = 0
        print(count, "-----------\n")

        for _ in range(NUMBER_OF_DAYS // (BATCH_DAYS)):
            count += 1
            print(end_date)
            print(start_date)
            print(count, "-----------\n")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=INTERVAL,
                progress=False,
            )

            all_data = pd.concat([all_data, data], axis=0)

            end_date -= datetime.timedelta(days=BATCH_DAYS)
            start_date = end_date - datetime.timedelta(days=BATCH_DAYS - 1)
            time.sleep(3)

        if NUMBER_OF_DAYS % (BATCH_DAYS):
            count += 1
            print(count, "last -----------\n")
            start_date = first_date
            print(end_date)
            print(start_date)

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=INTERVAL,
                progress=False,
            )
            all_data = pd.concat([all_data, data], axis=0)
            time.sleep(3)

        all_data["Open"] = all_data["Open"].apply(lambda x: round(number=x, ndigits=2))
        all_data["Close"] = all_data["Close"].apply(
            lambda x: round(number=x, ndigits=2),
        )
        all_data["High"] = all_data["High"].apply(lambda x: round(number=x, ndigits=2))
        all_data["Low"] = all_data["Low"].apply(lambda x: round(number=x, ndigits=2))
        all_data["Adj Close"] = all_data["Adj Close"].apply(
            lambda x: round(number=x, ndigits=2),
        )

        final_data = all_data.sort_index(ascending=True).copy()
        index_column_name: str = ""
        if INTERVAL == "1d":
            index_column_name = "Date"
        else:
            index_column_name = "Datetime"

        final_data.rename_axis(index_column_name, inplace=True)

        print(final_data)

        folder_name = f"./data_stock_price_yf/{INTERVAL} data"
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        filename = os.path.join(folder_name, f"{ticker} - {INTERVAL}.csv")
        if os.path.exists(filename):
            previous_data = pd.read_csv(filename, index_col=0)

            final_data = pd.concat([previous_data, final_data]).drop_duplicates(
                keep="first",
            )

        final_data.to_csv(filename, index=True)

        # repeating process to drop duplicates properly
        # for some reason have to do this. to remove duplicates.
        d1 = pd.read_csv(filename, index_col=0)
        d2 = d1.drop_duplicates(keep="first")
        d2 = d2.sort_values(by=index_column_name, ascending=True)
        d2.to_csv(filename, index=True)

        print(filename)
