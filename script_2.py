import pandas as pd

nifty50_symbols: list[str] = [
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "GAIL",
    "GRASIM",
    "HCLTECH",
    "LTIM",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "IOC",
    "ITC",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SUNPHARMA",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "ULTRACEMCO",
    "UPL",
    "WIPRO",
    "NIFTY 50",
    "NIFTY BANK",
]


df = pd.read_csv("instrument_df.csv")

# Assuming 'df' is your DataFrame and 'column_name' is the name of the column you want to check
filtered_df = df[df["tradingsymbol"].isin(nifty50_symbols)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("selected_stocks.csv", index=False)
