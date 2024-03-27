import os
from datetime import date, timedelta
from time import sleep

import pandas as pd
from kiteconnect import KiteConnect


def create_symbol_instrument_token_dict(exchange: str) -> dict[str, str]:
    df = pd.read_csv(f"data_z/selected_stocks_{exchange}.csv")

    symbol_instrument_token_dict = {}

    for index, row in df.iterrows():
        symbol_instrument_token_dict[row["tradingsymbol"]] = row["instrument_token"]

    return symbol_instrument_token_dict


if __name__ == "__main__":
    key_secret = open("api_key.txt").read().split()
    kite = KiteConnect(api_key=key_secret[0])
    print(kite.login_url())
    # create kite object

    request_token = str(input("Enter Your Request Token Here :"))
    data = kite.generate_session(request_token, api_secret=key_secret[1])

    kite.set_access_token(data["access_token"])

    print("Kite Session Generated")

    hard_start_date = date(2000, 1, 1)
    hard_stop_date = date.today()

    for exchange in ["NSE", "BSE"]:
        dict_symbols = create_symbol_instrument_token_dict(exchange)

        for symbol, instrument_token in dict_symbols.items():
            print("\n" * 2, symbol, "\t=", instrument_token)

            file_path: str = f"data_z/{exchange.lower()}/1m/{symbol} - 1m.csv"

            if os.path.exists(file_path):
                continue

            all_data_df = pd.DataFrame()

            stop_date = hard_start_date + timedelta(days=60)

            while stop_date <= hard_stop_date:
                print(stop_date)
                start_date = stop_date - timedelta(days=60)

                data_dump = kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=start_date,
                    to_date=stop_date,
                    interval="minute",
                )

                data_df = pd.DataFrame(data_dump)

                print(symbol)
                print(data_df.head())

                all_data_df = pd.concat([all_data_df, data_df]).drop_duplicates(keep="first")

                if stop_date == hard_stop_date:
                    break

                stop_date = stop_date + timedelta(days=61)

                if stop_date > hard_stop_date:
                    stop_date = hard_stop_date

                sleep(0.1)

            all_data_df.to_csv(file_path, index=False)

            print(symbol, "\tfull_data")
            print(all_data_df.head())
