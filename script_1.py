import pandas as pd
from kiteconnect import KiteConnect

from script_2 import nifty50_symbols

if __name__ == "__main__":
    key_secret = open("api_key.txt").read().split()

    kite = KiteConnect(api_key=key_secret[0])
    print(kite.login_url())

    request_token = str(input("Enter Your Request Token Here :"))
    data = kite.generate_session(request_token, api_secret=key_secret[1])

    kite.set_access_token(data["access_token"])

    print("Kite Session Generated")

    for exchange in ["NSE", "BSE"]:
        instrument_dump = kite.instruments(exchange=exchange)
        instrument_df = pd.DataFrame(instrument_dump)

        instrument_df.to_csv(f"data_z/instrument_df_{exchange}.csv", index=False)

        print(instrument_df.head())

        filtered_df = instrument_df[instrument_df["tradingsymbol"].isin(nifty50_symbols)]

        filtered_df.to_csv(f"data_z/selected_stocks_{exchange}.csv", index=False)
