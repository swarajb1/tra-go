from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, timedelta

import os
from dotenv import load_dotenv

load_dotenv()


zerodha_id = os.environ.get("ZERODHA_ID")
password = os.environ.get("PASSWORD")


api_key = os.environ.get("API_KEY")
access_token = os.environ.get("ACCESS_TOKEN")


# # API credentials
# api_key = "your_api_key"
# api_secret = "your_api_secret"
# access_token = "your_access_token"

# # Initialize KiteConnect client
# kite = KiteConnect(api_key=api_key)
# kite.set_access_token(access_token)

# # Define the instrument token for HDFC Bank India
# instrument_token = "HDFCBANK"

# # Calculate start and end timestamps for data retrieval
# end_date = datetime.now()
# start_date = end_date - timedelta(days=60)

# # Convert timestamps to required format (milliseconds)
# start_timestamp = int(start_date.timestamp() * 1000)
# end_timestamp = int(end_date.timestamp() * 1000)

# # Fetch historical data
# data = kite.historical_data(
#     instrument_token, start_timestamp, end_timestamp, interval="minute"
# )

# # Convert the data to a pandas DataFrame
# df = pd.DataFrame(data)

# # Print the downloaded data
# print(df)
# from kiteconnect import KiteConnect
# import pandas as pd
# from datetime import datetime, timedelta

# # API credentials
# api_key = "your_api_key"
# api_secret = "your_api_secret"
# access_token = "your_access_token"

# # Initialize KiteConnect client
# kite = KiteConnect(api_key=api_key)
# kite.set_access_token(access_token)

# # Define the instrument token for HDFC Bank India
# instrument_token = "HDFCBANK"

# # Calculate start and end timestamps for data retrieval
# end_date = datetime.now()
# start_date = end_date - timedelta(days=60)

# # Convert timestamps to required format (milliseconds)
# start_timestamp = int(start_date.timestamp() * 1000)
# end_timestamp = int(end_date.timestamp() * 1000)

# # Fetch historical data
# data = kite.historical_data(
#     instrument_token, start_timestamp, end_timestamp, interval="minute"
# )

# # Convert the data to a pandas DataFrame
# df = pd.DataFrame(data)

# # Print the downloaded data
# print(df)
