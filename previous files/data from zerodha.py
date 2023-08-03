from kiteconnect import KiteConnect
import datetime


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set your API key and access token
api_key = os.getenv("API_KEY")
access_token = os.getenv("ACCESS_TOKEN")

# Initialize KiteConnect client
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

instrument_token = "NSE:HDFCBANK"

num_days = 30

# Calculate start and end date for historical data
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=num_days)

# Fetch historical data
data = kite.historical_data(instrument_token, start_date, end_date, interval="minute")

# Print the fetched data
for candle in data:
    print(candle)
