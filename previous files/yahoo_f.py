import yfinance as yf 
from copy import deepcopy


data_req = [
    "currency", 
    "marketCap", 
    "fiftyDayAverage", 
    "twoHundredDayAverage", 
    "enterpriseValue", 
    "shortName",
    "longName",
    "currentPrice",
    "symbol",
    # "",
    # "",
    ]


def get_stock_info(stock_symbol):
    stock_info = {}
    stock_data = yf.Ticker(stock_symbol)
    stock_info['stock_price'] = stock_data.info["currentPrice"]
    stock_info['splits'] = deepcopy(stock_data.splits.to_dict())
    return stock_info




# stock_data = deepcopy(get_stock_info('IRCTC.NS'))

# print(stock_data['stock_price'])
# print(stock_data['splits'])


ticker = "HDFCBANK.NS"

# Get the current price of HDFC Bank stock
stock_data = yf.Ticker(ticker).history(period='5m', interval='1m')

print(type(stock_data))
print((stock_data))