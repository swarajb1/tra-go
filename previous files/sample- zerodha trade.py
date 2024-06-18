import configparser
import webbrowser

from kiteconnect import KiteConnect

# Read API credentials from the configuration file
config = configparser.ConfigParser()
config.read("config.ini")

api_key = config.get("API", "API_KEY")
api_secret = config.get("API", "API_SECRET")
redirect_url = config.get("API", "REDIRECT_URL")

# Initialize KiteConnect instance
kite = KiteConnect(api_key=api_key)

# Generate the login URL for authorization
login_url = kite.login_url()

# Open the login URL in a web browser to authorize the application
webbrowser.open(login_url)

# Get the request token from the callback URL
request_token = input("Enter the request token obtained from the callback URL: ")

# Generate the access token using the request token
data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]

# Set the access token
kite.set_access_token(access_token)


# Place an order
def place_order():
    try:
        order = {
            "exchange": "NSE",
            "tradingsymbol": "INFY",
            "transaction_type": "BUY",
            "quantity": 1,
            "order_type": "MARKET",
            "product": "CNC",
        }
        order_id = kite.place_order(order)

        print("Order placed successfully. Order ID:", order_id)
    except Exception as e:
        print("Error placing order:", str(e))


# Get pending orders
def get_pending_orders():
    try:
        orders = kite.orders()
        pending_orders = [order for order in orders if order["status"] == "OPEN"]

        print("Pending orders:")
        for order in pending_orders:
            print(order)
    except Exception as e:
        print("Error getting pending orders:", str(e))


# Get executed orders
def get_executed_orders():
    try:
        orders = kite.orders()
        executed_orders = [order for order in orders if order["status"] == "COMPLETE"]

        print("Executed orders:")
        for order in executed_orders:
            print(order)
    except Exception as e:
        print("Error getting executed orders:", str(e))


# Place an order
place_order()

# Get pending orders
get_pending_orders()

# Get executed orders
get_executed_orders()
