import pandas as pd

# Sample data: daily commodity prices
data = {"price": [100, 102, 101, 105, 107, 110, 108, 112, 115, 117]}
df = pd.DataFrame(data)

# Compute 7-day moving average
df["7_day_moving_avg"] = df["price"].rolling(window=7).mean()

# Compute daily price change
df["daily_change"] = df["price"].diff()

# Create lag features
df["price_1_day_ago"] = df["price"].shift(1)
df["price_2_days_ago"] = df["price"].shift(2)

# Compute rolling standard deviation over 7 days
df["7_day_std_dev"] = df["price"].rolling(window=7).std()

print("DataFrame with NaN values:")
print(df)

# Replace NaN values with the nearest non-NaN value (forward fill)
df.fillna(method="ffill", inplace=True)

# If there are still NaN values at the beginning, use backward fill
df.fillna(method="bfill", inplace=True)

print("\nDataFrame after replacing NaN values with the nearest non-NaN value:")
print(df)
