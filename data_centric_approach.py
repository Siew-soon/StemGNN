import pandas as pd
from datetime import datetime


def datetime_to_minutes(dt):
    return dt.hour * 60 + dt.minute


def calculate_size(data):
    sizes = {
        "15": [2],
        "30": [2],
        "60": [4],
    }

    # Define the date format
    date_format = "%Y.%m.%d %H:%M"

    # Convert the string dates to datetime objects
    date1 = datetime.strptime(data["Date"][0], date_format)
    date2 = datetime.strptime(data["Date"][1], date_format)

    # Calculate the difference
    minutes1 = datetime_to_minutes(date1)
    minutes2 = datetime_to_minutes(date2)
    timeframe = str(minutes1 - minutes2)

    for size in sizes:
        if timeframe == size:
            return sizes[size][0]

    return None


# Create more features based on existing data
def calculate_rolling_statistics(data):

    ohlc_columns = ["Open", "High", "Low", "Close"]
    indicator_columns = [
        "MACD Signal",
        "MACD Histogram",
        "EMA10",
        "EMA20",
        "EMA200",
        "Stochastic Main",
        "Stochastic Signal",
        "MA10",
        "MA20",
        "ATR",
        "Volume",
    ]

    window_size = calculate_size(data)

    # Check if columns exist before performing operations
    for column in ohlc_columns + indicator_columns:
        if column in data:
            # If column exists, calculate rolling mean and standard deviation
            data[column + "_RollingMean"] = (
                data[column].rolling(window=window_size).mean()
            )
            data[column + "_RollingStd"] = (
                data[column].rolling(window=window_size).std()
            )

    return data


def create_lagged_features(data):

    # Positive value which mena shift backward (To previous time) as the data is descending order
    lagged_size = calculate_size(data)

    # Shifting
    for column in data:
        if str(column) != "Date":
            data[column + "_lag" + str(lagged_size)] = data[column].shift(lagged_size)

    # Remove NaN row
    data.dropna(inplace=True)

    return data


def data_formating(data):

    # Convert "Date" column to datetime
    data["Date"] = pd.to_datetime(data["Date"], format="%Y.%m.%d %H:%M")

    # Extract the year from top_date and bottom_date
    top_year = datetime.strptime(str(data.iloc[0]["Date"]), "%Y-%m-%d %H:%M:%S").year
    bottom_year = datetime.strptime(
        str(data.iloc[-1]["Date"]), "%Y-%m-%d %H:%M:%S"
    ).year

    # Boolean indexing to remove rows within top_date and bottom_date
    train_data = data[
        (data["Date"].dt.year != top_year) & (data["Date"].dt.year != bottom_year)
    ]

    # Drop unneeded columns
    train_data = train_data.drop(columns=["Date"])

    return train_data


def extract_forward_data(data):

    # Convert "Date" column to datetime
    data["Date"] = pd.to_datetime(data["Date"], format="%Y.%m.%d %H:%M")

    # Extract the year from top_date and bottom_date
    top_year = datetime.strptime(str(data.iloc[0]["Date"]), "%Y-%m-%d %H:%M:%S").year

    # Boolean indexing to store rows within top_date
    forward_data = data[data["Date"].dt.year != top_year]

    # Drop unneeded columns
    forward_data = forward_data.drop(columns=["Date"])

    return forward_data


def ensure_numeric(data):
    # Attempt to convert all columns to numeric, ignoring errors
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data
