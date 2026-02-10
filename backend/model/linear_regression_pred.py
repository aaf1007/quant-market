
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

def predict_price(ticker_symbol, days_ahead=1) -> float:
    stock_symbol = ticker_symbol
    stock_data = yf.Ticker(stock_symbol)

    # Get stock info
    info = stock_data.info

    # Fetch price data for training the model
    price = stock_data.history(period="1y") # gets 1 year of data
    price["Target"] = price["Close"].shift(-days_ahead) # target is the next day's close price

    # Custom Features
    price["Moving_Average_60"] = price["Close"].rolling(window=60).mean() # 60 day moving average
    price["Rolling_STD_60"] = price["Close"].rolling(window=60).std() # 60 day rolling standard deviation

    price = price.dropna() # remove rows with NaN values (must be after all feature creation)

    # Linear Regression Equation: y = mx
    X = price[["Close", "Volume", "Moving_Average_60", "Rolling_STD_60"]] # features are the close prices
    y = price["Target"]

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Get returned value from model from current days data
    latest_price_df = price[["Close", "Volume", "Moving_Average_60", "Rolling_STD_60"]].iloc[[-1]]

    ret_prediction = model.predict(latest_price_df)

    # print(f"Features Used: {X_train.columns.tolist()}")

    # print("Mean Squared Error: ", round(mse, 2))
    # print("Root Mean Squared Error: ", round(rmse, 2))

    return round(ret_prediction[0], 2)



