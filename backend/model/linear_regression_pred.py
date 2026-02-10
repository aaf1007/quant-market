
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

stock_symbol = "AAPL"
stock_data = yf.Ticker(stock_symbol)
info = stock_data.info

days = 1 # Predict num days ahead

# Fetch price data for training the model
price = stock_data.history(period="1y") # gets 1 year of data
price["Target"] = price["Close"].shift(-days) # target is the next day's close price

# Custom Features
price["Moving_Average_60"] = price["Close"].rolling(window=60).mean() # 10 day moving average
price["Rolling_STD_60"] = price["Close"].rolling(window=60).std() # 10 day rolling standard deviation

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

print(f"Features Used: {X_train.columns.tolist()}")

print("Mean Squared Error: ", round(mse, 2))
print("Root Mean Squared Error: ", round(rmse, 2))


