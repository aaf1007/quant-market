from model import linear_regression_pred
import yfinance as yf

POPULAR_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]

def get_popular_stocks():
    """Display popular stocks with their names."""
    for i, ticker in enumerate(POPULAR_STOCKS):
        print(f"{i+1}.) {ticker} - {yf.Ticker(ticker).info.get('longName')}")
    
def main():
    print("Stock Price Predictor")
    print("=====================")

    while True:
        print("1. Predict Stock Price")
        print("2. Get Popular Stocks")
        print("3. Exit")
        print("=====================")
        choice = input("Enter your choice: ")
        match choice:
            case "1":
            # Ask for stock symbol
                stock_symbol = input("Enter stock symbol or name (e.g. AAPL or Apple): ")

                while (yf.Ticker(stock_symbol).info.get("regularMarketPrice") is None):
                    # Stock symbol / Ticker is invalid
                    stock_symbol = input("Enter stock symbol or name (e.g. AAPL or Apple): ")

                prediction = linear_regression_pred.predict_price(stock_symbol)

                print(f"\nPrediction for {stock_symbol}: {prediction}")

            case "2":
                get_popular_stocks()
            case "3":
                print('Thank you!')
                exit()
            case _:
                print("Invalid choice")
                main()

        print("\n")

main()