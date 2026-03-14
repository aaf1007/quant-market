import asyncio
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yfinance as yf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from model.train import linear_regression_factory as factory
from schemas import (
    HistoryPoint,
    PopularResponse,
    PopularStock,
    PredictRequest,
    PredictResponse,
    StatusResponse,
)

router = APIRouter()

POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_MAX_AGE_DAYS = 7


def _model_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker}.joblib"


def _model_is_fresh(ticker: str) -> bool:
    """Return True if the .joblib exists and is less than MODEL_MAX_AGE_DAYS old."""
    path = _model_path(ticker)
    if not path.exists():
        return False
    age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age_seconds < MODEL_MAX_AGE_DAYS * 86400


def _load_and_predict(ticker: str) -> PredictResponse:
    """Load cached model, run prediction, fetch 90-day history."""
    job = joblib.load(_model_path(ticker))
    model = job["pipeline"]
    features = job["features"]
    mse = job["mse_metric"]
    rmse = job["rmse_metric"]

    stock = yf.Ticker(ticker).history(period="365d")

    # Build history list (last 90 trading days)
    hist_df = stock.tail(90).copy()
    hist_df.index = hist_df.index.tz_localize(None)  # strip timezone for JSON
    history = [
        HistoryPoint(date=str(idx.date()), close=round(float(row["Close"]), 2))
        for idx, row in hist_df.iterrows()
    ]

    # Feature engineering (mirrors main.py / factory)
    stock["Moving_Average"] = stock["Close"].rolling(window=20).mean()
    stock["Volatility"] = stock["Close"].rolling(window=20).std()
    stock = stock.dropna()

    X = stock[features]
    pred = model.predict(X)

    predicted_price = float(pred[-1])
    prev_price = float(pred[-2])
    change = predicted_price - prev_price
    return_pct = (change / prev_price) * 100

    return PredictResponse(
        ticker=ticker,
        predicted_price=round(predicted_price, 2),
        change=round(change, 2),
        return_pct=round(return_pct, 4),
        mse=mse,
        rmse=rmse,
        history=history,
    )


async def _train_in_background(ticker: str, training_status: dict):
    """Run factory.make_model() in a thread, then mark status ready."""
    try:
        await asyncio.to_thread(factory.make_model, ticker)
        training_status[ticker] = "ready"
    except Exception:
        # If training fails, remove the key so the next request retries
        training_status.pop(ticker, None)


@router.post("/predict")
async def predict(body: PredictRequest, request: Request):
    ticker = body.ticker.upper().strip()
    training_status: dict = request.app.state.training_status

    # Validate ticker
    info = yf.Ticker(ticker).info
    if info.get("regularMarketPrice") is None:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    # If already training, tell the client to keep polling
    if training_status.get(ticker) == "training":
        return JSONResponse(status_code=202, content={"status": "training", "ticker": ticker})

    # If model is fresh, predict immediately
    if _model_is_fresh(ticker):
        result = _load_and_predict(ticker)
        return result

    # Model is stale or missing — start background training
    training_status[ticker] = "training"
    asyncio.create_task(_train_in_background(ticker, training_status))
    return JSONResponse(status_code=202, content={"status": "training", "ticker": ticker})


@router.get("/status/{ticker}", response_model=StatusResponse)
async def status(ticker: str, request: Request):
    ticker = ticker.upper().strip()
    training_status: dict = request.app.state.training_status

    if training_status.get(ticker) == "training":
        return StatusResponse(ticker=ticker, status="training")

    if _model_is_fresh(ticker):
        return StatusResponse(ticker=ticker, status="ready")

    return StatusResponse(ticker=ticker, status="not_found")


@router.get("/popular", response_model=PopularResponse)
async def popular():
    stocks = []
    for ticker_sym in POPULAR_TICKERS:
        info = yf.Ticker(ticker_sym).info
        price = info.get("regularMarketPrice") or info.get("previousClose") or 0.0
        name = info.get("longName") or ticker_sym
        stocks.append(PopularStock(ticker=ticker_sym, name=name, price=round(float(price), 2)))
    return PopularResponse(stocks=stocks)
