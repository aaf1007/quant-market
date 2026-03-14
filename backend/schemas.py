from pydantic import BaseModel


class PredictRequest(BaseModel):
    ticker: str


class HistoryPoint(BaseModel):
    date: str
    close: float


class PredictResponse(BaseModel):
    ticker: str
    predicted_price: float
    change: float
    return_pct: float
    mse: float
    rmse: float
    history: list[HistoryPoint]


class StatusResponse(BaseModel):
    ticker: str
    status: str  # "ready", "training", or "not_found"


class PopularStock(BaseModel):
    ticker: str
    name: str
    price: float


class PopularResponse(BaseModel):
    stocks: list[PopularStock]
