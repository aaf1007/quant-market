# Stock Market Predictor

This project is a simple cli stock price prediction tool that uses **linear regression** on historical stock data.  
It currently focuses on predicting the next-day closing price for a given stock using features such as:

- Close price
- Volume
- 60-day moving average
- 60-day rolling standard deviation

The core model lives in `backend/model/linear_regression_pred.py`.

---

## Features

- **Data source**: Fetches historical stock data via `yfinance`.
- **Feature engineering**:
  - 60-day moving average of close price
  - 60-day rolling standard deviation of close price
- **Model**: Linear Regression (`scikit-learn`)
- **Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

---

## Project Structure

```text
backend/
  requirements.txt        # Python dependencies
  model/
    linear_regression_pred.py  # Training & evaluation script
    stock_info.json            # (Reserved for stock-related metadata, if needed)
```

---

## Getting Started

### 1. Prerequisites

- Python 3.12 (recommended)
- `pip` or `pipx` (for installing dependencies)
- (Optional) `virtualenv` or `pyenv` for virtual environments

### 2. Setup Environment

From the `backend` directory:

```bash
cd backend

# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Linear Regression Script

From the `backend` directory:

```bash
python model/linear_regression_pred.py
```

This will:

1. Create features:
   - `Close`
   - `Volume`
   - `Moving_Average_60`
   - `Rolling_STD_60`
2. Split the data into training and test sets (80% / 20%).
3. Train a `LinearRegression` model.
4. Compute and print:
   - Features used
   - Mean Squared Error
   - Root Mean Squared Error

---

## Requirements

From `backend/requirements.txt`:

- `yfinance`
- `streamlit`
- `scikit-learn`

Install them via:

```bash
pip install -r backend/requirements.txt
```

---

## Future Improvements (Ideas)

- Wrap the model in a **FastAPI** or **Streamlit** app for a simple UI.
- Add support for:
  - Multiple tickers and batch runs.
  - Hyperparameter tuning and cross-validation.
  - Additional technical indicators as features.
- Persist trained models and metrics.
- Add unit tests and CI.


