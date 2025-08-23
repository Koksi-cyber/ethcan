"""
Realtime signal generation for ETH/USDT futures trading with leverage.

This script fetches the latest one‑minute candlesticks from Binance's public
REST API, computes the same engineered features used during training,
standardises them with the saved scaler and obtains a probability from the
trained XGBoost model.  If the probability exceeds the configured threshold,
the script emits a long signal along with calculated take‑profit (TP) and
stop‑loss (SL) price levels based on the specified leverage and margin.

Usage:
    python realtime_signal.py --leverage 50 --margin 10

The leverage value should be between 20 and 70 as per user requirements. The
margin is the capital (in USDT) allocated to each trade.  The default
configuration uses the median leverage (45x) and 10 USDT margin.  Adjust
`--threshold` to change the sensitivity of the model; higher thresholds yield
fewer, more confident signals.

Note: this script does not execute any trades. It only prints potential
signals and their parameters.  Integrate with a trading system at your
discretion.
"""

import argparse
import datetime as dt
import json
import math
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import xgboost as xgb  # required for typing and model loading


def fetch_recent_candles(symbol: str = "ETHUSDT", limit: int = 150) -> pd.DataFrame:
    """Fetch recent one‑minute candles from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (default 'ETHUSDT').
    limit : int
        Number of recent minutes to fetch (up to 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    # Parse into DataFrame
    rows = []
    for k in data:
        open_time = dt.datetime.fromtimestamp(k[0] / 1000.0)
        open_p, high_p, low_p, close_p, volume = map(float, k[1:6])
        rows.append({
            "timestamp": open_time,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        })
    df = pd.DataFrame(rows)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features on a DataFrame of candles.

    The feature definitions mirror those used during model training.
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Basic returns and ratios
    df["close_prev"] = df["close"].shift(1)
    df["ret1m"] = np.log(df["close"] / df["close_prev"])
    df["open_close_pct"] = (df["close"] - df["open"]) / df["open"]
    df["high_open_pct"] = (df["high"] - df["open"]) / df["open"]
    df["low_open_pct"] = (df["open"] - df["low"]) / df["open"]
    # Rolling statistics
    for w in [5, 10, 20, 60]:
        df[f"ret_mean_{w}"] = df["ret1m"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret1m"].rolling(w).std()
        df[f"vol_mean_{w}"] = df["volume"].rolling(w).mean()
        df[f"vol_std_{w}"] = df["volume"].rolling(w).std()
    # Exponential moving average difference
    df["ema_fast"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=60, adjust=False).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
    df.dropna(inplace=True)
    return df


def generate_signal(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    scaler: StandardScaler,
    feature_cols: List[str],
    threshold: float,
    leverage: float,
    margin: float,
) -> Tuple[bool, dict]:
    """Generate a trading signal based on the latest candle features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of recent candles with features computed.
    model : xgb.XGBClassifier
        Trained XGBoost model.
    scaler : StandardScaler
        Scaler used during training.
    feature_cols : list
        List of feature column names in order.
    threshold : float
        Probability threshold for generating a long signal.
    leverage : float
        Selected leverage (between 20 and 70).
    margin : float
        Margin in USDT for each trade (≥10).

    Returns
    -------
    (bool, dict)
        Tuple where first element indicates whether a signal is generated,
        and second element contains signal details (entry price, TP, SL,
        predicted probability).
    """
    latest = df.iloc[-1]
    X = df[feature_cols].iloc[-1:].to_numpy()
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X)[:, 1])
    signal = prob >= threshold
    details = {
        "timestamp": latest["timestamp"],
        "probability": prob,
    }
    if signal:
        entry_price = latest["open"]
        # Compute profit and loss price thresholds based on desired ROI and loss
        profit_pct = 1.5 / leverage
        loss_pct = 10.0 / (margin * leverage)
        details.update({
            "entry_price": entry_price,
            "take_profit_price": entry_price * (1 + profit_pct),
            "stop_loss_price": entry_price * (1 - loss_pct),
            "profit_pct": profit_pct,
            "loss_pct": loss_pct,
        })
    return signal, details


def main():
    parser = argparse.ArgumentParser(description="ETH/USDT realtime signal generator with leverage support")
    parser.add_argument("--leverage", type=float, default=45.0, help="Leverage (20–70), default 45")
    parser.add_argument("--margin", type=float, default=10.0, help="Margin in USDT (≥10), default 10")
    parser.add_argument("--threshold", type=float, default=None, help="Custom probability threshold; if not supplied, uses threshold saved with model")
    parser.add_argument("--limit", type=int, default=150, help="Number of recent candles to fetch")
    args = parser.parse_args()

    if args.leverage < 20 or args.leverage > 70:
        raise ValueError("Leverage must be between 20 and 70")
    if args.margin < 10:
        raise ValueError("Margin must be at least 10 USDT")

    # Load model and scaler
    model_path = os.path.join(os.path.dirname(__file__), "eth_signal_model_leverage.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model_data = joblib.load(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]
    threshold = args.threshold if args.threshold is not None else model_data.get("threshold", 0.5)

    # Fetch candles and compute features
    candles = fetch_recent_candles(limit=args.limit)
    features_df = compute_features(candles)
    if features_df.empty:
        print("Not enough data to compute features.")
        return
    # Generate signal
    signal, details = generate_signal(features_df, model, scaler, feature_cols, threshold, args.leverage, args.margin)
    if signal:
        print("LONG signal detected")
        print(json.dumps(details, default=str, indent=2))
    else:
        print(f"No trade signal (probability={details['probability']:.4f} < threshold={threshold:.4f})")


if __name__ == "__main__":
    main()