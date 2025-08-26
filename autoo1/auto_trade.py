"""
Automated trading bot for ETHUSDT perpetual futures.

This script combines real‑time data fetching, feature engineering,
signal generation with the provided machine‑learning model, and
order placement on Binance Futures.  It demonstrates how you could
automate the strategy tested in previous backtests.

Important:
  * This code is for educational purposes only.  High leverage
    trading carries significant risk.  Use at your own risk and test
    thoroughly on Binance’s testnet before deploying to a live
    account.
  * The model predicts only upward moves (long trades).  It does
    not handle short trades or other market conditions.
  * You must provide your own Binance API key and secret in a
    `.env` file (see `.env.example`).

"""

import os
import time
from decimal import Decimal
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *  # noqa: F401,F403

import pickle

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Path to the trained model (use the retrained model for best results)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "eth_signal_model_retrained.pkl")

# Trading parameters
SYMBOL = "ETHUSDT"
INTERVAL = "1m"
# Number of candles to fetch – must be at least as many as were used to
# compute rolling features during training.  We use 200, which is
# consistent with the backtest engine.
CANDLE_LIMIT = 200
LEVERAGE = 150  # 150× leverage as per the request
MARGIN_USDT = Decimal("16")  # 16 USDT margin, implying ~0.5 ETH at 150×

# Horizon and take‑profit/stop‑loss percentages are loaded from the model

# -----------------------------------------------------------------------------
# Helper functions for feature engineering and signal generation
# -----------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features identical to those used during model training.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns ['timestamp','open','high','low','close','volume']

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    # Previous close for log return
    df["close_prev"] = df["close"].shift(1)
    df["ret1m"] = np.log(df["close"] / df["close_prev"])
    # Price ratio features
    df["open_close_pct"] = (df["close"] - df["open"]) / df["open"]
    df["high_open_pct"] = (df["high"] - df["open"]) / df["open"]
    df["low_open_pct"] = (df["open"] - df["low"]) / df["open"]
    # Rolling statistics (5, 10, 20, 60 minutes)
    for w in [5, 10, 20, 60]:
        df[f"ret_mean_{w}"] = df["ret1m"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret1m"].rolling(w).std()
        df[f"vol_mean_{w}"] = df["volume"].rolling(w).mean()
        df[f"vol_std_{w}"] = df["volume"].rolling(w).std()
    # Exponential moving average difference
    df["ema_fast"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=60, adjust=False).mean()
    df["ema_diff"] = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"]
    # Drop rows with NaNs introduced by rolling windows
    return df.dropna().reset_index(drop=True)


def fetch_recent_candles(client: Client, symbol: str = SYMBOL, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """Fetch recent one‑minute candles from Binance Futures.

    Parameters
    ----------
    client : Client
        The Binance client instance.
    symbol : str
        Trading pair symbol.
    limit : int
        Number of recent candles to retrieve (up to 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the candle data.
    """
    # Binance futures klines return: open_time, open, high, low, close, volume,
    # close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
    # taker_buy_quote_asset_volume, ignore
    raw = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=limit)
    rows = []
    for k in raw:
        timestamp = datetime.fromtimestamp(k[0] / 1000.0)
        open_p, high_p, low_p, close_p, volume = map(Decimal, [k[1], k[2], k[3], k[4], k[5]])
        rows.append({
            "timestamp": timestamp,
            "open": float(open_p),
            "high": float(high_p),
            "low": float(low_p),
            "close": float(close_p),
            "volume": float(volume),
        })
    return pd.DataFrame(rows)


def load_model(path: str = MODEL_PATH) -> dict:
    """Load the saved model and associated configuration."""
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_signal(features_df: pd.DataFrame, model_data: dict) -> dict | None:
    """Generate a trade signal based on the latest row in the feature DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with engineered features.
    model_data : dict
        The loaded model dictionary.

    Returns
    -------
    dict or None
        A dictionary with signal details or None if no trade is signalled.
    """
    if features_df.empty:
        return None
    latest = features_df.iloc[-1]
    X = latest[model_data["feature_cols"]].to_frame().T
    X_scaled = model_data["scaler"].transform(X)
    # Compute probability
    if model_data.get("ensemble", False):
        probs = np.mean([m.predict_proba(X_scaled)[:, 1] for m in model_data["models"]], axis=0)
        prob = float(probs[0])
    else:
        prob = float(model_data["models"][0].predict_proba(X_scaled)[:, 1])
    threshold = model_data.get("threshold", 0.5)
    if prob < threshold:
        return None
    # Entry price is the open of the most recent candle
    entry_price = latest["open"]
    tp_pct = model_data.get("tp_pct")
    sl_pct = model_data.get("sl_pct")
    # Fallback in case tp_pct and sl_pct are missing
    if tp_pct is None or sl_pct is None:
        tp_pct = float(1.5 / LEVERAGE)
        sl_pct = float(10.0 / (float(MARGIN_USDT) * LEVERAGE))
    signal = {
        "timestamp": latest["timestamp"],
        "probability": prob,
        "entry_price": entry_price,
        "tp_price": entry_price * (1 + tp_pct),
        "sl_price": entry_price * (1 - sl_pct),
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
    }
    return signal

# -----------------------------------------------------------------------------
# API call wrapper with retry logic
# -----------------------------------------------------------------------------

def safe_api_call(func, desc: str = "API call", retries: int = 3, delay: float = 2.0):
    """Execute a Binance API call with retry and exception handling.

    Parameters
    ----------
    func : callable
        A callable that makes a single API request.
    desc : str
        Description of the call for logging.
    retries : int
        Number of retry attempts on failure.
    delay : float
        Delay between retries in seconds.

    Returns
    -------
    Any
        The return value of the API call if successful.
    """
    attempt = 0
    while True:
        try:
            return func()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            print(f"{desc} failed (attempt {attempt}/{retries}): {e}. Retrying in {delay}s...")
            time.sleep(delay)


def place_trade(client: Client, signal: dict, model_data: dict):
    """Place a long trade on Binance Futures with TP and SL orders.

    Parameters
    ----------
    client : Client
        Authenticated Binance client.
    signal : dict
        Signal dictionary returned by `generate_signal`.

    This function sends a market buy order sized such that the notional
    value is equal to `MARGIN_USDT * LEVERAGE`, then places a limit
    take‑profit order and a stop‑market stop‑loss order.  It monitors
    the position for `model_data['horizon']` minutes and closes the
    position at market if neither TP nor SL is triggered.
    """
    # Current price for sizing
    price = Decimal(str(signal["entry_price"]))
    notional = MARGIN_USDT * Decimal(LEVERAGE)
    quantity = (notional / price).quantize(Decimal("0.0001"))

    # Ensure the futures account uses the desired leverage with retries
    safe_api_call(lambda: client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE), desc="change leverage")

    # Open market long position and wait for full response
    order = safe_api_call(
        lambda: client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_BUY,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=float(quantity),
            newOrderRespType="FULL"
        ),
        desc="market buy"
    )
    print(f"Opened long position: qty={quantity}, entry_price={price}")

    # Compute TP and SL prices with safety buffer relative to liquidation price
    tp_price = Decimal(str(signal["tp_price"])).quantize(Decimal("0.01"))
    sl_price_raw = Decimal(str(signal["sl_price"])).quantize(Decimal("0.01"))
    # Approximate liquidation threshold: price * (1 - 1/leverage)
    # Add a small buffer of 0.1% above that threshold
    liquidation_pct = Decimal("1") - (Decimal("1") / Decimal(LEVERAGE))
    safety_buffer_pct = Decimal("0.001")
    min_sl_pct = liquidation_pct + safety_buffer_pct
    min_sl_price = price * min_sl_pct
    sl_price = max(sl_price_raw, min_sl_price).quantize(Decimal("0.01"))

    # Place TP and SL orders as MARKET types to ensure execution
    tp_order = safe_api_call(
        lambda: client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=str(tp_price),
            closePosition=True,
            newOrderRespType="FULL",
            reduceOnly=True
        ),
        desc="take profit"
    )
    sl_order = safe_api_call(
        lambda: client.futures_create_order(
            symbol=SYMBOL,
            side=SIDE_SELL,
            type=FUTURE_ORDER_TYPE_STOP_MARKET,
            stopPrice=str(sl_price),
            closePosition=True,
            newOrderRespType="FULL",
            reduceOnly=True
        ),
        desc="stop loss"
    )
    print(f"Set TP at {tp_price} and SL at {sl_price}")

    # Monitor position
    horizon_minutes = model_data['horizon']
    end_time = datetime.utcnow() + timedelta(minutes=horizon_minutes)
    position_closed = False
    while datetime.utcnow() < end_time:
        try:
            info = safe_api_call(lambda: client.futures_position_information(symbol=SYMBOL)[0], desc="position info")
            pos_amt = Decimal(info['positionAmt'])
            if pos_amt == 0:
                print("Position closed by TP or SL.")
                position_closed = True
                break
        except Exception as e:
            print("Error checking position:", e)
        time.sleep(5)

    if not position_closed:
        # Close at market
        try:
            safe_api_call(lambda: client.futures_create_order(
                symbol=SYMBOL,
                side=SIDE_SELL,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=float(pos_amt),
                reduceOnly=True,
                newOrderRespType="FULL"
            ), desc="forced close")
            print("Closed position after horizon elapsed.")
        finally:
            safe_api_call(lambda: client.futures_cancel_all_open_orders(symbol=SYMBOL), desc="cancel all")


def main():
    """Main loop: load model, connect to Binance, and trade based on signals."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        raise RuntimeError('Please set BINANCE_API_KEY and BINANCE_API_SECRET in your .env file.')
    client = Client(api_key, api_secret)

    # Load model once
    model_data = load_model(MODEL_PATH)

    print("Starting trading bot with model threshold:", model_data.get('threshold'))
    print("Using leverage:", LEVERAGE, "and margin (USDT):", MARGIN_USDT)

    while True:
        try:
            # Synchronise with Binance server time to start just after a new minute
            srv = safe_api_call(lambda: client.futures_time(), desc="fetch server time")
            server_ms = srv['serverTime']
            server_seconds = server_ms / 1000.0
            seconds_into_minute = server_seconds % 60
            wait = 60 - seconds_into_minute + 0.5  # 0.5s buffer
            if wait > 0 and wait < 60:
                time.sleep(wait)

            # Fetch recent candles and compute features
            candles = fetch_recent_candles(client, symbol=SYMBOL, limit=CANDLE_LIMIT)
            features_df = compute_features(candles)
            # Generate signal
            signal = generate_signal(features_df, model_data)
            if signal:
                print(f"Signal detected at {signal['timestamp']}: prob={signal['probability']:.4f}")
                place_trade(client, signal, model_data)
            else:
                print(f"{datetime.utcnow()} – no trade signal")
        except Exception as e:
            print("Error:", e)
        # Loop continues automatically; sleep a short time to avoid tight loop
        time.sleep(1)


if __name__ == '__main__':
    main()