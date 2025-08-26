# Auto Trading Bot for ETHUSDT Futures

This folder contains a fully‑assembled example of how you might automate
signal generation and trade execution on Binance Futures using the
retrained model from the earlier backtest.  **Use this code at your own
risk**—it is provided for educational purposes only.  Make sure you
understand the risks of trading with high leverage, and always test
thoroughly on Binance’s testnet before deploying to a live account.

## Contents

* `eth_signal_model_retrained.pkl` – the machine‑learning model that
  generates signals.  It contains the XGBoost model, scaler,
  engineered feature names, threshold, TP/SL percentages and horizon.
* `realtime_eth_signal_retrained.py` – a helper script to fetch
  recent candles from Binance and compute features consistent with
  those used in training.  It returns a dictionary describing a
  potential long trade when the model’s probability exceeds the
  threshold.
* `auto_trade.py` – a sample trading bot that loads API keys from a
  `.env` file, fetches the most recent candles every minute, uses
  the model to generate signals, and places/monitors trades on
  Binance Futures.  It uses 150× leverage and a 16 USDT margin,
  which is roughly equivalent to a 0.5 ETH notional position at
  current prices.
* `.env.example` – a template for your Binance API keys.  Copy this
  file to `.env` and fill in your API key and secret before running
  the bot.
* `requirements.txt` – a list of Python packages needed to run the
  bot.

## Quick start

1. **Create a `.env` file**

   ```
   cp .env.example .env
   # then edit `.env` and insert your BINANCE_API_KEY and BINANCE_API_SECRET
   ```

2. **Install dependencies**

   It’s recommended to use a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the bot**

   ```bash
   python auto_trade.py
   ```

The bot will fetch the latest 200 one‑minute candles for `ETHUSDT`,
compute features, generate a signal if appropriate, and place a
market order with TP and SL set at ±0.4 % and ±0.2 % relative to
the entry price.  It monitors the position for 10 minutes and
closes it at market if neither TP nor SL is triggered.

## Safety notes

* **High leverage is risky.**  This example uses 150× leverage on
  16 USDT margin.  Even small market moves can wipe out a position.
* **Test before deploying.**  Use Binance’s testnet by passing
  `testnet=True` to the `Client` constructor in `auto_trade.py`.
* **No guarantees.**  Past backtest performance does not guarantee
  future results.  You are responsible for any trades executed with
  this code.