"""
config.example.py — Copy this to config.py and fill in your keys.
On Railway: set these as environment variables instead (recommended).
"""

import os

BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

SYMBOL        = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL      = os.environ.get("INTERVAL", "1h")
LOOKBACK      = os.environ.get("LOOKBACK", "365 day ago UTC")
BUY_THRESHOLD = float(os.environ.get("BUY_THRESHOLD", "0.55"))

PAPER_STARTING_BALANCE = float(os.environ.get("PAPER_STARTING_BALANCE", "10000"))
TRADE_SIZE_PCT         = float(os.environ.get("TRADE_SIZE_PCT", "0.95"))
LOG_FILE               = os.environ.get("LOG_FILE", "trades_log.csv")

MODEL_FILE   = "model.pkl"
RETRAIN_DAYS = int(os.environ.get("RETRAIN_DAYS", "7"))
