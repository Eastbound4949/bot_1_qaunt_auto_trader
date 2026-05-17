"""
config.example.py — Copy this to config.py and fill in your keys.
On Railway/server: set these as environment variables (recommended — never commit real keys).

cp config.example.py config.py
# then edit config.py with your real values
"""

import os

# ─── Binance API (READ-ONLY permissions sufficient for paper trading) ──────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY",    "YOUR_BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "YOUR_BINANCE_API_SECRET")

# ─── Telegram (@BotFather for token, @userinfobot for chat ID) ─────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

# ─── Trading settings ──────────────────────────────────────────────────────────
# Exchange: "bybit" works from any server/region. "binance" requires non-US IP.
EXCHANGE = os.environ.get("EXCHANGE", "bybit")
SYMBOL   = os.environ.get("SYMBOL",   "BTCUSDT")
SYMBOLS  = os.environ.get("SYMBOLS",  "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
INTERVAL = os.environ.get("INTERVAL", "1h")
LOOKBACK = os.environ.get("LOOKBACK", "730 day ago UTC")

BUY_THRESHOLD  = float(os.environ.get("BUY_THRESHOLD",  "0.58"))
SELL_THRESHOLD = float(os.environ.get("SELL_THRESHOLD", "0.38"))

# ─── Risk management ───────────────────────────────────────────────────────────
PAPER_STARTING_BALANCE = float(os.environ.get("PAPER_STARTING_BALANCE", "10000"))
RISK_PER_TRADE         = float(os.environ.get("RISK_PER_TRADE",         "0.01"))
MAX_POSITION_PCT       = float(os.environ.get("MAX_POSITION_PCT",       "0.90"))
STOP_LOSS_PCT          = float(os.environ.get("STOP_LOSS_PCT",          "0.025"))
TAKE_PROFIT_ATR_MULT   = float(os.environ.get("TAKE_PROFIT_ATR_MULT",   "2.5"))
TRAIL_STOP_ATR_MULT    = float(os.environ.get("TRAIL_STOP_ATR_MULT",    "1.5"))
TRADE_SIZE_PCT         = float(os.environ.get("TRADE_SIZE_PCT",         "0.95"))
LOG_FILE               = os.environ.get("LOG_FILE", "trades_log.csv")

# ─── Regime filter ─────────────────────────────────────────────────────────────
TREND_FILTER        = os.environ.get("TREND_FILTER",       "true").lower() == "true"
ADX_TREND_THRESHOLD = int(os.environ.get("ADX_TREND_THRESHOLD", "20"))
EMA_ALIGN_THRESHOLD = int(os.environ.get("EMA_ALIGN_THRESHOLD", "2"))
REQUIRE_HTF_TREND   = os.environ.get("REQUIRE_HTF_TREND",  "true").lower() == "true"

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_FILE   = "model.pkl"
RETRAIN_DAYS = int(os.environ.get("RETRAIN_DAYS", "7"))
