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
SYMBOL   = "BTCUSDT"
SYMBOLS  = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]   # All pairs scanned each tick
INTERVAL = "1h"                                 # 1m/5m/15m/1h/4h/1d
LOOKBACK = "730 day ago UTC"                    # 2 years of training data

# Tune based on walk-forward precision table printed by model_trainer.py
# Set to threshold where precision first exceeds 40%
BUY_THRESHOLD  = 0.58
SELL_THRESHOLD = 0.38

# ─── Risk management ───────────────────────────────────────────────────────────
PAPER_STARTING_BALANCE = 10_000   # Simulated USDT
RISK_PER_TRADE         = 0.01     # Risk 1% of portfolio per trade (ATR-based sizing)
MAX_POSITION_PCT       = 0.90     # Cap position at 90% of balance
STOP_LOSS_PCT          = 0.025    # Hard stop floor: 2.5% below entry
TAKE_PROFIT_ATR_MULT   = 2.5      # Take-profit at entry + 2.5 × ATR
TRAIL_STOP_ATR_MULT    = 1.5      # Trailing stop 1.5 × ATR below running high
TRADE_SIZE_PCT         = 0.95     # Legacy — not used in v3
LOG_FILE               = "trades_log.csv"

# ─── Regime filter ─────────────────────────────────────────────────────────────
TREND_FILTER        = True   # Only BUY when price > EMA200
ADX_TREND_THRESHOLD = 20     # Minimum ADX for trending market
EMA_ALIGN_THRESHOLD = 2      # Minimum EMA alignment score (0–3)
REQUIRE_HTF_TREND   = True   # Higher-TF (4h when on 1h) must also be in uptrend

# ─── Model ─────────────────────────────────────────────────────────────────────
MODEL_FILE   = "model.pkl"   # Single-symbol fallback name
RETRAIN_DAYS = 7             # Auto-retrain every N days
