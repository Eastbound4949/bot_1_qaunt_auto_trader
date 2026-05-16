"""
config.py — All settings in one place.
Fill in your keys, then never touch this file again.
"""

# ─── Binance API ───────────────────────────────────────────────
# Get these from: https://www.binance.com → Account → API Management
# For paper trading, READ-ONLY permissions are enough (no trading needed)
BINANCE_API_KEY    = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET"

# ─── Telegram Alerts ───────────────────────────────────────────
# Step 1: Message @BotFather on Telegram → /newbot → copy the token
# Step 2: Message @userinfobot on Telegram → copy your Chat ID
TELEGRAM_TOKEN   = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# ─── Trading settings ──────────────────────────────────────────
SYMBOL        = "BTCUSDT"          # Binance pair to trade
INTERVAL      = "1h"               # Candle size: 1m, 5m, 15m, 1h, 4h, 1d
LOOKBACK      = "90 day ago UTC"   # How far back to fetch (for indicators)
BUY_THRESHOLD = 0.55               # Model confidence needed to signal BUY

# ─── Paper trading ─────────────────────────────────────────────
PAPER_STARTING_BALANCE = 10_000    # Simulated starting balance in USDT
TRADE_SIZE_PCT         = 0.95      # Use 95% of balance per trade
LOG_FILE               = "trades_log.csv"

# ─── Model ─────────────────────────────────────────────────────
MODEL_FILE     = "model.pkl"       # Saved model filename
RETRAIN_DAYS   = 7                 # Retrain the model every N days
