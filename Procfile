# Procfile — tells Railway how to start your bot
# Railway reads this automatically. No changes needed.
web: python bot.py


# ════════════════════════════════════════════════════════════════
# railway.toml — Railway project configuration
# Place this in the root of your repository.
# ════════════════════════════════════════════════════════════════

# [build]
# builder = "NIXPACKS"   # Railway auto-detects Python — no Docker needed

# [deploy]
# startCommand = "python bot.py"
# healthcheckPath = "/"
# restartPolicyType = "ON_FAILURE"
# restartPolicyMaxRetries = 3


# ════════════════════════════════════════════════════════════════
# RAILWAY SETUP GUIDE (takes ~10 minutes)
# ════════════════════════════════════════════════════════════════
#
# 1. Go to railway.app → sign up free with GitHub
#
# 2. New Project → Deploy from GitHub repo → select your repo
#
# 3. In Railway dashboard → your service → Variables tab
#    Add these environment variables (DO NOT hardcode keys in config.py):
#
#    BINANCE_API_KEY      = your_key_here
#    BINANCE_API_SECRET   = your_secret_here
#    TELEGRAM_TOKEN       = your_telegram_bot_token
#    TELEGRAM_CHAT_ID     = your_chat_id
#    PAPER_STARTING_BALANCE = 10000
#    BUY_THRESHOLD        = 0.55
#    LIVE_TRADING         = false     ← keep this false until bot is promoted
#
# 4. Update config.py to read from environment variables:
#    (copy the config_from_env.py file to config.py)
#
# 5. Railway auto-deploys on every git push to main.
#    Every push → tests run → if passed → bot restarts with new code.
#
# 6. View live logs: Railway dashboard → your service → Deployments → View Logs


# ════════════════════════════════════════════════════════════════
# config_from_env.py — Replace config.py contents with this
# so secrets come from Railway environment variables, not code.
# ════════════════════════════════════════════════════════════════

import os

BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

SYMBOL        = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL      = os.environ.get("INTERVAL", "1h")
LOOKBACK      = os.environ.get("LOOKBACK", "90 day ago UTC")
BUY_THRESHOLD = float(os.environ.get("BUY_THRESHOLD", "0.55"))

PAPER_STARTING_BALANCE = float(os.environ.get("PAPER_STARTING_BALANCE", "10000"))
TRADE_SIZE_PCT         = float(os.environ.get("TRADE_SIZE_PCT", "0.95"))
LOG_FILE               = os.environ.get("LOG_FILE", "trades_log.csv")
LIVE_TRADING           = os.environ.get("LIVE_TRADING", "false").lower() == "true"

MODEL_FILE   = "model.pkl"
RETRAIN_DAYS = int(os.environ.get("RETRAIN_DAYS", "7"))
