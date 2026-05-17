---
  Railway Setup (10 minutes)

  Step 1 — Create account

  Go to railway.app → sign up with GitHub (free)

  Step 2 — New project

  - Click New Project
  - Select Deploy from GitHub repo
  - Choose Eastbound4949/bot_1_qaunt_auto_trader
  - Railway auto-detects Python and railway.toml

  Step 3 — Set environment variables

  In Railway dashboard → your service → Variables tab → add these one by one:

  ┌────────────────────────┬─────────────────────────┐
  │        Variable        │          Value          │
  ├────────────────────────┼─────────────────────────┤
  │ BINANCE_API_KEY        │ your key                │
  ├────────────────────────┼─────────────────────────┤
  │ BINANCE_API_SECRET     │ your secret             │
  ├────────────────────────┼─────────────────────────┤
  │ TELEGRAM_TOKEN         │ your bot token          │
  ├────────────────────────┼─────────────────────────┤
  │ TELEGRAM_CHAT_ID       │ your chat ID            │
  ├────────────────────────┼─────────────────────────┤
  │ SYMBOLS                │ BTCUSDT,ETHUSDT,SOLUSDT │
  ├────────────────────────┼─────────────────────────┤
  │ INTERVAL               │ 1h                      │
  ├────────────────────────┼─────────────────────────┤
  │ LOOKBACK               │ 730 day ago UTC         │
  ├────────────────────────┼─────────────────────────┤
  │ BUY_THRESHOLD          │ 0.58                    │
  ├────────────────────────┼─────────────────────────┤
  │ PAPER_STARTING_BALANCE │ 10000                   │
  └────────────────────────┴─────────────────────────┘

  Step 4 — Deploy

  Click Deploy. Railway runs start.sh which:
  1. Creates config.py from env vars
  2. Trains all 3 models (~30 min first time — watch logs)
  3. Starts bot — runs 24/7, auto-restarts on crash

  Step 5 — Watch logs

  Dashboard → your service → Deployments → View Logs

  First deploy log will look like:
  [setup] Created config.py from config.example.py
  [setup] All required env vars present.
  [trainer] Fetching BTCUSDT 1h | 730 day ago UTC...
  ...
  [setup] All models ready. Starting bot...

  Step 6 — (Optional) Add volume for persistence

  Without volume: models retrain every redeploy (~30 min downtime).
  With volume (Railway Hobby $5/month): models persist.
  - Dashboard → your service → Volumes → Add → mount path /data
  - Add env var DATA_DIR = /data

  ---
  Every git push auto-redeploys. Bot sends Telegram when it starts.
