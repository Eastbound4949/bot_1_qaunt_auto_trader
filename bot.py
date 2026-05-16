"""
bot.py — The live paper trading bot.
Runs on a schedule, fetches live Binance data, makes ML predictions,
logs simulated trades, and sends Telegram alerts.

Usage:
    python bot.py

Keep this running 24/7 on your cloud server.
"""

import pickle
import pandas as pd
import numpy as np
import requests
import csv
import os
from datetime import datetime
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from apscheduler.schedulers.blocking import BlockingScheduler
import warnings
warnings.filterwarnings("ignore")

import config
from model_trainer import add_features, FEATURE_COLS, train_and_save


# ─── Telegram ────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    """Send a message to your Telegram bot."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"[telegram] Failed to send: {e}")


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_latest_data():
    """Fetch the most recent candles from Binance."""
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    raw = client.get_historical_klines(config.SYMBOL, config.INTERVAL, config.LOOKBACK)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    """Load the saved ML model from disk."""
    if not os.path.exists(config.MODEL_FILE):
        print("[bot] No model found — training now...")
        train_and_save()

    with open(config.MODEL_FILE, "rb") as f:
        payload = pickle.load(f)

    # Auto-retrain if model is too old
    trained_at = datetime.fromisoformat(payload["trained_at"])
    age_days = (datetime.utcnow() - trained_at).days
    if age_days >= config.RETRAIN_DAYS:
        print(f"[bot] Model is {age_days} days old — retraining...")
        payload = train_and_save()
        with open(config.MODEL_FILE, "rb") as f:
            payload = pickle.load(f)

    return payload["model"]


# ─── Paper trade state ────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulates trading without real money.
    Tracks: balance, position, P&L, trade history.
    """
    def __init__(self):
        self.balance  = config.PAPER_STARTING_BALANCE  # USDT
        self.position = 0.0   # BTC held
        self.entry_price = 0.0
        self.trades = 0
        self.wins   = 0
        self._load_state()

    def _load_state(self):
        """Restore balance & position from log file if it exists."""
        if not os.path.exists(config.LOG_FILE):
            return
        try:
            df = pd.read_csv(config.LOG_FILE)
            if len(df):
                last = df.iloc[-1]
                self.balance  = float(last["balance_usdt"])
                self.position = float(last["position_btc"])
                self.entry_price = float(last.get("entry_price", 0))
                self.trades = int(last.get("total_trades", 0))
                self.wins   = int(last.get("total_wins", 0))
                print(f"[bot] Restored state: ${self.balance:,.2f} USDT, {self.position:.6f} BTC")
        except Exception as e:
            print(f"[bot] Could not restore state: {e}")

    def execute(self, signal: str, price: float) -> str:
        """
        Execute a paper trade.
        signal: "BUY" or "SELL"
        price: current market price
        Returns: description of what happened
        """
        result = ""

        if signal == "BUY" and self.position == 0 and self.balance > 10:
            # Enter position: buy BTC with most of our USDT
            spend = self.balance * config.TRADE_SIZE_PCT
            self.position = spend / price
            self.balance -= spend
            self.entry_price = price
            result = f"BOUGHT {self.position:.6f} BTC @ ${price:,.2f}"
            self.trades += 1

        elif signal == "SELL" and self.position > 0:
            # Exit position: sell all BTC back to USDT
            proceeds = self.position * price
            pnl = proceeds - (self.position * self.entry_price)
            pnl_pct = (price / self.entry_price - 1) * 100
            if pnl > 0:
                self.wins += 1
            self.balance += proceeds
            result = (
                f"SOLD {self.position:.6f} BTC @ ${price:,.2f} | "
                f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
            )
            self.position = 0.0
            self.entry_price = 0.0
            self.trades += 1

        else:
            result = "HOLD — no action taken"

        return result

    def portfolio_value(self, price: float) -> float:
        return self.balance + self.position * price

    def log_trade(self, timestamp, signal, price, buy_prob, action_taken):
        """Append a row to the CSV trade log."""
        portfolio_val = self.portfolio_value(price)
        win_rate = (self.wins / max(self.trades, 1)) * 100
        row = {
            "timestamp":     timestamp,
            "symbol":        config.SYMBOL,
            "price":         round(price, 2),
            "buy_prob":      round(buy_prob, 4),
            "signal":        signal,
            "action":        action_taken,
            "balance_usdt":  round(self.balance, 2),
            "position_btc":  round(self.position, 8),
            "entry_price":   round(self.entry_price, 2),
            "portfolio_usd": round(portfolio_val, 2),
            "total_trades":  self.trades,
            "total_wins":    self.wins,
            "win_rate_pct":  round(win_rate, 1),
        }
        file_exists = os.path.exists(config.LOG_FILE)
        with open(config.LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


# ─── Main loop ────────────────────────────────────────────────────────────────

paper_trader = PaperTrader()

def run_bot():
    """
    Called by the scheduler every interval.
    1. Fetch latest data
    2. Compute features
    3. Get ML prediction
    4. Execute paper trade
    5. Send Telegram alert
    6. Log everything
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'─'*50}")
    print(f"[bot] Running at {now}")

    try:
        # 1. Data
        df = fetch_latest_data()
        df = add_features(df)

        # 2. Predict on the most recent candle
        model = load_model()
        latest = df[FEATURE_COLS].iloc[[-1]]
        buy_prob = model.predict_proba(latest)[0][1]
        price    = df["close"].iloc[-1]

        # 3. Signal
        signal = "BUY" if buy_prob > config.BUY_THRESHOLD else "SELL"
        print(f"[bot] Price: ${price:,.2f} | Buy prob: {buy_prob:.1%} | Signal: {signal}")

        # 4. Paper trade
        action = paper_trader.execute(signal, price)
        portfolio_val = paper_trader.portfolio_value(price)
        total_return = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100

        print(f"[bot] {action}")
        print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}% total)")

        # 5. Telegram alert
        win_rate = (paper_trader.wins / max(paper_trader.trades, 1)) * 100
        msg = (
            f"*{config.SYMBOL} Signal — {now}*\n"
            f"Price: ${price:,.2f}\n"
            f"ML signal: *{signal}* ({buy_prob:.1%} confidence)\n"
            f"Action: {action}\n"
            f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
            f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
        )
        send_telegram(msg)

        # 6. Log
        paper_trader.log_trade(now, signal, price, buy_prob, action)

    except Exception as e:
        err = f"[bot] ERROR: {e}"
        print(err)
        send_telegram(f"Bot error at {now}:\n{err}")


# ─── Scheduler ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print(" Crypto ML Paper Trading Bot")
    print(f" Symbol:   {config.SYMBOL}")
    print(f" Interval: {config.INTERVAL}")
    print(f" Balance:  ${config.PAPER_STARTING_BALANCE:,.0f} (paper)")
    print("=" * 50)

    # Run immediately on start
    run_bot()

    # Then schedule on the interval
    scheduler = BlockingScheduler()

    # Map Binance intervals to APScheduler minutes
    interval_map = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440,
    }
    minutes = interval_map.get(config.INTERVAL, 60)
    scheduler.add_job(run_bot, "interval", minutes=minutes)

    print(f"\n[bot] Scheduler started — running every {minutes} minutes. Ctrl+C to stop.\n")
    send_telegram(f"Bot started for {config.SYMBOL} on {config.INTERVAL} interval.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[bot] Stopped by user.")
        send_telegram("Bot stopped by user.")
