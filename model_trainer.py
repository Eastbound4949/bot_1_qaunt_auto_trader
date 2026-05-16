"""
model_trainer.py — Fetches data from Binance, trains the ML model, saves it.
Run this once to create model.pkl, then the bot loads it automatically.
Re-run every week (or set RETRAIN_DAYS in config.py for auto-retraining).

Usage:
    python model_trainer.py
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

import config


# ─── Step 1: Fetch OHLCV data from Binance ───────────────────────────────────

def fetch_binance_data(symbol, interval, lookback):
    """Download candle data directly from Binance (free, no subscription)."""
    print(f"[trainer] Fetching {symbol} {interval} data from Binance...")
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

    raw = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    print(f"[trainer] Got {len(df)} candles.")
    return df


# ─── Step 2: Feature engineering (same as crypto_model.py) ───────────────────

def add_features(df):
    close = df["close"]
    vol   = df["volume"]

    df["rsi"]        = RSIIndicator(close=close, window=14).rsi()
    macd_obj         = MACD(close=close)
    df["macd"]       = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_diff"]  = macd_obj.macd_diff()
    df["vol_ratio"]  = vol / vol.rolling(20).mean()
    df["returns"]    = close.pct_change()
    df["volatility"] = df["returns"].rolling(14).std()
    df["ma_20"]      = close.rolling(20).mean()
    df["ma_50"]      = close.rolling(50).mean()
    df["price_vs_20"] = (close - df["ma_20"]) / df["ma_20"]
    df["price_vs_50"] = (close - df["ma_50"]) / df["ma_50"]
    df["ret_1d"]     = close.pct_change(1)
    df["ret_3d"]     = close.pct_change(3)
    df["ret_5d"]     = close.pct_change(5)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_diff",
    "vol_ratio", "volatility",
    "price_vs_20", "price_vs_50",
    "ret_1d", "ret_3d", "ret_5d",
]


# ─── Step 3: Add labels ───────────────────────────────────────────────────────

def add_labels(df, forward_periods=1):
    df["label"] = (df["close"].shift(-forward_periods) > df["close"]).astype(int)
    df = df.iloc[:-forward_periods]
    return df


# ─── Step 4: Train & save ─────────────────────────────────────────────────────

def train_and_save():
    df = fetch_binance_data(config.SYMBOL, config.INTERVAL, config.LOOKBACK)
    df = add_features(df)
    df = add_labels(df)

    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean() * 100
    print(f"\n[trainer] Test accuracy: {acc:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["SELL","BUY"]))

    # Save model + metadata
    payload = {
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "trained_at":   datetime.utcnow().isoformat(),
        "accuracy":     round(acc, 2),
        "symbol":       config.SYMBOL,
    }
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"[trainer] Model saved to {config.MODEL_FILE}")
    return payload


if __name__ == "__main__":
    train_and_save()
