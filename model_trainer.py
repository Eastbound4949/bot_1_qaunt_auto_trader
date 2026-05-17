"""
model_trainer.py v2 — XGBoost with threshold-based labels, 30+ features,
lag features, time patterns, feature selection, walk-forward CV.

Usage:
    python model_trainer.py
"""

import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import ta
import xgboost as xgb
from binance.client import Client
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_score
from sklearn.model_selection import TimeSeriesSplit

import config

warnings.filterwarnings("ignore")

# ─── Label config ─────────────────────────────────────────────────────────────
LABEL_HORIZON   = 6      # Look ahead N candles
LABEL_THRESHOLD = 0.005  # Price must rise >0.5% to be labelled BUY

# ─── Top features kept after importance selection ──────────────────────────────
TOP_N_FEATURES = 25

# Set by add_features(); bot.py reads from pickle payload instead
FEATURE_COLS = []


# ─── Data fetch ───────────────────────────────────────────────────────────────

def fetch_binance_data(symbol, interval, lookback):
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    raw = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    print(f"[trainer] Got {len(df)} candles.")
    return df[["open", "high", "low", "close", "volume"]]


# ─── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # Momentum
    df["rsi"]        = ta.momentum.RSIIndicator(close, 14).rsi()
    df["rsi_6"]      = ta.momentum.RSIIndicator(close, 6).rsi()
    df["stoch_k"]    = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df["stoch_d"]    = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()
    df["cci"]        = ta.trend.CCIIndicator(high, low, close).cci()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
    df["mfi"]        = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index()

    # Trend
    macd = ta.trend.MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    adx = ta.trend.ADXIndicator(high, low, close)
    df["adx"]         = adx.adx()
    df["adx_pos"]     = adx.adx_pos()
    df["adx_neg"]     = adx.adx_neg()
    df["ema_9"]       = ta.trend.EMAIndicator(close, 9).ema_indicator()
    df["ema_21"]      = ta.trend.EMAIndicator(close, 21).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(close, 50).ema_indicator()

    # Volatility
    bb = ta.volatility.BollingerBands(close, 20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb.bollinger_mavg()
    df["bb_pos"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["atr"]      = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
    df["atr_pct"]  = df["atr"] / close

    # Volume
    df["vol_ratio"] = vol / vol.rolling(20).mean()
    df["vol_z"]     = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)
    obv = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["obv_diff"]  = (obv - obv.ewm(span=20).mean()) / (obv.ewm(span=20).mean().abs() + 1e-9)

    # Price-derived
    df["returns"]    = close.pct_change()
    df["log_ret"]    = np.log(close / close.shift(1))
    df["range_pct"]  = (high - low) / close
    df["body_pct"]   = (close - df["open"]).abs() / (high - low + 1e-9)
    df["upper_wick"] = (high - np.maximum(close, df["open"])) / (high - low + 1e-9)
    df["lower_wick"] = (np.minimum(close, df["open"]) - low) / (high - low + 1e-9)

    # Price vs EMAs (normalized)
    df["price_vs_ema9"]  = (close - df["ema_9"])  / df["ema_9"]
    df["price_vs_ema21"] = (close - df["ema_21"]) / df["ema_21"]
    df["price_vs_ema50"] = (close - df["ema_50"]) / df["ema_50"]

    # Multi-period returns
    df["ret_3"]  = close.pct_change(3)
    df["ret_6"]  = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)
    df["ret_24"] = close.pct_change(24)

    # Rolling stats
    df["ret_mean_12"] = df["returns"].rolling(12).mean()
    df["ret_std_12"]  = df["returns"].rolling(12).std()
    df["ret_mean_24"] = df["returns"].rolling(24).mean()

    # Time features — crypto has strong hourly/daily patterns
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Lag features — short-term momentum memory
    for col in ["rsi", "macd_diff", "vol_ratio", "returns", "bb_pos"]:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df.dropna(inplace=True)
    return df


# ─── Labels ───────────────────────────────────────────────────────────────────

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 = price rises >LABEL_THRESHOLD% within next LABEL_HORIZON candles
    0 = otherwise

    Using max future return instead of next-candle direction filters noise
    and focuses on meaningful moves the bot can actually profit from.
    """
    future_max = pd.Series(np.nan, index=df.index)
    for i in range(1, LABEL_HORIZON + 1):
        ret = df["close"].shift(-i) / df["close"] - 1
        future_max = np.fmax(future_max, ret)

    df["label"] = (future_max > LABEL_THRESHOLD).astype(int)
    df.dropna(subset=["label"], inplace=True)
    return df.iloc[:-LABEL_HORIZON]  # Last N rows have no future data


# ─── Feature selection ────────────────────────────────────────────────────────

def _select_top_features(model, cols, top_n):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    selected = [cols[i] for i in idx]
    print(f"\n[trainer] Top 10 features by importance:")
    for rank, col in enumerate(selected[:10], 1):
        print(f"  {rank:2d}. {col:30s} {imp[idx[rank-1]]:.4f}")
    return selected


# ─── Walk-forward precision ───────────────────────────────────────────────────

def _walk_forward_precision(X: pd.DataFrame, y: pd.Series, n_splits=5) -> tuple:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=10, gamma=2,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        scores.append(precision_score(y_te, preds, pos_label=1, zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


# ─── Train & save ─────────────────────────────────────────────────────────────

def train_and_save():
    print(f"[trainer] Fetching {config.SYMBOL} {config.INTERVAL} | {config.LOOKBACK}...")
    df = fetch_binance_data(config.SYMBOL, config.INTERVAL, config.LOOKBACK)

    print("[trainer] Engineering features...")
    df = add_features(df)
    df = add_labels(df)

    skip = ["label", "open", "high", "low", "close", "volume"]
    all_cols = [c for c in df.columns if c not in skip]

    X = df[all_cols]
    y = df["label"]
    print(f"[trainer] Rows: {len(df)} | BUY rate: {y.mean():.1%} | Features: {len(all_cols)}")

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Pass 1: all features, get importances (no pos_weight — keeps probs calibrated)
    m1 = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, gamma=2,
        reg_alpha=0.5, reg_lambda=2,
        eval_metric="logloss", random_state=42, verbosity=0,
        early_stopping_rounds=50,
    )
    m1.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    selected = _select_top_features(m1, all_cols, TOP_N_FEATURES)

    # Pass 2: retrain on selected features only
    m2 = xgb.XGBClassifier(
        n_estimators=m1.best_iteration + 1,
        max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, gamma=2,
        reg_alpha=0.5, reg_lambda=2,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    m2.fit(X_tr[selected], y_tr)

    # Calibrate probabilities so confidence scores are meaningful
    # cv=3 uses 3-fold internal CV on training data
    calibrated = CalibratedClassifierCV(m2, cv=3, method="isotonic")
    calibrated.fit(X_tr[selected], y_tr)

    y_pred = calibrated.predict(X_te[selected])
    y_prob = calibrated.predict_proba(X_te[selected])[:, 1]

    print(f"\n[trainer] === Hold-out test set ===")
    print(classification_report(y_te, y_pred, target_names=["HOLD", "BUY"]))

    # Precision at multiple thresholds — find the tradeable zone
    print(f"\n[trainer] Precision by confidence threshold:")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Signals':>10} {'Coverage':>10}")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = y_prob >= thresh
        if mask.sum() > 0:
            prec = (y_te[mask] == 1).mean()
            cov  = mask.sum() / len(y_te)
            print(f"  {thresh:>10.0%} {prec:>10.1%} {mask.sum():>10d} {cov:>10.1%}")

    print(f"\n[trainer] Walk-forward precision (5-fold)...")
    wf_mean, wf_std = _walk_forward_precision(X[selected], y)
    print(f"  BUY precision: {wf_mean:.1%} ± {wf_std:.1%}")

    payload = {
        "model":            calibrated,
        "feature_cols":     selected,
        "trained_at":       datetime.utcnow().isoformat(),
        "label_horizon":    LABEL_HORIZON,
        "label_threshold":  LABEL_THRESHOLD,
        "symbol":           config.SYMBOL,
    }
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[trainer] Saved -> {config.MODEL_FILE}")
    return payload


if __name__ == "__main__":
    train_and_save()
