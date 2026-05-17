"""
backtest.py — Historical win rate analysis per trading pair.

Usage:
    python backtest.py

Requires trained models (run model_trainer.py first for ML precision).
Falls back to label-based ground-truth win rate if no model is found.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
from model_trainer import (
    LABEL_HORIZON,
    LABEL_THRESHOLD,
    STOP_LOSS_THRESHOLD,
    REWARD_RISK_MIN,
    add_features,
    add_labels,
    fetch_binance_data,
    fetch_htf_data,
    model_file_for,
)

THRESHOLDS = [0.55, 0.60, 0.62, 0.65, 0.70, 0.75]


def _backtest_symbol(symbol: str) -> dict:
    print(f"[backtest] {symbol} — fetching {config.LOOKBACK}...")
    try:
        df     = fetch_binance_data(symbol, config.INTERVAL, config.LOOKBACK)
        htf_df = fetch_htf_data(symbol, config.INTERVAL, config.LOOKBACK)
        df     = add_features(df, htf_df)
        df     = add_labels(df)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    total       = len(df)
    buy_count   = int(df["label"].sum())
    label_wr    = float(df["label"].mean())

    # Simulate simple strategy: buy on label=1, measure forward return
    close   = df["close"]
    fwd_ret = close.shift(-LABEL_HORIZON) / close - 1
    df["fwd_ret"] = fwd_ret

    # Per-label outcome breakdown
    wins_label  = int(((df["label"] == 1) & (df["fwd_ret"] > 0)).sum())
    label_buys  = buy_count
    label_wr_actual = wins_label / max(label_buys, 1)

    # Average forward return on BUY labels
    avg_fwd = float(df.loc[df["label"] == 1, "fwd_ret"].mean()) if label_buys > 0 else 0.0

    # ML model precision per threshold (if model exists)
    ml_rows: dict[float, dict] = {}
    mf = model_file_for(symbol)
    if os.path.exists(mf):
        try:
            with open(mf, "rb") as f:
                payload = pickle.load(f)
            model = payload["model"]
            fcols = payload["feature_cols"]

            for c in fcols:
                if c not in df.columns:
                    df[c] = np.nan

            probs = model.predict_proba(df[fcols])[:, 1]

            for thresh in THRESHOLDS:
                mask = probs >= thresh
                n    = int(mask.sum())
                if n >= 3:
                    precision = float(df["label"][mask].mean())
                    avg_ret   = float(df.loc[mask, "fwd_ret"].mean())
                    ml_rows[thresh] = {
                        "precision": precision,
                        "signals":   n,
                        "coverage":  n / total,
                        "avg_fwd_ret": avg_ret,
                    }
        except Exception as e:
            print(f"[backtest] {symbol} model load error: {e}")
    else:
        print(f"[backtest] {symbol}: no model found at {mf} — skipping ML metrics")

    return {
        "symbol":        symbol,
        "total_candles": total,
        "buy_labels":    label_buys,
        "label_wr":      label_wr,
        "label_wr_actual": label_wr_actual,
        "avg_fwd_ret":   avg_fwd,
        "ml":            ml_rows,
        "model_file":    mf if os.path.exists(mf) else None,
    }


def run_backtest():
    print("=" * 70)
    print("  Historical Win Rate Analysis")
    print(f"  Pairs:    {', '.join(config.SYMBOLS)}")
    print(f"  Interval: {config.INTERVAL}  |  Lookback: {config.LOOKBACK}")
    print(f"  Label:    price rises >{LABEL_THRESHOLD:.0%} in {LABEL_HORIZON} candles, "
          f"no >{STOP_LOSS_THRESHOLD:.0%} dip, R:R >={REWARD_RISK_MIN}")
    print("=" * 70)

    results = [_backtest_symbol(sym) for sym in config.SYMBOLS]

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'Symbol':<12} {'Candles':>8} {'Buy Labels':>11} "
          f"{'Label WR':>10} {'Avg Fwd Ret':>12} {'Model':>8}")
    print(f"{'─'*70}")
    for r in results:
        if "error" in r:
            print(f"  {r['symbol']:<12}  ERROR: {r['error']}")
            continue
        model_tag = "yes" if r["model_file"] else "none"
        print(
            f"  {r['symbol']:<12} {r['total_candles']:>8,} {r['buy_labels']:>11,} "
            f"{r['label_wr']:>10.1%} {r['avg_fwd_ret']:>+12.2%} {model_tag:>8}"
        )

    # ── ML precision per threshold ─────────────────────────────────────────────
    for r in results:
        if "error" in r or not r.get("ml"):
            continue
        print(f"\n  {r['symbol']} — ML model precision by BUY threshold:")
        print(f"  {'Threshold':>10} {'Win Rate':>10} {'Signals':>9} "
              f"{'Coverage':>10} {'Avg Fwd Ret':>13}")
        print(f"  {'─'*55}")
        for thresh, d in sorted(r["ml"].items()):
            marker = " <-- active" if abs(thresh - config.BUY_THRESHOLD) < 0.005 else ""
            print(
                f"  {thresh:>10.0%} {d['precision']:>10.1%} {d['signals']:>9,} "
                f"{d['coverage']:>10.1%} {d['avg_fwd_ret']:>+13.2%}{marker}"
            )

    print(f"\n{'─'*70}")
    print("  Legend:")
    print(f"   Label WR     = % of candles meeting forward-return criteria (ground truth)")
    print(f"   ML Win Rate  = precision of model predictions at each threshold")
    print(f"   Avg Fwd Ret  = mean {LABEL_HORIZON}-candle return on predicted BUY signals")
    print(f"   Active threshold: {config.BUY_THRESHOLD:.0%} (config.BUY_THRESHOLD)")
    if any(r.get("model_file") is None for r in results if "error" not in r):
        print(f"\n  Train missing models: python model_trainer.py")
    print()

    return results


if __name__ == "__main__":
    run_backtest()
