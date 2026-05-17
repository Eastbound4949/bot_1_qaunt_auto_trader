# Crypto ML Auto Trader — Bot v3

Regime-aware, multi-pair, adaptive crypto trading bot using XGBoost with multi-timeframe feature engineering, ATR-based risk management, and walk-forward validated signals.

---

## Current Performance (BTCUSDT · 1h · 730-day lookback)

| Metric | Value |
|---|---|
| Walk-forward precision | **44.1%** |
| Break-even threshold | 37.5% |
| Expected value per trade | **+18% of risked amount** |
| Risk per trade | 1% of portfolio |
| R:R ratio | 1.67 : 1 (2.5 ATR target : 1.5 ATR stop) |
| BUY signal rate | ~10% of candles at 58% threshold |
| Training data | 730 days (17,315 candles) |
| BUY label rate | 26.9% of candles |

> Walk-forward precision = % of BUY signals that resulted in profitable exits on unseen future data.
> Break-even at 1.67 R:R = 37.5%. We are at 44.1% → positive edge confirmed.

---

## Pairs and Timeframes

| Pair | Model file | Status |
|---|---|---|
| BTCUSDT | model_BTCUSDT_1h.pkl | ✅ Trained (44.1% WF precision) |
| ETHUSDT | model_ETHUSDT_1h.pkl | Train before deploying |
| SOLUSDT | model_SOLUSDT_1h.pkl | Train before deploying |

**Base timeframe:** 1h candles
**Higher timeframe:** 4h candles (merged as confirmation features)
**Training lookback:** 730 days (2 years)

---

## Objective

Paper trade crypto perpetuals using a machine learning model that:
1. Only enters when market regime is trending (not choppy/ranging)
2. Sizes positions by ATR-based risk (not fixed %)
3. Exits via take-profit or trailing stop (not just ML SELL signal)
4. Scans multiple pairs and picks the highest-confidence setup each hour
5. Auto-retrains weekly with exponential recency weighting

Graduate to live trading automatically when performance metrics hit targets (Sharpe > 1.0, win rate > 52%, drawdown < 20%, 4 weeks paper trading, 50+ trades).

---

## Theories and Principles

### 1. Supervised Binary Classification (XGBoost)
Label each historical candle: **BUY (1)** if price rises >0.5% within 6 candles without hitting the stop-loss. Train XGBoost to predict this probability on new candles.

**Why XGBoost:** Handles non-linear feature interactions, robust to noisy data, fast to train and predict, built-in handling of NaN (missing HTF values), feature importance for selection.

### 2. Multi-Timeframe Analysis
Base signals (1h) are confirmed against higher-timeframe context (4h). A bullish 1h setup in a 4h downtrend is filtered out. HTF features included:
- 4h RSI, MACD diff (normalized), ADX
- Price vs 4h EMA50/EMA200
- 4h Bollinger Band position
- `htf_trend_up`: both EMAs stacked and price above both

### 3. Regime Detection
Markets alternate between **trending** (momentum works) and **ranging** (mean reversion works). Using the wrong strategy in the wrong regime destroys win rate. Gate: only enter when ALL conditions met:
- ADX > 20 (trending, not choppy)
- EMA alignment score ≥ 2/3 (9 > 21 > 50 stacking)
- Price above EMA200 (macro uptrend)
- 4h trend up (HTF confirmation)
- Not in extreme volatility regime (rolling vol percentile < 70%)

### 4. ATR-Based Risk Management
Average True Range (ATR) measures current market volatility. Stops and targets set in ATR units rather than fixed percentages — they adapt to market conditions automatically.

```
Position size = (Portfolio × Risk%) / (ATR × Stop multiplier)
Take-profit   = Entry + 2.5 × ATR
Trailing stop = Running high - 1.5 × ATR  (moves up, never down)
Hard stop     = Entry × (1 - 2.5%)        (catastrophic floor)
```

**Why R:R = 2.5:1.5:** Requires only 37.5% win rate to profit. With 44.1% walk-forward precision, expected value is positive on every trade.

### 5. Kelly-Inspired Position Sizing
Risk a fixed fraction (1%) of portfolio per trade rather than a fixed size. This:
- Prevents ruin (position sizes shrink after losses)
- Compounds gains (position sizes grow with portfolio)
- ATR scaling means stops are never hit by normal volatility

### 6. Recency-Weighted Training
Crypto market regime changes over time. Recent data is more predictive than data from 2 years ago. Each training candle gets weight `0.9997^(candles_ago)`. Most recent 500 candles carry ~3× the weight of candles from 18 months ago.

### 7. Walk-Forward Validation
Standard train/test split is insufficient for time-series. Walk-forward splits data into 5 sequential folds, trains on each past segment, tests on the next future segment. Reports average precision across folds — this is the realistic live-performance estimate.

**3-way split used (prevents data leakage):**
- Train (70%): model training
- Val (15%): early stopping only — never seen by calibrator or final metrics
- Test (15%): calibration + honest hold-out metrics

### 8. Isotonic Probability Calibration
Raw XGBoost probabilities are not well-calibrated (a 70% predicted probability does not mean 70% of those signals win). Isotonic regression maps raw probabilities to empirically observed win rates using the hold-out test set. Result: `predict_proba()[1] > 0.58` means roughly 44% of those signals will win — a usable, honest number.

### 9. Label Engineering
Labels predict: "will price rise >0.5% in the next 6 candles without hitting the stop-loss?". Key design choices:
- **0.5% threshold (not 1%):** Keeps BUY rate at ~27% — enough positive examples for the model to learn
- **Stop-loss check in label:** Never labels a setup as BUY if the stop would have been hit first
- **No reward/risk filter in labels:** The bot's ATR exits handle R:R. Pre-filtering labels by R:R just starves the model of training signal

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  model_trainer.py                                       │
│  ─────────────────                                      │
│  Binance API → OHLCV (1h + 4h)                         │
│       ↓                                                 │
│  add_features() → 75 features (base + HTF + regime)    │
│       ↓                                                 │
│  add_labels()  → BUY/HOLD labels (0.5% threshold)      │
│       ↓                                                 │
│  XGBoost Pass 1 → feature importance → select top 25   │
│       ↓                                                 │
│  XGBoost Pass 2 → recency-weighted refit on train+val  │
│       ↓                                                 │
│  IsotonicCalibration → calibrate on test set           │
│       ↓                                                 │
│  Walk-forward validation → report real precision        │
│       ↓                                                 │
│  Save model_SYMBOL_INTERVAL.pkl                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  bot.py (runs every 1h)                                 │
│  ─────────────────────                                  │
│  If in position:                                        │
│    → Update trailing stop                               │
│    → Check TP / trailing stop / hard stop / ML SELL     │
│    → Exit if triggered                                  │
│                                                         │
│  If no position:                                        │
│    → Scan BTCUSDT, ETHUSDT, SOLUSDT in parallel        │
│    → For each: fetch 1h+4h data → add_features() →    │
│      load model → predict_proba()                      │
│    → Pick highest-probability pair                      │
│    → Check regime gate (ADX + EMA + HTF trend)         │
│    → If prob ≥ 0.58 AND regime OK: enter position      │
│    → Size = (Portfolio × 1%) / (1.5 × ATR)            │
│                                                         │
│  Log to trades_log.csv                                  │
│  Send Telegram alert                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Feature Groups (75 total → top 25 selected)

| Group | Features |
|---|---|
| Momentum | RSI(14), RSI(6), Stochastic %K/%D, CCI, Williams %R, MFI |
| Trend | MACD diff (normalized), ADX/+DI/-DI, EMA 9/21/50/200 spreads |
| Volatility | Bollinger Band width/position, ATR%, realized vol percentile |
| Volume | OBV diff, volume ratio, volume z-score, taker buy ratio |
| Price action | Returns 3/6/12/24, rolling mean/std, candle body/wick ratios |
| Regime | ADX regime flag, EMA alignment score, trend strength, vol regime |
| Time | Hour sin/cos, day-of-week sin/cos (crypto has strong session patterns) |
| Lag | RSI, MACD, vol_ratio, returns, BB_pos lagged 1/2/3 bars |
| HTF (4h) | RSI, MACD diff, ADX, EMA50/200 flags, BB position, trend_up flag |

**Top features by importance:** `range_pct`, `dow_sin`, `price_vs_ema21`, `trend_strength`, `htf_bb_pos`, `htf_rsi`, `price_vs_ema50`, `atr_pct`

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Configuration
```bash
cp config.example.py config.py
# Edit config.py — add your Binance API key and Telegram token
```

**Binance:** Create API key at binance.com → Account → API Management. Read-only permissions are enough for paper trading.

**Telegram:** Message @BotFather → `/newbot` → copy token. Message @userinfobot → copy chat ID.

### Train Models
```bash
cd "path/to/bot 1 quant"

# Train all pairs (BTCUSDT + ETHUSDT + SOLUSDT)
python model_trainer.py

# Or train a specific pair
python model_trainer.py BTCUSDT
python model_trainer.py ETHUSDT SOLUSDT
```

**Check walk-forward output before deploying.** Look for:
```
@ BUY threshold 58%: 44.1%  [ok]
```
If it says `[LOSING (need >37%)]` for a pair — remove that pair from `SYMBOLS` in config.py.

**Set BUY_THRESHOLD** from the precision table:
```
Threshold  Precision  Signals
      58%      44.1%      ...   ← use this row
      62%      49.3%      ...   ← fewer trades, higher precision
      65%      55.0%      ...   ← very selective
```
Pick the threshold where precision first exceeds 40%. Higher threshold = fewer trades, higher win rate, lower total profit.

### Start Bot (Paper Trading)
```bash
python bot.py
```

Runs immediately, then every hour. Telegram alerts on every signal and trade.

---

## Monitoring

### performance_monitor.py
Reads `trades_log.csv` weekly and sends Telegram report. Checks promotion criteria:

| Metric | Target |
|---|---|
| Paper trading duration | ≥ 4 weeks |
| Completed trades | ≥ 50 |
| Sharpe ratio (annualised) | > 1.0 |
| Win rate | > 52% |
| Max drawdown | < 20% |

When all targets hit → writes `promoted_to_live.flag` + Telegram alert requiring manual confirmation to go live.

```bash
python performance_monitor.py   # run manually for current metrics
```

### trades_log.csv columns
`timestamp, symbol, price, buy_prob, signal, action, balance_usdt, position_units, entry_price, take_profit_price, trail_stop_price, portfolio_usd, total_trades, total_wins, win_rate_pct`

---

## Auto-Retraining

Model auto-retrains when `trained_at` timestamp in `.pkl` is older than `RETRAIN_DAYS` (default: 7 days). Retrain uses exponential recency weighting — the model continuously adapts to current market conditions.

To force retrain:
```bash
python model_trainer.py BTCUSDT
```

---

## Key Config Parameters

```python
# config.py
SYMBOLS              = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL             = "1h"
LOOKBACK             = "730 day ago UTC"

BUY_THRESHOLD        = 0.58    # set from walk-forward table
SELL_THRESHOLD       = 0.38

RISK_PER_TRADE       = 0.01    # 1% of portfolio risked per trade
STOP_LOSS_PCT        = 0.025   # hard stop floor (2.5%)
TAKE_PROFIT_ATR_MULT = 2.5     # TP at entry + 2.5 × ATR
TRAIL_STOP_ATR_MULT  = 1.5     # trail stop 1.5 × ATR below high

RETRAIN_DAYS         = 7       # auto-retrain interval
```

---

## Files

```
bot.py                  Main trading loop (runs 24/7)
model_trainer.py        Train/retrain ML models
config.py               Your keys and settings (git-ignored)
config.example.py       Template — copy to config.py
performance_monitor.py  Weekly metrics and live-promotion check
requirements.txt        Python dependencies
trades_log.csv          All signals and trades (runtime, git-ignored)
model_BTCUSDT_1h.pkl    Trained model (runtime, git-ignored)
model_ETHUSDT_1h.pkl    Trained model (runtime, git-ignored)
model_SOLUSDT_1h.pkl    Trained model (runtime, git-ignored)
```

---

## Disclaimer

This bot is for paper trading and educational purposes. Past walk-forward performance does not guarantee future results. Crypto markets are volatile. Never risk money you cannot afford to lose. Review all code before running with real funds.
