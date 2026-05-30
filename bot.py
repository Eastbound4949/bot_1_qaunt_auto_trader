"""
bot.py v3 — Regime-aware, multi-pair, adaptive trading bot.

Upgrades over v2:
  - ATR-based take-profit (2.5× ATR above entry)
  - Trailing stop (1.5× ATR below running high, hard floor at -2.5%)
  - ATR-based position sizing: risk exactly 1% of portfolio per trade
  - Regime gate: only trade when ADX trending + EMA stack aligned + HTF uptrend
  - Multi-pair scanner: picks highest-confidence signal across all SYMBOLS
  - Per-symbol model files (model_BTCUSDT_1h.pkl, etc.)
"""

import csv
import os
import pickle
from datetime import datetime

_BOT_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

from apscheduler.schedulers.blocking import BlockingScheduler
import config
from model_trainer import (
    add_features,
    fetch_binance_data,
    fetch_htf_data,
    model_file_for,
    short_model_file_for,
    train_and_save,
    train_and_save_short,
)


# ─── Telegram ────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    except Exception as e:
        print(f"[telegram] Failed: {e}")


# ─── Data fetching ────────────────────────────────────────────────────────────

def fetch_latest_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Returns (base_tf_df, htf_df) using ccxt. Exchange set by EXCHANGE env var."""
    df     = fetch_binance_data(symbol, config.INTERVAL, config.LOOKBACK)
    htf_df = fetch_htf_data(symbol, config.INTERVAL, config.LOOKBACK)
    return df, htf_df


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(symbol: str) -> tuple:
    """Load (model, feature_cols) for symbol. Trains if stale or missing."""
    mf = model_file_for(symbol)

    if not os.path.exists(mf):
        print(f"[bot] No model for {symbol} — training now...")
        train_and_save(symbol, mf)

    with open(mf, "rb") as f:
        payload = pickle.load(f)

    age_days = (datetime.utcnow() - datetime.fromisoformat(payload["trained_at"])).days
    if age_days >= config.RETRAIN_DAYS:
        print(f"[bot] {symbol} model is {age_days}d old — retraining...")
        train_and_save(symbol, mf)
        with open(mf, "rb") as f:
            payload = pickle.load(f)

    return payload["model"], payload["feature_cols"]


def load_short_model(symbol: str) -> tuple | None:
    """Load (short_model, feature_cols). Returns None if ENABLE_SHORTING is false."""
    if not config.ENABLE_SHORTING:
        return None
    mf = short_model_file_for(symbol)
    if not os.path.exists(mf):
        print(f"[bot] No short model for {symbol} — training now...")
        train_and_save_short(symbol, mf)
    with open(mf, "rb") as f:
        payload = pickle.load(f)
    age_days = (datetime.utcnow() - datetime.fromisoformat(payload["trained_at"])).days
    if age_days >= config.RETRAIN_DAYS:
        print(f"[bot] {symbol} short model is {age_days}d old — retraining...")
        train_and_save_short(symbol, mf)
        with open(mf, "rb") as f:
            payload = pickle.load(f)
    return payload["model"], payload["feature_cols"]


# ─── Regime filter ────────────────────────────────────────────────────────────

def _regime_ok(df: pd.DataFrame) -> bool:
    """Return True only when market conditions justify a long entry."""
    row = df.iloc[-1]

    def get(col, default):
        return row[col] if col in df.columns else default

    above_ema200    = bool(get("above_ema200", 1))
    adx_trending    = bool(get("regime_trending", 0)) or get("ema_alignment", 0) >= config.EMA_ALIGN_THRESHOLD
    no_extreme_vol  = not bool(get("regime_high_vol", 0))
    htf_trend       = not config.REQUIRE_HTF_TREND or bool(get("htf_trend_up", 1))

    return (
        (not config.TREND_FILTER or above_ema200) and
        adx_trending and
        no_extreme_vol and
        htf_trend
    )


def _bearish_regime_ok(df: pd.DataFrame) -> bool:
    """Return True only when market conditions justify a short entry."""
    row = df.iloc[-1]

    def get(col, default):
        return row[col] if col in df.columns else default

    below_ema200   = not bool(get("above_ema200", 1))
    ema_bearish    = int(get("ema_alignment", 3)) <= 1  # at most 1 of 3 EMAs bullish
    adx_trending   = bool(get("regime_trending", 0))    # need confirmed trend (down or up, ADX>25)
    no_extreme_vol = not bool(get("regime_high_vol", 0))
    htf_trend_down = not bool(get("htf_trend_up", 1))

    return (
        (not config.TREND_FILTER or below_ema200) and
        ema_bearish and
        adx_trending and
        no_extreme_vol and
        (not config.REQUIRE_HTF_TREND or htf_trend_down)
    )


# ─── Paper trade state ────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulates trading without real money.

    Position sizing: risk RISK_PER_TRADE% of portfolio per trade, sized by ATR.
    Exit logic:      take-profit at entry + TAKE_PROFIT_ATR_MULT × ATR
                     trailing stop 1.5 × ATR below running high
                     hard stop floor at entry × (1 - STOP_LOSS_PCT)
                     ML SELL signal also exits
    """

    _log_file = os.path.join(_BOT_DIR, config.LOG_FILE)

    def __init__(self):
        self.balance         = config.PAPER_STARTING_BALANCE
        self.position        = 0.0    # long: coin units held
        self.entry_price     = 0.0
        self.take_profit_price = 0.0
        self.trail_stop_price  = 0.0
        self.position_symbol = ""
        # Short position state (paper short: receive proceeds at entry, pay at cover)
        self.short_position    = 0.0  # units shorted
        self.short_entry_price = 0.0
        self.short_take_profit = 0.0  # TP is below entry for shorts
        self.short_trail_stop  = 0.0  # trailing stop is above entry, moves down
        self.short_symbol      = ""
        self.trades            = 0
        self.wins              = 0
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self._log_file):
            return
        try:
            df = pd.read_csv(self._log_file)
            if len(df):
                last = df.iloc[-1]
                self.balance           = float(last["balance_usdt"])
                self.position          = float(last["position_units"])
                self.entry_price       = float(last.get("entry_price", 0))
                self.take_profit_price = float(last.get("take_profit_price", 0))
                self.trail_stop_price  = float(last.get("trail_stop_price", 0))
                self.position_symbol   = str(last.get("position_symbol", ""))
                self.short_position    = float(last.get("short_units", 0))
                self.short_entry_price = float(last.get("short_entry_price", 0))
                self.short_take_profit = float(last.get("short_take_profit", 0))
                self.short_trail_stop  = float(last.get("short_trail_stop", 0))
                self.short_symbol      = str(last.get("short_symbol", ""))
                self.trades            = int(last.get("total_trades", 0))
                self.wins              = int(last.get("total_wins", 0))
                print(
                    f"[bot] Restored: ${self.balance:,.2f} USDT | "
                    f"{self.position:.6f} {self.position_symbol or 'none'}"
                )
        except Exception as e:
            print(f"[bot] Could not restore state: {e}")

    # ── Exit helpers ──────────────────────────────────────────────────────────

    def _close_position(self, price: float, reason: str) -> str:
        proceeds = self.position * price
        pnl      = proceeds - (self.position * self.entry_price)
        pnl_pct  = (price / self.entry_price - 1) * 100
        if pnl > 0:
            self.wins += 1
        self.trades           += 1
        self.balance          += proceeds
        result = (
            f"{reason} {self.position:.6f} {self.position_symbol} "
            f"@ ${price:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        self.position          = 0.0
        self.entry_price       = 0.0
        self.take_profit_price = 0.0
        self.trail_stop_price  = 0.0
        self.position_symbol   = ""
        return result

    def _close_short(self, price: float, reason: str) -> str:
        # At entry: balance += short_units × entry_price (received proceeds)
        # At cover: balance -= short_units × cover_price (pay to buy back)
        cost_to_cover = self.short_position * price
        pnl           = self.short_position * (self.short_entry_price - price)
        pnl_pct       = (self.short_entry_price / price - 1) * 100
        if pnl > 0:
            self.wins  += 1
        self.trades    += 1
        self.balance   -= cost_to_cover
        result = (
            f"{reason} {self.short_position:.6f} {self.short_symbol} "
            f"@ ${price:,.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        self.short_position    = 0.0
        self.short_entry_price = 0.0
        self.short_take_profit = 0.0
        self.short_trail_stop  = 0.0
        self.short_symbol      = ""
        return result

    # ── Main execute ──────────────────────────────────────────────────────────

    def execute(self, signal: str, price: float, atr: float, symbol: str) -> str:
        atr = max(atr, price * 0.001)  # floor: min 0.1% of price

        # ── Short position exit ───────────────────────────────────────────────
        if self.short_position > 0 and self.short_symbol == symbol:
            # Trailing stop for shorts moves DOWN as price falls, never up
            new_trail = price + config.TRAIL_STOP_ATR_MULT * atr
            if new_trail < self.short_trail_stop:
                self.short_trail_stop = new_trail

            hard_stop      = self.short_entry_price * (1 + config.STOP_LOSS_PCT)
            effective_stop = min(hard_stop, self.short_trail_stop)
            stop_type      = "SHORT-TRAIL" if self.short_trail_stop < hard_stop else "SHORT-STOP"

            if price >= effective_stop:
                return self._close_short(price, stop_type)
            if price <= self.short_take_profit:
                return self._close_short(price, "SHORT-TP")
            if signal == "COVER":
                return self._close_short(price, "ML-COVER")
            return "HOLD — monitoring short"

        # ── Long position exit ────────────────────────────────────────────────
        if self.position > 0 and self.position_symbol == symbol:
            # Update trailing stop: only moves up, never down
            new_trail = price - config.TRAIL_STOP_ATR_MULT * atr
            if new_trail > self.trail_stop_price:
                self.trail_stop_price = new_trail

            hard_stop      = self.entry_price * (1 - config.STOP_LOSS_PCT)
            effective_stop = max(hard_stop, self.trail_stop_price)
            stop_type      = "TRAIL-STOP" if self.trail_stop_price > hard_stop else "STOP-LOSS"

            if price <= effective_stop:
                return self._close_position(price, stop_type)
            if price >= self.take_profit_price:
                return self._close_position(price, "TAKE-PROFIT")
            if signal == "SELL":
                return self._close_position(price, "ML-SELL")
            return "HOLD — monitoring position"

        # ── Short entry ───────────────────────────────────────────────────────
        if signal == "SHORT" and self.short_position == 0 and self.position == 0 and self.balance > 10:
            portfolio_val  = self.portfolio_value(price)
            risk_amount    = portfolio_val * config.RISK_PER_TRADE
            # When hard stop (STOP_LOSS_PCT) is wider than ATR-based stop, TP must
            # scale with it — otherwise R:R collapses below 1:1 on low-ATR pairs.
            stop_distance  = max(config.TRAIL_STOP_ATR_MULT * atr, price * config.STOP_LOSS_PCT)
            tp_distance    = max(config.TAKE_PROFIT_ATR_MULT * atr, stop_distance * config.MIN_RR_RATIO)
            rr             = tp_distance / stop_distance
            if rr < config.MIN_RR_RATIO:
                return f"HOLD — SHORT R:R {rr:.2f}:1 below min {config.MIN_RR_RATIO}"
            position_units = risk_amount / stop_distance
            notional       = position_units * price
            notional       = min(notional, self.balance * config.MAX_POSITION_PCT)
            position_units = notional / price

            self.short_position    = position_units
            self.balance          += notional          # receive proceeds from short sale
            self.short_entry_price = price
            self.short_take_profit = price - tp_distance
            self.short_trail_stop  = price + config.TRAIL_STOP_ATR_MULT * atr
            self.short_symbol      = symbol
            self.trades           += 1

            return (
                f"SHORTED {position_units:.6f} {symbol} @ ${price:,.2f} | "
                f"TP: ${self.short_take_profit:,.2f} | "
                f"SL: ${self.short_trail_stop:,.2f} | "
                f"R:R: {rr:.2f} | "
                f"Risk: ${risk_amount:.2f}"
            )

        # ── Long entry ────────────────────────────────────────────────────────
        if signal == "BUY" and self.position == 0 and self.balance > 10:
            portfolio_val  = self.portfolio_value(price)
            risk_amount    = portfolio_val * config.RISK_PER_TRADE
            # When hard stop (STOP_LOSS_PCT) is wider than ATR-based stop, TP must
            # scale with it — otherwise R:R collapses below 1:1 on low-ATR pairs.
            stop_distance  = max(config.TRAIL_STOP_ATR_MULT * atr, price * config.STOP_LOSS_PCT)
            tp_distance    = max(config.TAKE_PROFIT_ATR_MULT * atr, stop_distance * config.MIN_RR_RATIO)
            rr             = tp_distance / stop_distance
            if rr < config.MIN_RR_RATIO:
                return f"HOLD — BUY R:R {rr:.2f}:1 below min {config.MIN_RR_RATIO}"
            position_units = risk_amount / stop_distance
            notional       = position_units * price
            notional       = min(notional, self.balance * config.MAX_POSITION_PCT)
            position_units = notional / price

            self.position           = position_units
            self.balance           -= notional
            self.entry_price        = price
            self.take_profit_price  = price + tp_distance
            self.trail_stop_price   = price - config.TRAIL_STOP_ATR_MULT * atr
            self.position_symbol    = symbol
            self.trades            += 1

            return (
                f"BOUGHT {position_units:.6f} {symbol} @ ${price:,.2f} | "
                f"TP: ${self.take_profit_price:,.2f} | "
                f"SL: ${self.trail_stop_price:,.2f} | "
                f"R:R: {rr:.2f} | "
                f"Risk: ${risk_amount:.2f}"
            )

        return "HOLD — no action"

    def portfolio_value(self, price: float) -> float:
        # balance already contains short sale proceeds; subtract liability to cover
        return self.balance + self.position * price - self.short_position * price

    def log_trade(self, timestamp, symbol, signal, price, buy_prob, action):
        portfolio_val = self.portfolio_value(price)
        win_rate      = (self.wins / max(self.trades, 1)) * 100
        row = {
            "timestamp":         timestamp,
            "symbol":            symbol,
            "price":             round(price, 4),
            "buy_prob":          round(buy_prob, 4),
            "signal":            signal,
            "action":            action,
            "balance_usdt":      round(self.balance, 2),
            "position_units":    round(self.position, 8),
            "position_symbol":   self.position_symbol,
            "entry_price":       round(self.entry_price, 4),
            "take_profit_price": round(self.take_profit_price, 4),
            "trail_stop_price":  round(self.trail_stop_price, 4),
            "short_units":       round(self.short_position, 8),
            "short_symbol":      self.short_symbol,
            "short_entry_price": round(self.short_entry_price, 4),
            "short_take_profit": round(self.short_take_profit, 4),
            "short_trail_stop":  round(self.short_trail_stop, 4),
            "portfolio_usd":     round(portfolio_val, 2),
            "total_trades":      self.trades,
            "total_wins":        self.wins,
            "win_rate_pct":      round(win_rate, 1),
        }
        file_exists = os.path.exists(self._log_file)
        with open(self._log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


# ─── Multi-pair scanner ───────────────────────────────────────────────────────

def _scan_symbols() -> list[tuple]:
    """
    Scan all configured symbols. Return list of
    (symbol, buy_prob, short_prob, price, atr, df) sorted by buy_prob descending.
    """
    results = []
    for sym in config.SYMBOLS:
        try:
            df, htf_df = fetch_latest_data(sym)
            df         = add_features(df, htf_df)
            model, fcols = load_model(sym)

            for c in fcols:
                if c not in df.columns:
                    df[c] = np.nan  # XGBoost treats NaN as missing — acceptable fallback

            buy_prob  = float(model.predict_proba(df[fcols].iloc[[-1]])[0][1])

            short_prob = 0.0
            if config.ENABLE_SHORTING:
                short_result = load_short_model(sym)
                if short_result:
                    sm, sfcols = short_result
                    for c in sfcols:
                        if c not in df.columns:
                            df[c] = np.nan
                    short_prob = float(sm.predict_proba(df[sfcols].iloc[[-1]])[0][1])

            price = float(df["close"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])
            results.append((sym, buy_prob, short_prob, price, atr, df))
        except Exception as e:
            print(f"[scanner] {sym}: {e}")
    return sorted(results, key=lambda x: x[1], reverse=True)


# ─── Main loop ────────────────────────────────────────────────────────────────

paper_trader = PaperTrader()


def run_bot():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'─'*55}")
    print(f"[bot] Running at {now}")

    try:
        # ── Phase 1: Monitor open SHORT position (exit checks) ────────────────
        if paper_trader.short_position > 0:
            sym = paper_trader.short_symbol
            df, htf_df = fetch_latest_data(sym)
            df    = add_features(df, htf_df)
            price = float(df["close"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])

            short_prob = 0.0
            short_result = load_short_model(sym)
            if short_result:
                sm, sfcols = short_result
                for c in sfcols:
                    if c not in df.columns:
                        df[c] = np.nan
                short_prob = float(sm.predict_proba(df[sfcols].iloc[[-1]])[0][1])

            signal = "COVER" if short_prob <= config.COVER_THRESHOLD else "HOLD"
            action = paper_trader.execute(signal, price, atr, sym)
            portfolio_val = paper_trader.portfolio_value(price)
            total_return  = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100
            win_rate      = (paper_trader.wins / max(paper_trader.trades, 1)) * 100

            print(f"[bot] {sym} ${price:,.4f} | short_prob={short_prob:.1%} | {signal} → {action}")
            print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%) | WR: {win_rate:.0f}%")

            msg = (
                f"*{sym} Short Monitor — {now}*\n"
                f"Price: ${price:,.4f} | ML: {short_prob:.1%}\n"
                f"Action: {action}\n"
                f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
                f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
            )
            send_telegram(msg)
            paper_trader.log_trade(now, sym, signal, price, short_prob, action)
            return

        # ── Phase 2: Monitor open LONG position (exit checks) ─────────────────
        if paper_trader.position > 0:
            sym  = paper_trader.position_symbol
            df, htf_df = fetch_latest_data(sym)
            df    = add_features(df, htf_df)
            price = float(df["close"].iloc[-1])
            atr   = float(df["atr"].iloc[-1])

            model, fcols = load_model(sym)
            for c in fcols:
                if c not in df.columns:
                    df[c] = np.nan
            buy_prob = float(model.predict_proba(df[fcols].iloc[[-1]])[0][1])

            signal = "SELL" if buy_prob <= config.SELL_THRESHOLD else "HOLD"
            action = paper_trader.execute(signal, price, atr, sym)
            portfolio_val = paper_trader.portfolio_value(price)
            total_return  = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100
            win_rate      = (paper_trader.wins / max(paper_trader.trades, 1)) * 100

            print(f"[bot] {sym} ${price:,.4f} | prob={buy_prob:.1%} | {signal} → {action}")
            print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%) | "
                  f"WR: {win_rate:.0f}%")

            msg = (
                f"*{sym} Monitor — {now}*\n"
                f"Price: ${price:,.4f} | ML: {buy_prob:.1%}\n"
                f"Action: {action}\n"
                f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
                f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
            )
            send_telegram(msg)
            paper_trader.log_trade(now, sym, signal, price, buy_prob, action)
            return

        # ── Phase 3: Scan for best entry opportunity (LONG or SHORT) ──────────
        candidates = _scan_symbols()
        if not candidates:
            print("[bot] No scan results.")
            return

        # Best BUY: highest buy_prob (already sorted)
        best_sym, best_prob, _, best_price, best_atr, best_df = candidates[0]
        # Best SHORT: highest short_prob across all candidates
        best_short = max(candidates, key=lambda x: x[2])
        sh_sym, _, sh_prob, sh_price, sh_atr, sh_df = best_short

        trend_tag  = "↑" if _regime_ok(best_df) else "↓regime"
        print(f"[bot] Best BUY:   {best_sym} prob={best_prob:.1%} ${best_price:,.4f} {trend_tag}")
        for sym, prob, s_prob, price, _, _ in candidates[1:]:
            s_tag = f" | short={s_prob:.1%}" if config.ENABLE_SHORTING else ""
            print(f"      {sym} buy={prob:.1%} ${price:,.4f}{s_tag}")
        if config.ENABLE_SHORTING:
            sh_tag = "↓" if _bearish_regime_ok(sh_df) else "↑regime"
            print(f"[bot] Best SHORT: {sh_sym} prob={sh_prob:.1%} ${sh_price:,.4f} {sh_tag}")

        signal     = "HOLD"
        action_sym = best_sym
        action_price, action_atr = best_price, best_atr
        exec_prob  = best_prob

        if best_prob >= config.BUY_THRESHOLD and _regime_ok(best_df):
            signal = "BUY"
        elif config.ENABLE_SHORTING and sh_prob >= config.SHORT_THRESHOLD and _bearish_regime_ok(sh_df):
            signal = "SHORT"
            action_sym, action_price, action_atr = sh_sym, sh_price, sh_atr
            exec_prob = sh_prob

        action = paper_trader.execute(signal, action_price, action_atr, action_sym)
        portfolio_val = paper_trader.portfolio_value(action_price)
        total_return  = (portfolio_val / config.PAPER_STARTING_BALANCE - 1) * 100
        win_rate      = (paper_trader.wins / max(paper_trader.trades, 1)) * 100

        print(f"[bot] Signal: {signal} → {action}")
        print(f"[bot] Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%) | WR: {win_rate:.0f}%")

        msg = (
            f"*{action_sym} Signal — {now}*\n"
            f"Price: ${action_price:,.4f}\n"
            f"ML signal: *{signal}* ({exec_prob:.1%}) {trend_tag}\n"
            f"Action: {action}\n"
            f"Portfolio: ${portfolio_val:,.2f} ({total_return:+.2f}%)\n"
            f"Trades: {paper_trader.trades} | Win rate: {win_rate:.0f}%"
        )
        send_telegram(msg)
        paper_trader.log_trade(now, action_sym, signal, action_price, exec_prob, action)

    except Exception as e:
        err = f"[bot] ERROR: {e}"
        print(err)
        send_telegram(f"Bot error at {now}:\n{err}")


# ─── Scheduler ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(" Crypto ML Bot v3 — Multi-pair, Regime-aware")
    print(f" Pairs:    {', '.join(config.SYMBOLS)}")
    print(f" Interval: {config.INTERVAL}")
    print(f" Balance:  ${config.PAPER_STARTING_BALANCE:,.0f} (paper)")
    print(f" BUY threshold: {config.BUY_THRESHOLD:.0%} | Risk/trade: {config.RISK_PER_TRADE:.0%}")
    if config.ENABLE_SHORTING:
        print(f" SHORT threshold: {config.SHORT_THRESHOLD:.0%} | Shorting: ENABLED")
    print("=" * 55)

    run_bot()

    scheduler = BlockingScheduler()
    interval_map = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "1d": 1440,
    }
    minutes = interval_map.get(config.INTERVAL, 60)
    scheduler.add_job(run_bot, "interval", minutes=minutes)

    print(f"\n[bot] Scheduler running every {minutes} min. Ctrl+C to stop.\n")
    send_telegram(
        f"Bot v3 started\n"
        f"Pairs: {', '.join(config.SYMBOLS)}\n"
        f"Interval: {config.INTERVAL} | BUY≥{config.BUY_THRESHOLD:.0%}"
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[bot] Stopped.")
        send_telegram("Bot stopped.")
