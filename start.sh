#!/bin/bash
# start.sh — Railway startup script
# Runs on every deploy: creates config, trains models if missing, starts bot.

set -e

echo "============================================"
echo " Bot startup — $(date)"
echo "============================================"

# ── Step 1: Create config.py from example (reads Railway env vars) ────────────
# config.py is git-ignored so Railway never has it.
# config.example.py uses os.environ.get() — env vars set in Railway UI are picked up.
if [ ! -f config.py ]; then
    cp config.example.py config.py
    echo "[setup] Created config.py from config.example.py"
fi

# ── Step 2: Validate required env vars are set ────────────────────────────────
python - <<'EOF'
import os, sys
required = ["BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]
missing = [k for k in required if not os.environ.get(k) or os.environ.get(k, "").startswith("YOUR_")]
if missing:
    print(f"[setup] MISSING ENV VARS: {', '.join(missing)}")
    print("[setup] Set these in Railway dashboard → Variables tab")
    sys.exit(1)
print("[setup] All required env vars present.")
EOF

# ── Step 3: Train models if missing (runs on first deploy, ~30 min) ───────────
python - <<'EOF'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
import config
from model_trainer import model_file_for, train_and_save

symbols = config.SYMBOLS
for sym in symbols:
    mf = model_file_for(sym)
    if not os.path.exists(mf):
        print(f"[setup] Model missing for {sym} — training now (this takes ~10 min per pair)...")
        try:
            train_and_save(sym, mf)
            print(f"[setup] {sym} model ready.")
        except Exception as e:
            print(f"[setup] WARNING: {sym} training failed ({e}) — skipping.")
    else:
        print(f"[setup] {sym} model found: {mf}")
EOF

# ── Step 3b: Train short models if ENABLE_SHORTING=true ──────────────────────
python - <<'EOF'
import os, sys
sys.path.insert(0, ".")
import config
if not config.ENABLE_SHORTING:
    print("[setup] Shorting disabled — skipping short model training.")
    sys.exit(0)
from model_trainer import short_model_file_for, train_and_save_short

for sym in config.SYMBOLS:
    mf = short_model_file_for(sym)
    if not os.path.exists(mf):
        print(f"[setup] Short model missing for {sym} — training now...")
        try:
            train_and_save_short(sym, mf)
            print(f"[setup] {sym} short model ready.")
        except Exception as e:
            print(f"[setup] WARNING: {sym} short training failed ({e}) — skipping.")
    else:
        print(f"[setup] {sym} short model found: {mf}")
EOF

# ── Step 4: Start the bot ─────────────────────────────────────────────────────
echo "[setup] All models ready. Starting bot..."
python bot.py
