#!/bin/bash
# FraudGuard — Railway Startup Script
# Railway runs this as the START command.
# It handles: data generation + model training + server launch in one shot.
set -e

echo "========================================"
echo "  FraudGuard — Railway Startup"
echo "========================================"

# Step 1: Generate dataset (if not cached)
if [ ! -f "data/creditcard.csv" ]; then
  echo "[1/3] Generating dataset..."
  python3 data/generate_data.py
else
  echo "[1/3] Dataset already exists, skipping."
fi

# Step 2: Train models (if not cached)
if [ ! -f "static/analysis.json" ] || [ ! -f "models/best_model.pkl" ]; then
  echo "[2/3] Training models..."
  python3 analysis/engine.py
else
  echo "[2/3] Models already trained, skipping."
fi

# Step 3: Start gunicorn
echo "[3/3] Starting gunicorn on port ${PORT:-8000}..."
exec gunicorn app:app \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers 1 \
  --timeout 180 \
  --preload
