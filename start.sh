#!/bin/bash
set -e
python3 data/generate_data.py
python3 analysis/engine.py
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 180 --preload
