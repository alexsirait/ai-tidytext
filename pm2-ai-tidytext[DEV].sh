#!/bin/bash

PORT="40010"
HOST="192.168.88.60"
APP_NAME="ai-tidytext"

# Definisikan variabel
APP_DIR="/var/lib/jenkins/workspace/$APP_NAME"
VENV_DIR="$APP_DIR/venv"
APP_FILE="main.py"  # Ganti ini kalau file utamamu bukan main.py
APP_MODULE="main:app"  # Format: nama_file:variabel_app (tanpa .py)

# Aktivasi virtual environment
source "$VENV_DIR/bin/activate"

# Pindah ke direktori aplikasi (opsional tapi biasanya aman)
cd "$APP_DIR"

# Jalankan server menggunakan uvicorn
exec uvicorn "$APP_MODULE" \
    --host "$HOST" \
    --port "$PORT" \
    --workers 4 \
    --timeout-keep-alive 30 \
    --log-level info