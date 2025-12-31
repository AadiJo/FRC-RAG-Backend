#!/bin/bash
# Start FastAPI backend with Uvicorn
# CPU Cores: 0, 1

set -e

cd "$(dirname "$0")/.."

echo "[API] Starting FastAPI backend..."

# Configuration
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-3}"
LOG_LEVEL="${API_LOG_LEVEL:-info}"

echo "[API] Host: $HOST:$PORT"
echo "[API] Workers: $WORKERS"
echo "[API] Log level: $LOG_LEVEL"
echo "[API] Pinned to CPU cores 0,1"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[API] Activated virtualenv"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[API] Activated .venv"
fi

# Ensure logs directory exists
mkdir -p logs

# Run with restart loop
while true; do
    taskset -c 0,1 python -m uvicorn src.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        2>&1 | tee -a logs/api.log
    
    echo "[API] Process exited. Restarting in 2 seconds..."
    sleep 2
done
