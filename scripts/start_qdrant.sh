#!/bin/bash
# Start Qdrant vector database
# CPU Core: 3

set -e

echo "[Qdrant] Starting Qdrant vector database..."

# Check if qdrant binary exists
if ! command -v qdrant &> /dev/null; then
    echo "[Qdrant] ERROR: qdrant binary not found in PATH"
    echo "[Qdrant] Please install Qdrant: https://github.com/qdrant/qdrant/releases"
    exit 1
fi

# Storage path
STORAGE_PATH="${QDRANT_STORAGE_PATH:-./qdrant_storage}"
mkdir -p "$STORAGE_PATH"

echo "[Qdrant] Storage path: $STORAGE_PATH"
echo "[Qdrant] Pinned to CPU core 3"

# Run with restart loop
while true; do
    taskset -c 3 qdrant --storage-path "$STORAGE_PATH" 2>&1 | tee -a logs/qdrant.log
    
    echo "[Qdrant] Process exited. Restarting in 2 seconds..."
    sleep 2
done
