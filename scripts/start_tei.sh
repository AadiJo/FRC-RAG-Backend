#!/bin/bash
# Start Text Embeddings Inference (TEI) service
# CPU Core: 2

set -e

echo "[TEI] Starting Text Embeddings Inference..."

# Configuration
MODEL="${TEI_MODEL:-BAAI/bge-large-en-v1.5}"
PORT="${TEI_PORT:-8080}"
MAX_BATCH_TOKENS="${TEI_MAX_BATCH_TOKENS:-8192}"

echo "[TEI] Model: $MODEL"
echo "[TEI] Port: $PORT"
echo "[TEI] Max batch tokens: $MAX_BATCH_TOKENS"
echo "[TEI] Pinned to CPU core 2"

# Check if text-embeddings-inference is installed
if command -v text-embeddings-inference &> /dev/null; then
    # Rust binary available
    while true; do
        taskset -c 2 text-embeddings-inference \
            --model-id "$MODEL" \
            --port "$PORT" \
            --max-batch-tokens "$MAX_BATCH_TOKENS" \
            2>&1 | tee -a logs/tei.log
        
        echo "[TEI] Process exited. Restarting in 2 seconds..."
        sleep 2
    done
elif command -v text-embeddings-router &> /dev/null; then
    # Docker-style binary
    while true; do
        taskset -c 2 text-embeddings-router \
            --model-id "$MODEL" \
            --port "$PORT" \
            --max-batch-tokens "$MAX_BATCH_TOKENS" \
            2>&1 | tee -a logs/tei.log
        
        echo "[TEI] Process exited. Restarting in 2 seconds..."
        sleep 2
    done
else
    echo "[TEI] ERROR: text-embeddings-inference not found"
    echo "[TEI] Install options:"
    echo "  1. Rust binary: cargo install text-embeddings-inference"
    echo "  2. Download from: https://github.com/huggingface/text-embeddings-inference/releases"
    echo "  3. Docker: docker run ghcr.io/huggingface/text-embeddings-inference:cpu-1.0"
    exit 1
fi
