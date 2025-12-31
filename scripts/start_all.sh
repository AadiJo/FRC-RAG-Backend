#!/bin/bash
# Start all services in screen sessions
# This script creates and manages all necessary screen sessions

set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "FRC RAG Backend - Multi-Worker Startup"
echo "========================================"
echo ""
echo "CPU Core Allocation:"
echo "  Core 0,1: FastAPI workers"
echo "  Core 2:   TEI (embeddings)"
echo "  Core 3:   Qdrant"
echo ""

# Create logs directory
mkdir -p logs

# Check for existing screens
existing_screens=$(screen -ls 2>/dev/null | grep -E "qdrant|tei|api" || true)
if [ -n "$existing_screens" ]; then
    echo "WARNING: Existing screen sessions found:"
    echo "$existing_screens"
    echo ""
    read -p "Kill existing sessions and restart? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        screen -S qdrant -X quit 2>/dev/null || true
        screen -S tei -X quit 2>/dev/null || true
        screen -S api -X quit 2>/dev/null || true
        sleep 1
    else
        echo "Aborting. Use 'screen -r <name>' to attach to existing sessions."
        exit 1
    fi
fi

echo "Starting services..."

# Start Qdrant
echo "[1/3] Starting Qdrant (Core 3)..."
screen -dmS qdrant bash -c './scripts/start_qdrant.sh; exec bash'
sleep 3

# Start TEI
echo "[2/3] Starting TEI (Core 2)..."
screen -dmS tei bash -c './scripts/start_tei.sh; exec bash'
sleep 5  # TEI needs time to load model

# Start API
echo "[3/3] Starting API (Cores 0,1)..."
screen -dmS api bash -c './scripts/start_api.sh; exec bash'

echo ""
echo "========================================"
echo "All services started!"
echo "========================================"
echo ""
echo "Screen sessions:"
screen -ls | grep -E "qdrant|tei|api" || echo "  (no sessions found - check for errors)"
echo ""
echo "Commands:"
echo "  Attach to session:  screen -r <name>"
echo "  Detach from screen: Ctrl+A D"
echo "  List sessions:      screen -ls"
echo "  Stop all:           ./scripts/stop_all.sh"
echo ""
echo "Health checks:"
echo "  Qdrant:  curl http://127.0.0.1:6333/health"
echo "  TEI:     curl http://127.0.0.1:8080/health"
echo "  API:     curl http://127.0.0.1:8000/api/v1/health"
echo ""
