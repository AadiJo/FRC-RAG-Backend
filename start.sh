#!/bin/bash

# FRC RAG Backend Server Startup Script
# For local development and testing

echo "🤖 Starting FRC RAG Backend Server..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env with your configuration."
    echo ""
fi

# Create logs directory
mkdir -p logs

# Detect model provider from .env (default: local)
MODEL_PROVIDER_VALUE="local"
if [ -f ".env" ]; then
    MODEL_PROVIDER_VALUE=$(grep -E '^MODEL_PROVIDER=' .env | tail -n 1 | cut -d '=' -f 2- | tr -d '\r' | tr -d '"' | tr -d "'")
fi
if [ "$MODEL_PROVIDER_VALUE" = "chute" ]; then
    MODEL_PROVIDER_VALUE="openrouter"
fi

# Check if Ollama is running (only needed for local provider)
if [ "$MODEL_PROVIDER_VALUE" = "local" ] || [ -z "$MODEL_PROVIDER_VALUE" ]; then
    echo "🤖 Checking Ollama service..."
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "⚠️  Ollama is not running. Please start Ollama:"
        echo "   ollama serve"
        echo ""
        echo "And install required models:"
        echo "   ollama pull mistral"
        echo ""
        read -p "Press Enter when Ollama is ready..."
    fi
else
    echo "🤖 Model provider is '$MODEL_PROVIDER_VALUE' — skipping Ollama check."
fi

# Check if database exists
if [ ! -d "db" ] || [ -z "$(ls -A db 2>/dev/null)" ]; then
    echo "⚠️  Database not found. You may need to run:"
    echo "   python src/utils/database_setup.py"
    echo ""
fi

echo ""
echo "================================"
echo "🚀 Starting Backend Server"
echo "================================"
echo ""

# Start the server
python app.py
