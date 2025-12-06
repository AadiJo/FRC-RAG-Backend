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

# Check if Ollama is running
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
