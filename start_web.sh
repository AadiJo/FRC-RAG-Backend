#!/bin/bash

# FRC RAG Chat Interface Launcher
echo "🚀 Starting FRC RAG Chat Interface..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📚 Installing/updating dependencies..."
pip install -r requirements.txt

# Check if database exists
if [ ! -d "chroma_enhanced" ] && [ ! -d "db" ]; then
    echo "⚠️  No database found! You need to create the database first."
    echo "   Run: python create_database.py"
    echo "   or:  python create_database2.py (for enhanced version)"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Exiting. Please create the database first."
        exit 1
    fi
fi

# Check if Ollama is running
echo "🤖 Checking Ollama service..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Please install Ollama from https://ollama.ai/"
    echo "   After installation, run: ollama pull mistral"
else
    # Try to check if Ollama is running by testing the API
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "⚠️  Ollama service not running. Starting Ollama..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
    fi
    
    # Check if mistral model is available
    if ! ollama list | grep -q "mistral"; then
        echo "📥 Mistral model not found. Downloading..."
        ollama pull mistral
    fi
fi

echo ""
echo "✅ Setup complete! Starting the web interface..."
echo "📱 Open your browser to: http://localhost:5000"
echo "💬 Chat interface will be ready in a few seconds..."
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"

# Start the Flask application
python app.py
