#!/bin/bash

# Development server startup script
# For local development and testing

echo "🤖 Starting FRC RAG Development Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    echo "Please edit .env file with your configuration."
fi

# Set development environment
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=DEBUG

# Create logs directory
mkdir -p logs

# Check if Ollama is running
echo "🤖 Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Please start Ollama in another terminal:"
    echo "   ollama serve"
    echo ""
    echo "And install required models:"
    echo "   ollama pull mistral"
    echo ""
    read -p "Press Enter when Ollama is ready..."
fi

# Check if database exists
if [ ! -d "db" ]; then
    echo "🗄️  Database not found. Setting up database..."
    if [ -f "src/utils/database_setup.py" ]; then
        python src/utils/database_setup.py
    else
        echo "⚠️  Database setup script not found. Please set up the database manually."
    fi
fi

echo "🚀 Starting development server..."
echo "📝 Server will be available at: http://localhost:5000"
echo "📊 Health check: http://localhost:5000/health"
echo "📋 Press Ctrl+C to stop the server"
echo ""

# Start the server
python server.py