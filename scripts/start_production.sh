#!/bin/bash

# Production server startup script
# Uses gunicorn for production deployment

# Set environment variables
export ENVIRONMENT=production
export SERVER_HOST=0.0.0.0
export SERVER_PORT=${SERVER_PORT:-5000}
export OLLAMA_HOST=${OLLAMA_HOST:-localhost}
export OLLAMA_PORT=${OLLAMA_PORT:-11434}

# Gunicorn settings
WORKERS=${WORKERS:-4}                    # Number of worker processes (usually 2x CPU cores)
WORKER_CLASS=${WORKER_CLASS:-gevent}     # Async worker class for better concurrency
WORKER_CONNECTIONS=${WORKER_CONNECTIONS:-1000}     # Max simultaneous clients per worker
MAX_REQUESTS=${MAX_REQUESTS:-1000}       # Restart workers after this many requests
MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-100}     # Add randomness to prevent thundering herd
TIMEOUT=${TIMEOUT:-30}                   # Worker timeout in seconds
KEEPALIVE=${KEEPALIVE:-5}                # Keep-alive timeout

# Logging
ACCESS_LOG=${ACCESS_LOG:-logs/access.log}
ERROR_LOG=${ERROR_LOG:-logs/error.log}
LOG_LEVEL=${LOG_LEVEL:-info}

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting FRC RAG Server with Gunicorn..."
echo "Workers: $WORKERS"
echo "Worker class: $WORKER_CLASS"
echo "Listening on: $SERVER_HOST:$SERVER_PORT"

# Check if gunicorn is available
if ! command -v gunicorn &> /dev/null; then
    echo "Gunicorn not found. Installing..."
    pip install gunicorn gevent
fi

# Start Gunicorn
exec gunicorn \
    --bind $SERVER_HOST:$SERVER_PORT \
    --workers $WORKERS \
    --worker-class $WORKER_CLASS \
    --worker-connections $WORKER_CONNECTIONS \
    --max-requests $MAX_REQUESTS \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --timeout $TIMEOUT \
    --keep-alive $KEEPALIVE \
    --access-logfile $ACCESS_LOG \
    --error-logfile $ERROR_LOG \
    --log-level $LOG_LEVEL \
    --preload \
    --enable-stdio-inheritance \
    --pid logs/gunicorn.pid \
    server:app