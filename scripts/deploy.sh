#!/bin/bash

# FRC RAG Server Deployment Script
# Sets up and runs the FRC RAG server with Ollama on a VM

set -e  # Exit on any error

echo "🚀 Starting FRC RAG Server Deployment..."

# Configuration
OLLAMA_HOST="0.0.0.0"
OLLAMA_PORT="11434"
SERVER_PORT="5000"
MODELS_TO_INSTALL=("mistral" "llama2")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (for system-level installations)
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is fine for VM deployment."
    else
        print_status "Running as regular user."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Detect OS
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y curl wget git python3 python3-pip python3-venv \
                                tesseract-ocr libtesseract-dev poppler-utils
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum update -y
        sudo yum install -y curl wget git python3 python3-pip \
                           tesseract tesseract-devel poppler-utils
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -Syu --noconfirm curl wget git python python-pip \
                                     tesseract poppler
    else
        print_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
    
    print_success "System dependencies installed"
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        print_warning "Ollama already installed, checking version..."
        ollama --version
    else
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed"
    fi
}

# Start Ollama service
start_ollama() {
    print_status "Starting Ollama service..."
    
    # Kill any existing Ollama processes
    pkill -f "ollama serve" || true
    
    # Start Ollama in background
    OLLAMA_HOST=$OLLAMA_HOST OLLAMA_PORT=$OLLAMA_PORT nohup ollama serve > logs/ollama.log 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    print_status "Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; then
            print_success "Ollama service started (PID: $OLLAMA_PID)"
            return 0
        fi
        sleep 2
    done
    
    print_error "Ollama failed to start within 60 seconds"
    return 1
}

# Install AI models
install_models() {
    print_status "Installing AI models..."
    
    for model in "${MODELS_TO_INSTALL[@]}"; do
        print_status "Installing model: $model"
        if ollama pull "$model"; then
            print_success "Model $model installed successfully"
        else
            print_warning "Failed to install model $model"
        fi
    done
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found!"
        return 1
    fi
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    
    if [ -f "src/utils/database_setup.py" ]; then
        source venv/bin/activate
        python src/utils/database_setup.py
        print_success "Database setup completed"
    else
        print_warning "Database setup script not found. Database may need manual setup."
    fi
}

# Configure firewall (optional)
configure_firewall() {
    print_status "Configuring firewall..."
    
    # Check if UFW is available
    if command -v ufw &> /dev/null; then
        print_status "Configuring UFW firewall..."
        sudo ufw allow $SERVER_PORT
        sudo ufw allow $OLLAMA_PORT
        print_success "Firewall configured for ports $SERVER_PORT and $OLLAMA_PORT"
    else
        print_warning "UFW not available. Please configure firewall manually."
        print_warning "Required ports: $SERVER_PORT (Flask), $OLLAMA_PORT (Ollama)"
    fi
}

# Create systemd service files
create_systemd_services() {
    print_status "Creating systemd service files..."
    
    # Create logs directory
    mkdir -p logs
    
    # Ollama service
    sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama API Server
After=network.target

[Service]
Type=simple
User=$USER
Environment=OLLAMA_HOST=$OLLAMA_HOST
Environment=OLLAMA_PORT=$OLLAMA_PORT
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
StandardOutput=append:$(pwd)/logs/ollama.log
StandardError=append:$(pwd)/logs/ollama.log

[Install]
WantedBy=multi-user.target
EOF

    # FRC RAG service
    CURRENT_DIR=$(pwd)
    sudo tee /etc/systemd/system/frc-rag.service > /dev/null <<EOF
[Unit]
Description=FRC RAG Server
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$CURRENT_DIR
Environment=PATH=$CURRENT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=ENVIRONMENT=production
ExecStart=$CURRENT_DIR/venv/bin/python server.py
Restart=always
RestartSec=3
StandardOutput=append:$CURRENT_DIR/logs/server.log
StandardError=append:$CURRENT_DIR/logs/server.log

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    sudo systemctl daemon-reload
    
    print_success "Systemd services created"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Enable and start Ollama
    sudo systemctl enable ollama
    sudo systemctl start ollama
    
    # Wait for Ollama to be ready
    sleep 5
    
    # Enable and start FRC RAG
    sudo systemctl enable frc-rag
    sudo systemctl start frc-rag
    
    print_success "Services started"
}

# Install tunneling tools (optional)
install_tunnel_tools() {
    print_status "Installing tunneling tools (optional)..."
    
    # Install ngrok
    if ! command -v ngrok &> /dev/null; then
        print_status "Installing ngrok..."
        curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
        echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
        sudo apt update && sudo apt install ngrok
        print_success "ngrok installed"
    fi
    
    # Install cloudflared
    if ! command -v cloudflared &> /dev/null; then
        print_status "Installing cloudflared..."
        curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
        sudo dpkg -i cloudflared.deb
        rm cloudflared.deb
        print_success "cloudflared installed"
    fi
}

# Display server information
show_server_info() {
    print_success "🎉 FRC RAG Server deployment completed!"
    echo ""
    echo "=== Server Information ==="
    echo "Server URL: http://$(hostname -I | awk '{print $1}'):$SERVER_PORT"
    echo "Ollama API: http://$(hostname -I | awk '{print $1}'):$OLLAMA_PORT"
    echo ""
    echo "=== Service Management ==="
    echo "Check FRC RAG status: sudo systemctl status frc-rag"
    echo "Check Ollama status:  sudo systemctl status ollama"
    echo "View FRC RAG logs:    sudo journalctl -u frc-rag -f"
    echo "View Ollama logs:     sudo journalctl -u ollama -f"
    echo ""
    echo "=== API Endpoints ==="
    echo "Health Check:   GET  /health"
    echo "Query:          POST /api/query"
    echo "Ollama Proxy:   *    /api/ollama/*"
    echo "Stats:          GET  /api/stats"
    echo "Tunnel Control: POST /api/tunnel"
    echo ""
    echo "=== Configuration ==="
    echo "Edit .env file to customize settings"
    echo "Default rate limit: 60 requests per minute"
    echo "For tunneling: Configure TUNNEL_SERVICE in .env"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Configure your .env file with appropriate settings"
    echo "2. Set up API keys if needed for production"
    echo "3. Configure tunneling for remote access"
    echo "4. Test the deployment with: curl http://localhost:$SERVER_PORT/health"
}

# Main deployment function
main() {
    print_status "Starting FRC RAG Server deployment on VM..."
    
    check_privileges
    install_system_deps
    install_ollama
    setup_python_env
    start_ollama
    install_models
    setup_database
    configure_firewall
    install_tunnel_tools
    create_systemd_services
    start_services
    show_server_info
}

# Run deployment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi