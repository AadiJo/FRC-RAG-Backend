#!/bin/bash

# Tunnel Setup Script
# Helps configure tunneling for remote access

echo "🌐 FRC RAG Tunnel Setup"
echo "======================"

# Function to setup ngrok
setup_ngrok() {
    echo ""
    echo "Setting up ngrok tunnel..."
    echo ""
    
    # Check if ngrok is installed
    if ! command -v ngrok &> /dev/null; then
        echo "Installing ngrok..."
        
        # Detect OS and install
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
            echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
            sudo apt update && sudo apt install ngrok
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install ngrok/ngrok/ngrok
        else
            echo "Please install ngrok manually from https://ngrok.com/download"
            exit 1
        fi
    fi
    
    echo ""
    echo "📝 To complete ngrok setup:"
    echo "1. Sign up at https://dashboard.ngrok.com/signup"
    echo "2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "3. Add to your .env file:"
    echo "   TUNNEL_SERVICE=ngrok"
    echo "   NGROK_AUTH_TOKEN=your_token_here"
    echo ""
}

# Function to setup Cloudflare tunnel
setup_cloudflare() {
    echo ""
    echo "Setting up Cloudflare tunnel..."
    echo ""
    
    # Check if cloudflared is installed
    if ! command -v cloudflared &> /dev/null; then
        echo "Installing cloudflared..."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
            sudo dpkg -i cloudflared.deb
            rm cloudflared.deb
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install cloudflare/cloudflare/cloudflared
        else
            echo "Please install cloudflared manually from https://github.com/cloudflare/cloudflared"
            exit 1
        fi
    fi
    
    echo ""
    echo "📝 To complete Cloudflare tunnel setup:"
    echo "1. Log in to Cloudflare dashboard"
    echo "2. Go to Zero Trust > Networks > Tunnels"
    echo "3. Create a new tunnel and get the token"
    echo "4. Add to your .env file:"
    echo "   TUNNEL_SERVICE=cloudflare"
    echo "   CLOUDFLARE_TUNNEL_TOKEN=your_token_here"
    echo ""
}

# Main menu
echo "Choose a tunneling service:"
echo "1) ngrok (free tier available)"
echo "2) Cloudflare Tunnel (free)"
echo "3) Skip tunnel setup"
echo ""

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        setup_ngrok
        ;;
    2)
        setup_cloudflare
        ;;
    3)
        echo "Skipping tunnel setup. You can run this script again later."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure your .env file with the tunnel settings"
echo "2. Start the server with: ./start.sh"
echo "3. The tunnel will automatically start when the server starts"
echo ""
echo "For manual tunnel control, use the API:"
echo "  curl -X POST http://localhost:5000/api/tunnel -d '{\"action\":\"start\"}'"
echo "  curl -X POST http://localhost:5000/api/tunnel -d '{\"action\":\"status\"}'"
echo "  curl -X POST http://localhost:5000/api/tunnel -d '{\"action\":\"stop\"}'"