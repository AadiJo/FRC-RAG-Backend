#!/bin/bash

# VPS Setup Script for FRC RAG
# Run this after uploading the project folder to /opt/frc-rag

set -e

echo "🚀 Setting up FRC RAG on VPS..."

# Ensure we're in the right directory
cd /opt/frc-rag

# Update system
echo "📦 Installing system dependencies..."
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv nginx

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install gunicorn

# Copy VPS environment file
echo "⚙️ Configuring environment..."
cp .env.vps .env

# Create systemd service
echo "🔧 Creating system service..."
cat > /etc/systemd/system/frc-rag-vps.service << EOF
[Unit]
Description=FRC RAG VPS Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/frc-rag
Environment=PATH=/opt/frc-rag/venv/bin
ExecStart=/opt/frc-rag/venv/bin/gunicorn --bind 127.0.0.1:8000 --workers 2 --timeout 300 vps_full_server:app
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx
echo "🌐 Configuring web server..."
cat > /etc/nginx/sites-available/frc-rag << EOF
server {
    listen 80;
    server_name _;
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
    
    location /static/ {
        alias /opt/frc-rag/static/;
        expires 7d;
    }
}
EOF

# Enable nginx site
ln -sf /etc/nginx/sites-available/frc-rag /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Create log file
touch /var/log/frc-rag-vps.log

# Start services
echo "🎬 Starting services..."
systemctl daemon-reload
systemctl enable frc-rag-vps
systemctl start frc-rag-vps
systemctl enable nginx
systemctl restart nginx

# Configure firewall
if command -v ufw &> /dev/null; then
    echo "🔒 Configuring firewall..."
    ufw allow 'Nginx Full'
    ufw allow OpenSSH
    echo "y" | ufw enable
fi

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "your-vps-ip")

echo ""
echo "🎉 FRC RAG VPS Setup Complete!"
echo ""
echo "🌐 Frontend URL: http://$PUBLIC_IP"
echo "🔍 Health Check: http://$PUBLIC_IP/health"
echo "📊 API Endpoint: http://$PUBLIC_IP/api/query"
echo ""
echo "🔧 Management Commands:"
echo "• Check status: systemctl status frc-rag-vps"
echo "• View logs: tail -f /var/log/frc-rag-vps.log"
echo "• Restart service: systemctl restart frc-rag-vps"
echo ""
echo "⚠️  To update ngrok URL:"
echo "• Edit: nano /opt/frc-rag/.env"
echo "• Change: REMOTE_OLLAMA_URL=https://your-new-ngrok-url"
echo "• Restart: systemctl restart frc-rag-vps"