"""
Tunneling utilities for remote access to FRC RAG Server
Supports ngrok and Cloudflare tunnels
"""

import os
import subprocess
import time
import logging
import requests
from typing import Optional, Dict, Any

from .config import get_config

Config = get_config()
logger = logging.getLogger(__name__)

class TunnelManager:
    """Manages tunneling services for remote access"""
    
    def __init__(self):
        self.tunnel_process = None
        self.tunnel_url = None
        self.service = Config.TUNNEL_SERVICE.lower() if Config.TUNNEL_SERVICE else None
        
    def start_tunnel(self) -> Optional[str]:
        """Start tunnel service and return public URL"""
        if not self.service:
            logger.info("No tunnel service configured")
            return None
            
        if self.service == 'ngrok':
            return self._start_ngrok()
        elif self.service == 'cloudflare':
            return self._start_cloudflare()
        else:
            logger.error(f"Unsupported tunnel service: {self.service}")
            return None
    
    def _start_ngrok(self) -> Optional[str]:
        """Start ngrok tunnel"""
        if not Config.NGROK_AUTH_TOKEN:
            logger.error("NGROK_AUTH_TOKEN not configured")
            return None
            
        try:
            # Authenticate ngrok
            subprocess.run(['ngrok', 'config', 'add-authtoken', Config.NGROK_AUTH_TOKEN], 
                         check=True, capture_output=True)
            
            # Start ngrok tunnel
            self.tunnel_process = subprocess.Popen(
                ['ngrok', 'http', str(Config.PORT), '--log=stdout'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for tunnel to establish and get URL
            time.sleep(3)
            tunnel_url = self._get_ngrok_url()
            
            if tunnel_url:
                self.tunnel_url = tunnel_url
                return tunnel_url
            else:
                logger.error("Failed to get ngrok tunnel URL")
                self.stop_tunnel()
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start ngrok: {e}")
            return None
        except FileNotFoundError:
            logger.error("ngrok not found. Please install ngrok: https://ngrok.com/download")
            return None
    
    def _get_ngrok_url(self) -> Optional[str]:
        """Get ngrok tunnel URL from API"""
        try:
            response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
            if response.status_code == 200:
                data = response.json()
                tunnels = data.get('tunnels', [])
                for tunnel in tunnels:
                    if tunnel.get('proto') == 'https':
                        return tunnel.get('public_url')
            return None
        except requests.exceptions.RequestException:
            return None
    
    def _start_cloudflare(self) -> Optional[str]:
        """Start Cloudflare tunnel"""
        if not Config.CLOUDFLARE_TUNNEL_TOKEN:
            logger.error("CLOUDFLARE_TUNNEL_TOKEN not configured")
            return None
            
        try:
            logger.info("Starting Cloudflare tunnel")
            self.tunnel_process = subprocess.Popen([
                'cloudflared', 'tunnel', 'run',
                '--token', Config.CLOUDFLARE_TUNNEL_TOKEN
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Cloudflare tunnels use pre-configured domains
            # URL provided in the Cloudflare dashboard
            logger.info("Cloudflare tunnel started. Check Cloudflare dashboard for URL.")
            return "Check Cloudflare dashboard for tunnel URL"
            
        except FileNotFoundError:
            logger.error("cloudflared not found. Please install cloudflared: https://github.com/cloudflare/cloudflared")
            return None
    
    def stop_tunnel(self):
        """Stop the tunnel service"""
        if self.tunnel_process:
            logger.info("Stopping tunnel service")
            self.tunnel_process.terminate()
            try:
                self.tunnel_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.tunnel_process.kill()
            self.tunnel_process = None
            self.tunnel_url = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get tunnel status"""
        is_running = self.tunnel_process is not None and self.tunnel_process.poll() is None
        
        return {
            'service': self.service,
            'running': is_running,
            'url': self.tunnel_url,
            'process_id': self.tunnel_process.pid if self.tunnel_process else None
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_tunnel()