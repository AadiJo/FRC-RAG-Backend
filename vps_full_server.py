"""
FRC RAG Full Server for VPS Deployment
Serves the frontend and proxies Ollama requests to local PC
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for VPS deployment
class VPSConfig:
    # VPS server settings
    HOST = os.getenv('VPS_HOST', '0.0.0.0')
    PORT = int(os.getenv('VPS_PORT', 80))
    
    # Your local PC's tunnel URL
    REMOTE_OLLAMA_URL = os.getenv('REMOTE_OLLAMA_URL', '')  # Your ngrok URL
    REMOTE_API_KEY = os.getenv('REMOTE_API_KEY', '')       # API key for your local server
    
    # VPS settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', '/var/log/frc-rag-vps.log')
    
    # Rate limiting for VPS
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 200))
    
    # Security
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

# Configure logging
logging.basicConfig(
    level=getattr(logging, VPSConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(VPSConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app, origins=VPSConfig.ALLOWED_ORIGINS)

# Simple rate limiting (in production, use Redis)
request_counts = {}

def is_rate_limited(client_ip: str) -> bool:
    """Simple rate limiting check"""
    current_time = datetime.now()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests (older than 1 minute)
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if (current_time - req_time).seconds < 60
    ]
    
    # Check if limit exceeded
    if len(request_counts[client_ip]) >= VPSConfig.RATE_LIMIT_REQUESTS:
        return True
    
    # Add current request
    request_counts[client_ip].append(current_time)
    return False

def forward_request(endpoint: str, method: str = 'GET', data: Dict = None, params: Dict = None) -> Dict[str, Any]:
    """Forward request to local PC"""
    if not VPSConfig.REMOTE_OLLAMA_URL:
        return {"error": "Remote Ollama URL not configured", "status": "error"}
    
    try:
        url = f"{VPSConfig.REMOTE_OLLAMA_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'FRC-RAG-VPS-Proxy/1.0'
        }
        
        if VPSConfig.REMOTE_API_KEY:
            headers['Authorization'] = f'Bearer {VPSConfig.REMOTE_API_KEY}'
        
        logger.info(f"Forwarding {method} request to: {url}")
        
        if method.upper() == 'POST':
            response = requests.post(url, json=data, headers=headers, params=params, timeout=120)
        else:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        
        return response.json() if response.content else {"status": "success"}
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout forwarding request to {endpoint}")
        return {"error": "Request timeout - your local PC may be slow to respond", "status": "timeout"}
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error forwarding request to {endpoint}")
        return {"error": "Cannot connect to local PC - check if ngrok tunnel is running", "status": "connection_error"}
    except Exception as e:
        logger.error(f"Error forwarding request to {endpoint}: {str(e)}")
        return {"error": f"Proxy error: {str(e)}", "status": "proxy_error"}

# Frontend Routes
@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index_clean.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve image files from data/images directory"""
    import os
    images_dir = os.path.join(os.path.dirname(__file__), 'data', 'images')
    return send_from_directory(images_dir, filename)

# API Routes - Proxy to local PC
@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests"""
    client_ip = request.remote_addr
    
    # Rate limiting
    if is_rate_limited(client_ip):
        return jsonify({
            "error": "Rate limit exceeded. Please wait before making another request.",
            "status": "rate_limited"
        }), 429
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required", "status": "bad_request"}), 400
        
        logger.info(f"Query from {client_ip}: {data['query'][:100]}...")
        result = forward_request('api/query', 'POST', data)
        
        if result.get('status') in ['error', 'timeout', 'connection_error', 'proxy_error']:
            status_code = 503 if result.get('status') == 'connection_error' else 500
            return jsonify(result), status_code
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in query endpoint: {str(e)}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/api/ollama/<path:endpoint>', methods=['GET', 'POST'])
def ollama_proxy(endpoint):
    """Proxy all Ollama API requests to local PC"""
    client_ip = request.remote_addr
    
    # Rate limiting
    if is_rate_limited(client_ip):
        return jsonify({
            "error": "Rate limit exceeded. Please wait before making another request.",
            "status": "rate_limited"
        }), 429
    
    try:
        data = request.get_json() if request.method == 'POST' else None
        params = request.args.to_dict()
        
        logger.info(f"Ollama {request.method} from {client_ip}: {endpoint}")
        result = forward_request(f'api/ollama/{endpoint}', request.method, data, params)
        
        if result.get('status') in ['error', 'timeout', 'connection_error', 'proxy_error']:
            status_code = 503 if result.get('status') == 'connection_error' else 500
            return jsonify(result), status_code
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ollama proxy: {str(e)}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check connection to local PC
        local_health = forward_request('health', 'GET')
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "vps": {
                "status": "running",
                "rate_limit": f"{VPSConfig.RATE_LIMIT_REQUESTS} requests per minute",
                "remote_configured": bool(VPSConfig.REMOTE_OLLAMA_URL)
            },
            "local_pc": local_health,
            "proxy_stats": {
                "active_clients": len(request_counts),
                "total_rate_limited_ips": sum(1 for ip_requests in request_counts.values() 
                                            if len(ip_requests) >= VPSConfig.RATE_LIMIT_REQUESTS)
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/config')
def config():
    """Show current configuration (safe info only)"""
    return jsonify({
        "remote_url_configured": bool(VPSConfig.REMOTE_OLLAMA_URL),
        "remote_url_domain": VPSConfig.REMOTE_OLLAMA_URL.split('//')[1].split('/')[0] if VPSConfig.REMOTE_OLLAMA_URL else None,
        "rate_limit": VPSConfig.RATE_LIMIT_REQUESTS,
        "log_level": VPSConfig.LOG_LEVEL,
        "allowed_origins": VPSConfig.ALLOWED_ORIGINS
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found", "status": "not_found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status": "error"}), 500

if __name__ == '__main__':
    if not VPSConfig.REMOTE_OLLAMA_URL:
        logger.warning("REMOTE_OLLAMA_URL not configured! Set it in .env file")
        print("⚠️  Warning: REMOTE_OLLAMA_URL not configured!")
        print("Edit /opt/frc-rag/.env and set REMOTE_OLLAMA_URL=https://your-ngrok-url.ngrok.io")
    
    logger.info(f"Starting FRC RAG VPS Server on {VPSConfig.HOST}:{VPSConfig.PORT}")
    logger.info(f"Remote Ollama URL: {VPSConfig.REMOTE_OLLAMA_URL or 'NOT CONFIGURED'}")
    
    app.run(
        host=VPSConfig.HOST,
        port=VPSConfig.PORT,
        debug=False,
        threaded=True
    )