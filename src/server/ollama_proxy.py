"""
Ollama proxy server with monitoring and health checks
"""

import time
import logging
import requests
from typing import Dict, Any, Optional
from collections import defaultdict
from threading import Lock
from werkzeug.exceptions import TooManyRequests

from .config import get_config
from .rate_limiter import RateLimiter

Config = get_config()
logger = logging.getLogger(__name__)

class OllamaProxy:
    """Proxy server for Ollama with rate limiting and monitoring"""
    
    def __init__(self):
        self.ollama_base_url = Config.get_ollama_url()
        self.rate_limiter = RateLimiter(
            max_requests=Config.RATE_LIMIT_REQUESTS,
            window_minutes=Config.RATE_LIMIT_WINDOW
        )
        self.model_usage = defaultdict(int)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.usage_lock = Lock()
        
        if Config.MODEL_PROVIDER == 'local':
            logger.info(f"Ollama proxy initialized for {self.ollama_base_url}")
        
    def check_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def get_client_id(self, request) -> str:
        """Get client identifier for rate limiting"""
        # Check for API key first
        api_key = request.headers.get('X-API-Key')
        if api_key and Config.API_KEY_REQUIRED:
            if api_key in Config.VALID_API_KEYS:
                return f"api_key:{api_key}"
            else:
                raise ValueError("Invalid API key")
        
        # Try to get real IP behind proxy
        client_ip = request.headers.get('X-Real-IP') or \
                   request.headers.get('X-Forwarded-For', '').split(',')[0] or \
                   request.remote_addr
        
        return f"ip:{client_ip}"
    
    def proxy_request(self, client_id: str, path: str, method: str = 'GET', 
                     json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Proxy request to Ollama with rate limiting"""
        
        with self.usage_lock:
            self.request_count += 1
        
        # Check rate limit
        if not self.rate_limiter.is_allowed(client_id):
            client_stats = self.rate_limiter.get_client_stats(client_id)
            raise TooManyRequests(
                description=f"Rate limit exceeded. {client_stats['requests_remaining']} requests remaining. Reset at {client_stats['reset_time']}"
            )
        
        # Check Ollama health
        if not self.check_health():
            with self.usage_lock:
                self.error_count += 1
            return {"error": "Ollama service is not available", "status": "service_unavailable"}
        
        # Make request to Ollama
        url = f"{self.ollama_base_url}/{path}"
        
        try:
            if method.upper() == 'POST':
                response = requests.post(url, json=json_data, timeout=Config.OLLAMA_TIMEOUT)
            else:
                response = requests.get(url, timeout=Config.OLLAMA_TIMEOUT)
            
            # Track usage
            if json_data and 'model' in json_data:
                with self.usage_lock:
                    self.model_usage[json_data['model']] += 1
            
            # Return response
            if response.status_code == 200:
                return response.json()
            else:
                with self.usage_lock:
                    self.error_count += 1
                return {
                    "error": f"Ollama request failed with status {response.status_code}",
                    "status": "ollama_error"
                }
                
        except requests.exceptions.Timeout:
            with self.usage_lock:
                self.error_count += 1
            return {"error": "Request to Ollama timed out", "status": "timeout"}
        except requests.exceptions.RequestException as e:
            with self.usage_lock:
                self.error_count += 1
            return {"error": f"Request to Ollama failed: {str(e)}", "status": "request_error"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics"""
        uptime = time.time() - self.start_time
        with self.usage_lock:
            stats = {
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_uptime(uptime),
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "error_rate": self.error_count / max(1, self.request_count),
                "requests_per_second": self.request_count / max(1, uptime),
                "model_usage": dict(self.model_usage),
                "ollama_healthy": self.check_health(),
                "rate_limiter": self.rate_limiter.get_stats()
            }
        
        return stats
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def cleanup(self):
        """Cleanup old rate limit data"""
        self.rate_limiter.cleanup_old_clients()