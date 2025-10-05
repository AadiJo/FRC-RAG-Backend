"""
Rate limiting implementation for FRC RAG Server
Thread-safe sliding window rate limiter
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import defaultdict, deque
from threading import Lock

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter using sliding window algorithm"""
    
    def __init__(self, max_requests: int = 60, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = defaultdict(deque)
        self.lock = Lock()
        logger.info(f"Rate limiter initialized: {max_requests} requests per {window_minutes} minute(s)")
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests outside the window
            while client_requests and client_requests[0] <= now - self.window_seconds:
                client_requests.popleft()
            
            # Check if under limit
            if len(client_requests) < self.max_requests:
                client_requests.append(now)
                return True
            
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False
    
    def get_reset_time(self, client_id: str) -> Optional[datetime]:
        """Get when the rate limit resets for a client"""
        with self.lock:
            client_requests = self.requests[client_id]
            if not client_requests:
                return None
            
            oldest_request = client_requests[0]
            reset_time = datetime.fromtimestamp(oldest_request + self.window_seconds)
            return reset_time
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for a client"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests
            while client_requests and client_requests[0] <= now - self.window_seconds:
                client_requests.popleft()
            
            return max(0, self.max_requests - len(client_requests))
    
    def get_client_stats(self, client_id: str) -> Dict[str, any]:
        """Get detailed stats for a client"""
        with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]
            
            # Remove old requests
            while client_requests and client_requests[0] <= now - self.window_seconds:
                client_requests.popleft()
            
            remaining = max(0, self.max_requests - len(client_requests))
            reset_time = None
            
            if client_requests:
                oldest_request = client_requests[0]
                reset_time = datetime.fromtimestamp(oldest_request + self.window_seconds)
            
            return {
                'requests_made': len(client_requests),
                'requests_remaining': remaining,
                'reset_time': reset_time.isoformat() if reset_time else None,
                'window_seconds': self.window_seconds,
                'max_requests': self.max_requests
            }
    
    def cleanup_old_clients(self, max_age_hours: int = 24):
        """Remove inactive clients to prevent memory leaks"""
        with self.lock:
            now = time.time()
            cutoff = now - (max_age_hours * 3600)
            
            clients_to_remove = []
            for client_id, requests in self.requests.items():
                # Remove old requests for this client
                while requests and requests[0] <= cutoff:
                    requests.popleft()
                
                # If no recent requests, mark for removal
                if not requests:
                    clients_to_remove.append(client_id)
            
            for client_id in clients_to_remove:
                del self.requests[client_id]
            
            if clients_to_remove:
                logger.info(f"Cleaned up {len(clients_to_remove)} inactive clients")
    
    def get_stats(self) -> Dict[str, any]:
        """Get overall rate limiter statistics"""
        with self.lock:
            total_clients = len(self.requests)
            total_active_requests = sum(len(requests) for requests in self.requests.values())
            
            return {
                'total_clients': total_clients,
                'total_active_requests': total_active_requests,
                'window_seconds': self.window_seconds,
                'max_requests_per_window': self.max_requests
            }