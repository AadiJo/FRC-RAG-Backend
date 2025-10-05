#!/usr/bin/env python3
"""
FRC RAG Server Test Script
Quick verification of server components
"""

import sys
import os
import importlib.util

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test core imports
        from src.server.config import get_config
        from src.server.rate_limiter import RateLimiter
        from src.server.ollama_proxy import OllamaProxy
        from src.server.tunnel import TunnelManager
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from src.server.config import get_config
        
        Config = get_config()
        config_dict = Config.to_dict()
        
        print(f"✅ Configuration loaded")
        print(f"   Environment: {Config.ENVIRONMENT}")
        print(f"   Server: {Config.HOST}:{Config.PORT}")
        print(f"   Ollama: {Config.get_ollama_url()}")
        print(f"   Rate limit: {Config.RATE_LIMIT_REQUESTS} req/{Config.RATE_LIMIT_WINDOW}min")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_rate_limiter():
    """Test rate limiter functionality"""
    print("🚦 Testing rate limiter...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from src.server.rate_limiter import RateLimiter
        
        # Create rate limiter with small limits for testing
        limiter = RateLimiter(max_requests=2, window_minutes=1)
        
        # Test client
        client_id = "test_client"
        
        # First request should be allowed
        assert limiter.is_allowed(client_id), "First request should be allowed"
        
        # Second request should be allowed
        assert limiter.is_allowed(client_id), "Second request should be allowed"
        
        # Third request should be denied
        assert not limiter.is_allowed(client_id), "Third request should be denied"
        
        # Check remaining requests
        remaining = limiter.get_remaining_requests(client_id)
        assert remaining == 0, f"Should have 0 remaining requests, got {remaining}"
        
        print("✅ Rate limiter working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Rate limiter error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 FRC RAG Server Component Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config, 
        test_rate_limiter
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Server components are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())