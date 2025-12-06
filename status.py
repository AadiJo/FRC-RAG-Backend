#!/usr/bin/env python3
"""
Quick status check for FRC RAG backend configuration
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from server.config import get_config
    Config = get_config()
    
    print("🔧 FRC RAG Backend Configuration")
    print("=" * 40)
    print(f"Environment: {Config.ENVIRONMENT}")
    print(f"Server: {Config.HOST}:{Config.PORT}")
    print(f"Debug: {Config.DEBUG}")
    print()
    
    print(f"Model Provider: {Config.MODEL_PROVIDER}")
    if Config.MODEL_PROVIDER == 'local':
        print(f"Ollama URL: {Config.get_ollama_url()}")
    elif Config.MODEL_PROVIDER == 'chute':
        print(f"Chutes API: {'✅ Configured' if Config.CHUTES_API_TOKEN else '❌ Not configured'}")
        print(f"Show Reasoning: {Config.SHOW_MODEL_REASONING}")
    print()
    
    print(f"Database: {Config.CHROMA_PATH}")
    print(f"Images: {Config.IMAGES_PATH}")
    print(f"Rate Limit: {Config.RATE_LIMIT_REQUESTS} requests/{Config.RATE_LIMIT_WINDOW}min")
    print()
    
    # Check file existence
    db_exists = os.path.exists(Config.CHROMA_PATH)
    images_exist = os.path.exists(Config.IMAGES_PATH)
    
    print("📁 File System")
    print(f"Database: {'✅' if db_exists else '❌'} {Config.CHROMA_PATH}")
    print(f"Images: {'✅' if images_exist else '❌'} {Config.IMAGES_PATH}")
    
    # Test model provider
    print("\n🤖 Model Provider Test")
    if Config.MODEL_PROVIDER == 'chute':
        try:
            from server.chutes_client import ChutesClient
            client = ChutesClient()
            health = client.check_health()
            print(f"Chutes AI: {'✅ Healthy' if health else '❌ Unhealthy'}")
        except Exception as e:
            print(f"Chutes AI: ❌ Error - {e}")
    else:
        import requests
        try:
            response = requests.get(f"{Config.get_ollama_url()}/api/tags", timeout=5)
            health = response.status_code == 200
            print(f"Ollama: {'✅ Healthy' if health else '❌ Unhealthy'}")
        except Exception as e:
            print(f"Ollama: ❌ Error - {e}")

except Exception as e:
    print(f"❌ Configuration error: {e}")
    import traceback
    traceback.print_exc()