"""
Configuration management for FRC RAG Server
Loads settings from environment variables with proper defaults
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Resolve backend root for path defaults
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    BASE_DIR = BASE_DIR
    
    # Server settings
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    HOST = os.getenv('SERVER_HOST', '0.0.0.0')
    PORT = int(os.getenv('SERVER_PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Ollama settings
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
    OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', 11434))
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 30))
    
    # Model provider settings
    # Provider values:
    # - 'local' (Ollama)
    # - 'openrouter' (OpenRouter)
    # Legacy: 'chute' is treated as OpenRouter for backward compatibility.
    MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'local')

    # Legacy (deprecated): retained for compatibility with older setups
    CHUTES_API_TOKEN = os.getenv('CHUTES_API_TOKEN', '')

    # OpenRouter settings
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    # Default to a reasonable OpenAI-compatible model (gpt-oss-20b) when using OpenRouter
    OPENROUTER_DEFAULT_MODEL = os.getenv('OPENROUTER_DEFAULT_MODEL', 'openai/gpt-oss-20b:free')
    # Optional but recommended attribution headers
    OPENROUTER_HTTP_REFERER = os.getenv('OPENROUTER_HTTP_REFERER', '')
    OPENROUTER_APP_TITLE = os.getenv('OPENROUTER_APP_TITLE', 'FRC RAG')

    SHOW_MODEL_REASONING = os.getenv('SHOW_MODEL_REASONING', 'false').lower() == 'true'
    
    # Rate limiting settings
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 60))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 1))
    
    # Database settings
    CHROMA_PATH = os.getenv('CHROMA_PATH', 'db')
    IMAGES_PATH = os.getenv('IMAGES_PATH', 'data/images')
    UPLOADS_BASE_PATH = os.getenv('UPLOADS_BASE_PATH', os.path.join(BASE_DIR, 'users'))
    
    # Security settings
    API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true'
    VALID_API_KEYS = [key.strip() for key in os.getenv('VALID_API_KEYS', '').split(',') if key.strip()]
    
    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/server.log')
    
    # CORS settings
    CORS_ORIGINS = [origin.strip() for origin in os.getenv('CORS_ORIGINS', '*').split(',')]
    
    # Performance settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))  # 16MB
    
    # Tunneling settings
    TUNNEL_SERVICE = os.getenv('TUNNEL_SERVICE', '')
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', '')
    CLOUDFLARE_TUNNEL_TOKEN = os.getenv('CLOUDFLARE_TUNNEL_TOKEN', '')
    
    @classmethod
    def get_ollama_url(cls) -> str:
        """Get Ollama base URL"""
        return f"http://{cls.OLLAMA_HOST}:{cls.OLLAMA_PORT}"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        provider = cls.MODEL_PROVIDER
        if provider == 'chute':
            provider = 'openrouter'
        return {
            'server': {
                'environment': cls.ENVIRONMENT,
                'host': cls.HOST,
                'port': cls.PORT,
                'debug': cls.DEBUG,
                'max_content_length': cls.MAX_CONTENT_LENGTH
            },
            'ollama': {
                'host': cls.OLLAMA_HOST,
                'port': cls.OLLAMA_PORT,
                'timeout': cls.OLLAMA_TIMEOUT,
                'url': cls.get_ollama_url()
            },
            'model_provider': {
                'provider': provider,
                'openrouter_configured': bool((cls.OPENROUTER_API_KEY or cls.CHUTES_API_TOKEN).strip()),
                'show_reasoning': cls.SHOW_MODEL_REASONING
            },
            'rate_limiting': {
                'requests': cls.RATE_LIMIT_REQUESTS,
                'window_minutes': cls.RATE_LIMIT_WINDOW
            },
            'database': {
                'chroma_path': cls.CHROMA_PATH,
                'images_path': cls.IMAGES_PATH,
                'uploads_base_path': cls.UPLOADS_BASE_PATH
            },
            'security': {
                'api_key_required': cls.API_KEY_REQUIRED,
                'cors_origins': cls.CORS_ORIGINS
            },
            'logging': {
                'level': cls.LOG_LEVEL,
                'file': cls.LOG_FILE
            },
            'tunneling': {
                'service': cls.TUNNEL_SERVICE,
                'ngrok_configured': bool(cls.NGROK_AUTH_TOKEN),
                'cloudflare_configured': bool(cls.CLOUDFLARE_TUNNEL_TOKEN)
            }
        }

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    RATE_LIMIT_REQUESTS = 1000  # More lenient for development
    API_KEY_REQUIRED = False

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    API_KEY_REQUIRED = True
    RATE_LIMIT_REQUESTS = 60  # Strict for production

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    RATE_LIMIT_REQUESTS = 1000
    CHROMA_PATH = 'test_db'
    API_KEY_REQUIRED = False

# Select configuration based on environment
def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProductionConfig
    elif env == 'testing':
        return TestingConfig
    else:
        return DevelopmentConfig