"""
Configuration management module.

Loads environment variables from .env.local with sensible defaults.
Supports environment-specific behavior (development vs production).
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development", description="Environment mode")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Server
    server_host: str = Field(default="0.0.0.0", description="Server bind host")
    server_port: int = Field(default=5000, description="Server bind port")

    # Paths
    db_path: Path = Field(default=Path("db"), description="Vector database path")
    images_path: Path = Field(
        default=Path("data/images"), description="Image storage path"
    )
    data_path: Path = Field(default=Path("data"), description="Data directory path")

    # Authentication
    api_key_required: bool = Field(
        default=False, description="Require API key for requests"
    )
    valid_api_keys: str = Field(
        default="", description="Comma-separated list of valid API keys"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path = Field(
        default=Path("logs/backend.log"), description="Log file path"
    )

    # Tunnel (development only)
    tunnel: bool = Field(default=False, description="Enable ngrok tunnel")
    ngrok_auth_token: Optional[str] = Field(
        default=None, description="Ngrok authentication token"
    )

    # Embedding models
    text_embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5", description="Text embedding model"
    )
    image_embedding_model: str = Field(
        default="ViT-L-14", description="CLIP model for image embeddings"
    )
    image_embedding_pretrained: str = Field(
        default="openai", description="CLIP model pretrained weights"
    )

    # Vision model for captioning
    vision_model: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct", description="Vision model for captioning"
    )

    # Chunking
    chunk_min_tokens: int = Field(
        default=100, description="Minimum tokens per chunk"
    )
    chunk_target_tokens: int = Field(
        default=500, description="Target tokens per chunk"
    )
    chunk_max_tokens: int = Field(
        default=800, description="Maximum tokens per chunk"
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=10, description="Number of chunks to retrieve"
    )
    text_weight: float = Field(
        default=0.7, description="Weight for text similarity in fusion"
    )
    image_weight: float = Field(
        default=0.3, description="Weight for image similarity in fusion"
    )

    # Runtime device selection
    cpu_only: bool = Field(
        default=False, description="Force CPU-only mode (ignore GPUs)"
    )
    # Rate limiting
    rate_limit_requests: int = Field(
        default=100, description="Max requests per minute"
    )
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Qdrant configuration (for remote/standalone mode)
    qdrant_host: Optional[str] = Field(
        default=None, description="Qdrant server host (None = local disk mode)"
    )
    qdrant_port: int = Field(
        default=6333, description="Qdrant server port"
    )

    # TEI (Text Embeddings Inference) configuration
    tei_url: Optional[str] = Field(
        default=None, description="TEI server URL (None = use local embedder)"
    )
    tei_timeout: float = Field(
        default=30.0, description="TEI request timeout in seconds"
    )

    # Concurrency limits (backpressure)
    max_concurrent_embeddings: int = Field(
        default=10, description="Max concurrent embedding requests"
    )
    max_concurrent_qdrant: int = Field(
        default=20, description="Max concurrent Qdrant queries"
    )
    max_concurrent_bm25: int = Field(
        default=4, description="Max concurrent BM25 searches"
    )
    bm25_thread_workers: int = Field(
        default=2, description="Thread pool size for BM25"
    )

    @field_validator("valid_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: str) -> str:
        """Keep as string, parse to list when needed."""
        return v if v else ""

    @property
    def api_keys_list(self) -> List[str]:
        """Get list of valid API keys."""
        if not self.valid_api_keys:
            return []
        return [k.strip() for k in self.valid_api_keys.split(",") if k.strip()]

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"

    @property
    def should_use_tunnel(self) -> bool:
        """Check if ngrok tunnel should be enabled."""
        return self.tunnel and self.is_development and self.ngrok_auth_token

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def get_image_url_base(self, public_url: Optional[str] = None) -> str:
        """
        Get base URL for image hosting.
        
        Args:
            public_url: Public URL if behind tunnel/proxy
            
        Returns:
            Base URL for constructing image URLs
        """
        if public_url:
            return f"{public_url.rstrip('/')}/images"
        return f"http://{self.server_host}:{self.server_port}/images"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Use this function to access settings throughout the application.
    Settings are cached for performance.
    """
    return Settings()


# Convenience alias
settings = get_settings()
