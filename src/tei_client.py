"""
Async Text Embeddings Inference (TEI) client.

Provides async HTTP client for external TEI service with:
- Connection pooling
- Request batching
- Semaphore-based backpressure
- Timeout handling
- Fallback to local embedder
"""

import asyncio
from typing import List, Optional, Union

import httpx

from .utils.config import settings
from .utils.logger import get_logger

logger = get_logger(__name__)


class AsyncTEIClient:
    """
    Async client for HuggingFace Text Embeddings Inference service.
    
    Features:
    - Async HTTP requests with connection pooling
    - Semaphore-based concurrency limiting
    - Automatic batching support
    - Timeout handling with retries
    - Graceful fallback to local embedder
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_concurrent: int = 10,
    ):
        """
        Initialize TEI client.
        
        Args:
            base_url: TEI server URL (e.g., http://127.0.0.1:8080)
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests (backpressure)
        """
        self.base_url = base_url or settings.tei_url
        self.timeout = timeout or settings.tei_timeout
        self.max_concurrent = max_concurrent or settings.max_concurrent_embeddings
        
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._local_embedder = None
        
        # Track if TEI is available
        self._tei_available: Optional[bool] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                ),
            )
        return self._client

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for backpressure."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    def _get_local_embedder(self):
        """Get local embedder for fallback (lazy load)."""
        if self._local_embedder is None:
            from .ingestion.embedder import TextEmbedder
            self._local_embedder = TextEmbedder(device="cpu")
            logger.info("Loaded local TextEmbedder as fallback")
        return self._local_embedder

    async def check_health(self) -> bool:
        """Check if TEI service is healthy."""
        if not self.base_url:
            return False
            
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"TEI health check failed: {e}")
            return False

    async def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(
        self,
        texts: List[str],
        retry_count: int = 2,
    ) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            retry_count: Number of retries on failure
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check if TEI is configured and available
        if not self.base_url:
            logger.debug("TEI not configured, using local embedder")
            return self._embed_local(texts)

        # Check TEI availability (cached)
        if self._tei_available is None:
            self._tei_available = await self.check_health()
            if self._tei_available:
                logger.info(f"TEI service available at {self.base_url}")
            else:
                logger.warning(f"TEI service unavailable at {self.base_url}, falling back to local")

        if not self._tei_available:
            return self._embed_local(texts)

        # Use semaphore for backpressure
        semaphore = self._get_semaphore()
        
        async with semaphore:
            for attempt in range(retry_count + 1):
                try:
                    client = await self._get_client()
                    
                    response = await client.post(
                        "/embed",
                        json={"inputs": texts},
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        # TEI returns embeddings directly as list of lists
                        return data
                    elif response.status_code == 429:
                        # Rate limited, wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"TEI rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"TEI error: {response.status_code} - {response.text}")
                        
                except httpx.TimeoutException:
                    logger.warning(f"TEI timeout on attempt {attempt + 1}")
                    if attempt < retry_count:
                        await asyncio.sleep(1)
                        continue
                except Exception as e:
                    logger.error(f"TEI request failed: {e}")
                    if attempt < retry_count:
                        await asyncio.sleep(1)
                        continue
            
            # All retries failed, fallback to local
            logger.warning("TEI failed after retries, using local embedder")
            self._tei_available = False
            return self._embed_local(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using local embedder (synchronous fallback)."""
        embedder = self._get_local_embedder()
        return [embedder.embed_text(text) for text in texts]

    def embed_sync(self, text: str) -> List[float]:
        """
        Synchronous embedding (for compatibility).
        
        Uses local embedder directly to avoid blocking event loop.
        """
        if self.base_url and self._tei_available:
            # If TEI is available, warn about sync usage
            logger.warning("Sync embedding called with TEI available - consider using async")
        
        embedder = self._get_local_embedder()
        return embedder.embed_text(text)

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# Global TEI client instance
_tei_client: Optional[AsyncTEIClient] = None


def get_tei_client() -> AsyncTEIClient:
    """Get or create global TEI client instance."""
    global _tei_client
    if _tei_client is None:
        _tei_client = AsyncTEIClient()
    return _tei_client


async def shutdown_tei_client():
    """Shutdown global TEI client (call on app shutdown)."""
    global _tei_client
    if _tei_client:
        await _tei_client.close()
        _tei_client = None
