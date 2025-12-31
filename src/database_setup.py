"""
Vector database setup module.

Handles:
- Qdrant vector database initialization
- Collection creation for text and image chunks
- Metadata indexing and filtering
- Bulk ingestion from Parquet/JSONL
- Backup and restore
- Async operations for non-blocking queries
"""

import asyncio
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .utils.config import settings
from .utils.logger import get_logger
from .utils.metrics import metrics

logger = get_logger(__name__)


# Collection names
TEXT_COLLECTION = "frc_text_chunks"
IMAGE_COLLECTION = "frc_image_chunks"
COLPALI_COLLECTION = "frc_colpali"
USER_DOCS_COLLECTION = "user_docs"  # Multi-tenant user documents

# Embedding dimensions
TEXT_EMBEDDING_DIM = 1024  # bge-large-en-v1.5
IMAGE_EMBEDDING_DIM = 768  # CLIP ViT-L/14
COLPALI_EMBEDDING_DIM = 128  # ColQwen2 output dim


@dataclass
class CollectionInfo:
    """Information about a vector collection."""
    
    name: str
    vector_size: int
    points_count: int
    indexed_fields: List[str]


class VectorDatabase:
    """
    Qdrant vector database manager.
    
    Features:
    - Disk-backed storage or remote server
    - Separate collections for text and images
    - Metadata filtering
    - Bulk ingestion
    - Async operations for non-blocking queries
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize vector database.
        
        Args:
            path: Path for local disk storage (default: from settings)
            host: Qdrant server host (for remote mode, default: from settings)
            port: Qdrant server port (default: from settings)
        """
        self.path = Path(path or settings.db_path)
        # Use settings for remote mode if not explicitly provided
        self.host = host if host is not None else settings.qdrant_host
        self.port = port if port is not None else settings.qdrant_port
        
        self._client: Optional[QdrantClient] = None
        self._async_client: Optional[AsyncQdrantClient] = None
        
        # Semaphore for async query backpressure
        self._qdrant_semaphore: Optional[asyncio.Semaphore] = None

    def _get_qdrant_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for Qdrant query backpressure."""
        if self._qdrant_semaphore is None:
            self._qdrant_semaphore = asyncio.Semaphore(settings.max_concurrent_qdrant)
        return self._qdrant_semaphore

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            if self.host:
                # Remote mode
                logger.info(
                    "Connecting to remote Qdrant",
                    host=self.host,
                    port=self.port,
                )
                self._client = QdrantClient(host=self.host, port=self.port)
            else:
                # Local disk mode
                self.path.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Initializing local Qdrant",
                    path=str(self.path),
                )
                self._client = QdrantClient(path=str(self.path))
        
        return self._client

    async def _get_async_client(self) -> AsyncQdrantClient:
        """Get or create async Qdrant client for non-blocking operations.
        
        NOTE: Only works with remote Qdrant. For local mode, use _run_sync_in_executor.
        """
        if not self.host:
            raise RuntimeError(
                "AsyncQdrantClient requires remote Qdrant server. "
                "Local embedded Qdrant doesn't support concurrent clients. "
                "Use QDRANT_HOST environment variable to connect to remote Qdrant."
            )
        
        if self._async_client is None:
            logger.info(
                "Creating async Qdrant client",
                host=self.host,
                port=self.port,
            )
            self._async_client = AsyncQdrantClient(host=self.host, port=self.port)
        
        return self._async_client

    async def _run_sync_in_executor(self, func, *args, **kwargs):
        """Run a sync function in thread pool executor (for local Qdrant mode)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Use default executor
            lambda: func(*args, **kwargs)
        )

    @property
    def is_remote(self) -> bool:
        """Check if connected to remote Qdrant server."""
        return self.host is not None

    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client."""
        return self._get_client()

    def _create_collection_if_not_exists(
        self,
        name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """
        Create collection if it doesn't exist.
        
        Returns:
            True if created, False if already exists
        """
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]
        
        if name in existing:
            logger.debug(f"Collection {name} already exists")
            return False
        
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        
        logger.info(
            "Collection created",
            name=name,
            vector_size=vector_size,
        )
        
        return True

    def _create_payload_index(
        self, collection_name: str, field_name: str, field_type: str = "keyword"
    ) -> None:
        """Create index on a payload field for faster filtering."""
        try:
            schema_type = {
                "keyword": rest.PayloadSchemaType.KEYWORD,
                "integer": rest.PayloadSchemaType.INTEGER,
                "float": rest.PayloadSchemaType.FLOAT,
                "text": rest.PayloadSchemaType.TEXT,
            }.get(field_type, rest.PayloadSchemaType.KEYWORD)
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            
            logger.debug(
                "Payload index created",
                collection=collection_name,
                field=field_name,
            )
        except Exception as e:
            # Index might already exist
            logger.debug(f"Index creation skipped: {e}")

    def initialize(self, image_embedding_dim: Optional[int] = None) -> None:
        """
        Initialize database with required collections and indexes.
        
        Creates:
        - frc_text_chunks: Text embeddings collection
        - frc_image_chunks: Image embeddings collection
        
        Args:
            image_embedding_dim: Optional image embedding dimension. If not provided,
                uses IMAGE_EMBEDDING_DIM. If provided, will recreate collection if
                dimension doesn't match existing collection.
        """
        logger.info("Initializing vector database")
        
        # Create text chunks collection
        self._create_collection_if_not_exists(
            TEXT_COLLECTION,
            TEXT_EMBEDDING_DIM,
        )
        
        # Determine image embedding dimension
        actual_image_dim = image_embedding_dim or IMAGE_EMBEDDING_DIM
        
        # Check if image collection exists and has different dimension
        try:
            existing_collection_info = self.client.get_collection(IMAGE_COLLECTION)
            existing_dim = existing_collection_info.config.params.vectors.size
            if existing_dim != actual_image_dim:
                logger.warning(
                    f"Image collection exists with dimension {existing_dim}, "
                    f"but need {actual_image_dim}. Collection will need to be recreated."
                )
                # Delete and recreate collection
                logger.info(f"Deleting existing {IMAGE_COLLECTION} collection")
                self.client.delete_collection(IMAGE_COLLECTION)
        except Exception:
            # Collection doesn't exist, which is fine
            pass
        
        # Create image chunks collection
        self._create_collection_if_not_exists(
            IMAGE_COLLECTION,
            actual_image_dim,
        )

        # Create ColPali collection (Multi-vector)
        # Check if exists first
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]
        
        if COLPALI_COLLECTION not in existing:
            logger.info("Creating ColPali multi-vector collection")
            from qdrant_client.http.models import VectorParams, MultiVectorConfig, MultiVectorComparator
            
            self.client.create_collection(
                collection_name=COLPALI_COLLECTION,
                vectors_config={
                    "colpali": VectorParams(
                        size=COLPALI_EMBEDDING_DIM,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )
        
        # Create user documents collection (uses same embedding dim as text)
        self._create_collection_if_not_exists(
            USER_DOCS_COLLECTION,
            TEXT_EMBEDDING_DIM,
        )
        
        # Create indexes for filtering
        for collection in [TEXT_COLLECTION, IMAGE_COLLECTION, COLPALI_COLLECTION]:
            for field in ["team", "year", "subsystem", "binder"]:
                self._create_payload_index(collection, field, "keyword")
            self._create_payload_index(collection, "page_number", "integer")
        
        # Create indexes for user_docs (multi-tenant filtering)
        self._create_payload_index(USER_DOCS_COLLECTION, "user_id", "keyword")
        self._create_payload_index(USER_DOCS_COLLECTION, "doc_id", "keyword")
        self._create_payload_index(USER_DOCS_COLLECTION, "title", "keyword")
        
        logger.info(
            "Vector database initialized",
            image_embedding_dim=actual_image_dim,
        )

    def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(name)
            
            return CollectionInfo(
                name=name,
                vector_size=info.config.params.vectors.size,
                points_count=info.points_count,
                indexed_fields=[],  # Would need to query indexes
            )
        except Exception:
            return None

    def upsert_text_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert text chunks to the database.
        
        Args:
            chunks: List of dicts with 'id', 'embedding', and metadata
            batch_size: Batch size for upserting
            
        Returns:
            Number of chunks upserted
        """
        logger.info(
            "Upserting text chunks",
            count=len(chunks),
        )
        
        points = []
        for chunk in chunks:
            # Generate UUID from chunk_id for Qdrant
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))
            
            # Prepare payload (all metadata except embedding)
            payload = {k: v for k, v in chunk.items() if k != "embedding"}
            
            points.append(PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload,
            ))
        
        # Upsert in batches
        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            self.client.upsert(
                collection_name=TEXT_COLLECTION,
                points=batch,
            )
            
            upserted += len(batch)
            logger.debug(f"Upserted {upserted}/{len(points)} text chunks")
        
        logger.info(
            "Text chunks upserted",
            count=upserted,
        )
        
        return upserted

    def upsert_image_chunks(
        self,
        images: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert image embeddings to the database.
        
        Args:
            images: List of dicts with 'id', 'embedding', and metadata
            batch_size: Batch size for upserting
            
        Returns:
            Number of images upserted
        """
        logger.info(
            "Upserting image chunks",
            count=len(images),
        )
        
        points = []
        for img in images:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, img["id"]))
            
            payload = {k: v for k, v in img.items() if k != "embedding"}
            
            points.append(PointStruct(
                id=point_id,
                vector=img["embedding"],
                payload=payload,
            ))
        
        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            self.client.upsert(
                collection_name=IMAGE_COLLECTION,
                points=batch,
            )
            
            upserted += len(batch)
        
        logger.info(
            "Image chunks upserted",
            count=upserted,
        )
        
        return upserted

    def upsert_colpali_pages(
        self,
        pages: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> int:
        """
        Upsert ColPali multi-vector pages.
        
        Args:
            pages: List of dicts with 'id', 'multivector' (List[List[float]]), and metadata
            batch_size: Batch size (smaller due to large payloads)
        """
        logger.info("Upserting ColPali pages", count=len(pages))
        
        points = []
        for page in pages:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, page["id"]))
            payload = {k: v for k, v in page.items() if k != "multivector"}
            
            points.append(PointStruct(
                id=point_id,
                vector={"colpali": page["multivector"]},
                payload=payload,
            ))
            
        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=COLPALI_COLLECTION,
                points=batch,
            )
            upserted += len(batch)
            logger.debug(f"Upserted {upserted}/{len(points)} ColPali pages")
            
        return upserted

    def ingest_from_parquet(
        self,
        text_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Bulk ingest from Parquet files.
        
        Args:
            text_path: Path to text embeddings Parquet
            image_path: Path to image embeddings Parquet
            
        Returns:
            Dict with counts of ingested records
        """
        import pandas as pd
        
        results = {"text": 0, "image": 0}
        
        if text_path and Path(text_path).exists():
            logger.info(f"Loading text embeddings from {text_path}")
            df = pd.read_parquet(text_path)
            
            chunks = []
            for _, row in df.iterrows():
                chunk = row.to_dict()
                # Convert numpy array to list if needed
                if hasattr(chunk.get("embedding"), "tolist"):
                    chunk["embedding"] = chunk["embedding"].tolist()
                chunks.append(chunk)
            
            results["text"] = self.upsert_text_chunks(chunks)
        
        if image_path and Path(image_path).exists():
            logger.info(f"Loading image embeddings from {image_path}")
            df = pd.read_parquet(image_path)
            
            images = []
            for _, row in df.iterrows():
                img = row.to_dict()
                if hasattr(img.get("embedding"), "tolist"):
                    img["embedding"] = img["embedding"].tolist()
                images.append(img)
            
            results["image"] = self.upsert_image_chunks(images)
        
        return results

    def ingest_from_jsonl(
        self,
        text_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Bulk ingest from JSONL files.
        
        Args:
            text_path: Path to text embeddings JSONL
            image_path: Path to image embeddings JSONL
            
        Returns:
            Dict with counts of ingested records
        """
        results = {"text": 0, "image": 0}
        
        if text_path and Path(text_path).exists():
            logger.info(f"Loading text embeddings from {text_path}")
            chunks = []
            with open(text_path) as f:
                for line in f:
                    chunks.append(json.loads(line))
            results["text"] = self.upsert_text_chunks(chunks)
        
        if image_path and Path(image_path).exists():
            logger.info(f"Loading image embeddings from {image_path}")
            images = []
            with open(image_path) as f:
                for line in f:
                    images.append(json.loads(line))
            results["image"] = self.upsert_image_chunks(images)
        
        return results

    def search_text(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search text chunks collection.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum results to return
            filters: Optional metadata filters
            score_threshold: Minimum score threshold
            
        Returns:
            List of matching chunks with scores
        """
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        results = self.client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": result.payload.get("id", str(result.id)),
                "score": result.score,
                **result.payload,
            }
            for result in results.points
        ]

    def search_images(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search image chunks collection.
        
        Args:
            query_vector: Query embedding vector (CLIP)
            limit: Maximum results to return
            filters: Optional metadata filters
            score_threshold: Minimum score threshold
            
        Returns:
            List of matching images with scores
        """
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        results = self.client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": result.payload.get("id", str(result.id)),
                "score": result.score,
                **result.payload,
            }
            for result in results.points
        ]

    def search_colpali(
        self,
        query_multivector: List[List[float]],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search ColPali collection using MaxSim (Late Interaction).
        """
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)
                
        results = self.client.query_points(
            collection_name=COLPALI_COLLECTION,
            query=query_multivector,
            using="colpali",
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": result.payload.get("id", str(result.id)),
                "score": result.score,
                **result.payload,
            }
            for result in results.points
        ]

    # =========================================================================
    # Async Search Methods (Non-blocking for FastAPI)
    # =========================================================================

    async def async_search_text(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Async search text chunks collection (non-blocking).
        
        Uses async client for remote Qdrant, or run_in_executor for local mode.
        """
        semaphore = self._get_qdrant_semaphore()
        async with semaphore:
            if self.is_remote:
                # True async for remote Qdrant
                client = await self._get_async_client()
                qdrant_filter = self._build_filter(filters)
                results = await client.query_points(
                    collection_name=TEXT_COLLECTION,
                    query=query_vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                )
                return [
                    {
                        "id": result.payload.get("id", str(result.id)),
                        "score": result.score,
                        **result.payload,
                    }
                    for result in results.points
                ]
            else:
                # Run sync method in executor for local Qdrant
                return await self._run_sync_in_executor(
                    self.search_text,
                    query_vector=query_vector,
                    limit=limit,
                    filters=filters,
                    score_threshold=score_threshold,
                )

    async def async_search_images(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Async search image chunks collection (non-blocking).
        """
        semaphore = self._get_qdrant_semaphore()
        async with semaphore:
            if self.is_remote:
                client = await self._get_async_client()
                qdrant_filter = self._build_filter(filters)
                results = await client.query_points(
                    collection_name=IMAGE_COLLECTION,
                    query=query_vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                )
                return [
                    {
                        "id": result.payload.get("id", str(result.id)),
                        "score": result.score,
                        **result.payload,
                    }
                    for result in results.points
                ]
            else:
                return await self._run_sync_in_executor(
                    self.search_images,
                    query_vector=query_vector,
                    limit=limit,
                    filters=filters,
                    score_threshold=score_threshold,
                )

    async def async_search_colpali(
        self,
        query_multivector: List[List[float]],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Async search ColPali collection using MaxSim.
        """
        semaphore = self._get_qdrant_semaphore()
        async with semaphore:
            if self.is_remote:
                client = await self._get_async_client()
                qdrant_filter = self._build_filter(filters)
                results = await client.query_points(
                    collection_name=COLPALI_COLLECTION,
                    query=query_multivector,
                    using="colpali",
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                )
                return [
                    {
                        "id": result.payload.get("id", str(result.id)),
                        "score": result.score,
                        **result.payload,
                    }
                    for result in results.points
                ]
            else:
                return await self._run_sync_in_executor(
                    self.search_colpali,
                    query_multivector=query_multivector,
                    limit=limit,
                    filters=filters,
                    score_threshold=score_threshold,
                )

    async def async_search_user_docs(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 50,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Async search user documents collection.
        
        Always filters by user_id for multi-tenant security.
        """
        semaphore = self._get_qdrant_semaphore()
        async with semaphore:
            if self.is_remote:
                # CRITICAL: Always filter by user_id for multi-tenancy
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                )
                client = await self._get_async_client()
                results = await client.query_points(
                    collection_name=USER_DOCS_COLLECTION,
                    query=query_vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                )
                return [
                    {
                        "id": result.payload.get("id", str(result.id)),
                        "score": result.score,
                        **result.payload,
                    }
                    for result in results.points
                ]
            else:
                return await self._run_sync_in_executor(
                    self.search_user_docs,
                    query_vector=query_vector,
                    user_id=user_id,
                    limit=limit,
                    score_threshold=score_threshold,
                )

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Build Qdrant filter from dict."""
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
        
        return Filter(must=conditions) if conditions else None

    def check_colpali_pdf_exists(self, filename: str) -> bool:
        """Check if any pages for a specific PDF already exist in ColPali collection."""
        try:
            results = self.client.scroll(
                collection_name=COLPALI_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="binder",
                            match=MatchValue(value=filename),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results[0]) > 0
        except Exception:
            return False

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        try:
            results = self.client.retrieve(
                collection_name=TEXT_COLLECTION,
                ids=[point_id],
            )
            
            if results:
                return {
                    "id": results[0].payload.get("id", chunk_id),
                    **results[0].payload,
                }
        except Exception:
            pass
        
        return None

    def get_image_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific image record by its logical image_id.

        Note: Qdrant point IDs are UUID5 hashes of the logical IDs.

        Args:
            image_id: Image identifier

        Returns:
            Image payload (including metadata) or None if not found
        """
        if not image_id:
            return None

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, image_id))

        try:
            results = self.client.retrieve(
                collection_name=IMAGE_COLLECTION,
                ids=[point_id],
            )

            if results:
                return {
                    "id": results[0].payload.get("id", image_id),
                    **results[0].payload,
                }
        except Exception:
            pass

        return None

    def get_all_text_chunks(self) -> List[Dict[str, Any]]:
        """
        Fetch all text chunks for BM25 indexing.
        WARNING: This loads all chunks into memory.
        """
        # Get count for pagination
        count = self.client.count(collection_name=TEXT_COLLECTION).count
        
        all_points = []
        offset = None
        
        while len(all_points) < count:
            batch, offset = self.client.scroll(
                collection_name=TEXT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            
            if offset is None:
                break
                
        return [
            {
                "id": p.payload.get("id", str(p.id)),
                **p.payload
            }
            for p in all_points
        ]

    def delete_by_filter(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> int:
        """
        Delete points matching filters.
        
        Args:
            collection: Collection name
            filters: Metadata filters
            
        Returns:
            Number of points deleted (estimate)
        """
        conditions = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
        
        if not conditions:
            return 0
        
        # Get count before delete
        count_before = self.client.count(
            collection_name=collection,
            count_filter=Filter(must=conditions),
        ).count
        
        self.client.delete(
            collection_name=collection,
            points_selector=rest.FilterSelector(
                filter=Filter(must=conditions),
            ),
        )
        
        logger.info(
            "Points deleted",
            collection=collection,
            count=count_before,
        )
        
        return count_before

    def backup(self, backup_dir: Path) -> Path:
        """
        Create backup of the database.
        
        Args:
            backup_dir: Directory to store backup
            
        Returns:
            Path to backup
        """
        backup_dir = Path(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"qdrant_backup_{timestamp}"
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # For local mode, copy the entire directory
        if not self.host:
            shutil.copytree(self.path, backup_path / "db", dirs_exist_ok=True)
            
            logger.info(
                "Database backed up",
                path=str(backup_path),
            )
        else:
            # For remote mode, would need to use Qdrant's snapshot API
            logger.warning("Remote backup not implemented, use Qdrant snapshots")
        
        return backup_path

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        for collection_name in [TEXT_COLLECTION, IMAGE_COLLECTION, USER_DOCS_COLLECTION]:
            try:
                info = self.client.get_collection(collection_name)
                stats[collection_name] = {
                    "points_count": getattr(info, "points_count", 0),
                    "status": str(getattr(info, "status", "unknown")),
                }
            except Exception as e:
                stats[collection_name] = {"error": str(e)}
        
        return stats

    # =========================================================================
    # User Documents Methods
    # =========================================================================

    def upsert_user_docs(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert user document chunks to the database.
        
        Args:
            chunks: List of dicts with 'id', 'embedding', 'user_id', 'doc_id', and metadata
            batch_size: Batch size for upserting
            
        Returns:
            Number of chunks upserted
        """
        if not chunks:
            return 0
            
        logger.info(
            "Upserting user document chunks",
            count=len(chunks),
            user_id=chunks[0].get("user_id", "unknown"),
        )
        
        points = []
        for chunk in chunks:
            # Generate UUID from chunk_id for Qdrant
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))
            
            # Prepare payload (all metadata except embedding)
            payload = {k: v for k, v in chunk.items() if k != "embedding"}
            
            points.append(PointStruct(
                id=point_id,
                vector=chunk["embedding"],
                payload=payload,
            ))
        
        # Upsert in batches
        upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            self.client.upsert(
                collection_name=USER_DOCS_COLLECTION,
                points=batch,
            )
            
            upserted += len(batch)
            logger.debug(f"Upserted {upserted}/{len(points)} user doc chunks")
        
        logger.info(
            "User document chunks upserted",
            count=upserted,
        )
        
        return upserted

    def delete_user_docs(
        self,
        user_id: str,
        doc_ids: List[str],
    ) -> Dict[str, List[str]]:
        """
        Delete user documents by user_id and doc_ids.
        
        Args:
            user_id: User identifier (required for security)
            doc_ids: List of document IDs to delete
            
        Returns:
            Dict with 'deleted' and 'not_found' lists
        """
        deleted = []
        not_found = []
        
        for doc_id in doc_ids:
            # Count matching points before delete
            conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ]
            
            count_before = self.client.count(
                collection_name=USER_DOCS_COLLECTION,
                count_filter=Filter(must=conditions),
            ).count
            
            if count_before == 0:
                not_found.append(doc_id)
                continue
            
            # Delete matching points
            self.client.delete(
                collection_name=USER_DOCS_COLLECTION,
                points_selector=rest.FilterSelector(
                    filter=Filter(must=conditions),
                ),
            )
            
            deleted.append(doc_id)
            logger.info(
                "User document deleted",
                user_id=user_id,
                doc_id=doc_id,
                chunks_deleted=count_before,
            )
        
        return {"deleted": deleted, "not_found": not_found}

    def search_user_docs(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 50,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search user documents collection.
        
        Always filters by user_id for multi-tenant security.
        
        Args:
            query_vector: Query embedding vector
            user_id: User identifier (REQUIRED for security)
            limit: Maximum results to return
            score_threshold: Minimum score threshold
            
        Returns:
            List of matching chunks with scores
        """
        # CRITICAL: Always filter by user_id for multi-tenancy
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
        )
        
        results = self.client.query_points(
            collection_name=USER_DOCS_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": result.payload.get("id", str(result.id)),
                "score": result.score,
                **result.payload,
            }
            for result in results.points
        ]

    def count_user_docs(self, user_id: str) -> int:
        """Count total chunks for a user."""
        try:
            count = self.client.count(
                collection_name=USER_DOCS_COLLECTION,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]
                ),
            ).count
            return count
        except Exception:
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Database connection closed")

    async def async_close(self) -> None:
        """Close all database connections including async client."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
            logger.info("Async database connection closed")
        
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Sync database connection closed")


# Convenience function to get database instance
_db_instance: Optional[VectorDatabase] = None


def get_database() -> VectorDatabase:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = VectorDatabase()
    return _db_instance
