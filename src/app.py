"""
FastAPI application for FRC RAG backend.

Provides:
- Query endpoint with filtering
- Health check
- Static image hosting
- API key authentication (optional)
- Rate limiting
- CORS configuration
- Ngrok tunnel for development
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .database_setup import VectorDatabase, get_database
from .query_processor import QueryProcessor, get_query_processor
from .utils.config import settings
from .utils.logger import get_logger, setup_logging
from .utils.metrics import metrics

# Initialize logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    json_format=not settings.is_development,
    is_development=settings.is_development,
)

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# API key security (optional)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Request/Response Models
# ============================================================================


class QueryRequest(BaseModel):
    """Query request body."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset for pagination")
    team: Optional[str] = Field(default=None, description="Filter by team number")
    year: Optional[str] = Field(default=None, description="Filter by year")
    subsystem: Optional[str] = Field(default=None, description="Filter by subsystem")
    binder: Optional[str] = Field(default=None, description="Filter by binder name")
    include_images: bool = Field(default=True, description="Include image results")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")


class ChunkResponse(BaseModel):
    """Individual chunk in response."""
    
    chunk_id: str
    text: str
    score: float
    page_number: int
    team: str
    year: str
    binder: str
    subsystem: Optional[str]
    headers: List[str]
    image_ids: List[str]


class ImageResponse(BaseModel):
    """Individual image in response."""
    
    image_id: str
    score: float
    caption: Optional[str]
    url: Optional[str]
    page: int
    team: str
    year: str


class QueryResponse(BaseModel):
    """Query response body."""
    
    query_id: str
    query: str
    chunks: List[ChunkResponse]
    images: List[ImageResponse]
    total_chunks: int
    total_images: int
    visual_pages: List[ChunkResponse] = Field(default_factory=list)
    latency_ms: float
    filters_applied: Dict[str, Any]
    images_skipped: bool = False


class ContextRequest(BaseModel):
    """Context request for LLM."""
    
    query: str = Field(..., min_length=1, max_length=1000)
    max_chunks: int = Field(default=5, ge=1, le=20)
    max_context_length: int = Field(default=4000, ge=100, le=16000)
    team: Optional[str] = None
    year: Optional[str] = None
    subsystem: Optional[str] = None
    user_id: Optional[str] = Field(default=None, description="User ID for including user documents in search")


class ContextResponse(BaseModel):
    """Formatted context for LLM."""
    
    context: str
    citations: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    image_map: Dict[str, Dict[str, Any]]  # [img:id] -> {image_id, url, caption}
    query_id: str
    total_chunks: int
    images_skipped: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    database: Dict[str, Any]
    environment: str


class CitationValidationRequest(BaseModel):
    """Citation validation request."""
    
    chunk_ids: List[str]


class CitationValidationResponse(BaseModel):
    """Citation validation response."""
    
    results: Dict[str, bool]


# ============================================================================
# User Document Models
# ============================================================================


class DocumentSource(BaseModel):
    """Source information for a user document."""
    type: str = Field(..., description="Source type: gdrive, manual, etc.")
    uri: Optional[str] = Field(default=None, description="Source URI")


class UserDocument(BaseModel):
    """A user document to upsert."""
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    text: str = Field(..., min_length=1, description="Full text content")
    source: DocumentSource
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    strategy: str = Field(default="recursive")
    chunk_size: int = Field(default=900, ge=100, le=2000)
    chunk_overlap: int = Field(default=150, ge=0, le=500)


class UserDocumentUpsertRequest(BaseModel):
    """Request to upsert user documents."""
    user_id: str = Field(..., min_length=1, description="User identifier")
    documents: List[UserDocument] = Field(..., min_length=1)
    chunking: Optional[ChunkingConfig] = Field(default=None)


class UpsertedDocument(BaseModel):
    """Result for a successfully upserted document."""
    doc_id: str
    chunks_created: int
    bytes_indexed: int


class FailedDocument(BaseModel):
    """Result for a failed document."""
    doc_id: str
    error: Dict[str, Any]


class UserDocumentUpsertResponse(BaseModel):
    """Response from upserting user documents."""
    user_id: str
    upserted: List[UpsertedDocument]
    failed: List[FailedDocument]


class UserDocumentDeleteRequest(BaseModel):
    """Request to delete user documents."""
    user_id: str = Field(..., min_length=1)
    doc_ids: List[str] = Field(..., min_length=1)


class UserDocumentDeleteResponse(BaseModel):
    """Response from deleting user documents."""
    user_id: str
    deleted: List[str]
    not_found: List[str]


# ============================================================================
# Application Setup
# ============================================================================


# Tunnel manager for ngrok
tunnel_url: Optional[str] = None


def start_tunnel() -> Optional[str]:
    """Start ngrok tunnel if configured."""
    global tunnel_url
    
    if not settings.should_use_tunnel:
        return None
    
    try:
        from pyngrok import ngrok
        
        # Set auth token
        ngrok.set_auth_token(settings.ngrok_auth_token)
        
        # Start tunnel
        tunnel = ngrok.connect(settings.server_port, "http")
        tunnel_url = tunnel.public_url
        
        logger.info(
            "Ngrok tunnel started",
            url=tunnel_url,
        )
        
        return tunnel_url
        
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        return None


def stop_tunnel():
    """Stop ngrok tunnel."""
    global tunnel_url
    
    if tunnel_url:
        try:
            from pyngrok import ngrok
            ngrok.disconnect(tunnel_url)
            ngrok.kill()
            logger.info("Ngrok tunnel stopped")
        except Exception as e:
            logger.warning(f"Error stopping tunnel: {e}")
        
        tunnel_url = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(
        "Starting FRC RAG API",
        environment=settings.environment,
        host=settings.server_host,
        port=settings.server_port,
    )
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize database
    try:
        db = get_database()
        db.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Start tunnel if configured
    start_tunnel()
    
    if tunnel_url:
        logger.info(f"Public URL: {tunnel_url}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FRC RAG API")
    
    stop_tunnel()
    
    # Export metrics
    try:
        metrics.export_metrics(Path("logs/metrics.json"))
    except Exception as e:
        logger.warning(f"Failed to export metrics: {e}")


# Create app
app = FastAPI(
    title="FRC RAG Backend",
    description="Multimodal RAG system for FRC team binders",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependencies
# ============================================================================


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Verify API key if required."""
    if not settings.api_key_required:
        return None
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
        )
    
    if api_key not in settings.api_keys_list:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return api_key


def get_processor() -> QueryProcessor:
    """Get query processor dependency."""
    return get_query_processor()


def get_db() -> VectorDatabase:
    """Get database dependency."""
    return get_database()


def get_public_url() -> Optional[str]:
    """Get public URL if available."""
    return tunnel_url


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check(db: VectorDatabase = Depends(get_db)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database=db.get_stats(),
        environment=settings.environment,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
@app.post("/api/v1/rag/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    processor: QueryProcessor = Depends(get_processor),
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Search the FRC binder database.
    
    Returns relevant chunks and images based on the query.
    Supports filtering by team, year, subsystem, and binder.
    """
    # Log the exact request payload
    logger.info(
        "Incoming search request",
        payload=body.model_dump(),
        client_host=request.client.host if request.client else "unknown"
    )
    print(f"\n[REQUEST] /api/v1/query | Payload: {body.model_dump()}")

    try:
        # Debug: classification of incoming query (helpful when frontend sends system prompts)
        try:
            classification = processor.classify_query(body.query, include_images=body.include_images)
            logger.debug("Query classification", **classification)
            # Also print to stdout for easy copy/paste
            print("[DEBUG QUERY CLASSIFICATION]", classification)
        except Exception:
            classification = None

        result = processor.search(
            query=body.query,
            limit=body.limit,
            offset=body.offset,
            team=body.team,
            year=body.year,
            subsystem=body.subsystem,
            binder=body.binder,
            include_images=body.include_images,
            min_score=body.min_score,
        )
        
        # Build chunks list and log each chunk
        chunk_responses = []
        for c in result.chunks:
            chunk_response = ChunkResponse(
                chunk_id=c.chunk_id,
                text=c.text,
                score=c.score,
                page_number=c.page_number,
                team=c.team,
                year=c.year,
                binder=c.binder,
                subsystem=c.subsystem,
                headers=c.headers,
                image_ids=c.image_ids,
            )
            chunk_responses.append(chunk_response)
            
            # Print chunk being sent to frontend
            logger.info(
                "Sending chunk to frontend",
                chunk_id=chunk_response.chunk_id,
                score=chunk_response.score,
                page_number=chunk_response.page_number,
                team=chunk_response.team,
                year=chunk_response.year,
                binder=chunk_response.binder,
                subsystem=chunk_response.subsystem,
                text_preview=chunk_response.text[:100] + "..." if len(chunk_response.text) > 100 else chunk_response.text,
                headers=chunk_response.headers,
                image_ids=chunk_response.image_ids,
            )
            print(f"[CHUNK] ID: {chunk_response.chunk_id} | Score: {chunk_response.score:.4f} | Team: {chunk_response.team} | Year: {chunk_response.year} | Page: {chunk_response.page_number}")
            print(f"[CHUNK] Text preview: {chunk_response.text[:200]}...")
            print(f"[CHUNK] Headers: {chunk_response.headers}")
            print(f"[CHUNK] Image IDs: {chunk_response.image_ids}")
            print("-" * 80)
        # Debug: summary of images returned
        try:
            image_ids = [i.image_id for i in result.images]
            logger.debug("Query image summary", image_count=len(image_ids), image_ids=image_ids)
            print("[DEBUG QUERY IMAGES] count=", len(image_ids), "ids=", image_ids)
        except Exception:
            pass
        
        return QueryResponse(
            query_id=result.query_id,
            query=result.query,
            chunks=chunk_responses,
            images=[
                ImageResponse(
                    image_id=i.image_id,
                    score=i.score,
                    caption=i.caption,
                    url=i.url,
                    page=i.page,
                    team=i.team,
                    year=i.year,
                )
                for i in result.images
            ],
            total_chunks=result.total_chunks,
            total_images=result.total_images,
            visual_pages=[
                ChunkResponse(
                    chunk_id=v.chunk_id,
                    text=v.text,
                    score=v.score,
                    page_number=v.page_number,
                    team=v.team,
                    year=v.year,
                    binder=v.binder,
                    subsystem=v.subsystem,
                    headers=v.headers,
                    image_ids=v.image_ids,
                )
                for v in result.visual_pages
            ],
            latency_ms=result.latency_ms,
            filters_applied=result.filters_applied,
            images_skipped=getattr(result, "images_skipped", False),
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/context", response_model=ContextResponse)
@app.post("/api/v1/rag/context_fused", response_model=ContextResponse)
async def get_context(
    request: Request,
    body: ContextRequest,
    processor: QueryProcessor = Depends(get_processor),
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Get formatted context for LLM consumption.
    
    Returns a formatted string with citations that can be used
    directly in an LLM prompt.
    """
    # Log the exact request payload
    logger.info(
        "Incoming context request",
        payload=body.model_dump(),
        client_host=request.client.host if request.client else "unknown"
    )
    print(f"\n[REQUEST] {request.url.path} | Payload: {body.model_dump()}")

    try:
        result = processor.get_context_for_llm(
            query=body.query,
            max_chunks=body.max_chunks,
            max_context_length=body.max_context_length,
            team=body.team,
            year=body.year,
            subsystem=body.subsystem,
            user_id=body.user_id,  # Include user documents if user_id provided
        )
        # Debug: classification for context request
        try:
            classification = processor.classify_query(body.query, include_images=True)
            logger.debug("Context classification", **classification)
            print("[DEBUG CONTEXT CLASSIFICATION]", classification)
        except Exception:
            pass
        
        # Log chunks being used for context (they're in the citations)
        logger.info(
            "Sending context to frontend",
            query_id=result["query_id"],
            total_chunks=result["total_chunks"],
            num_citations=len(result["citations"]),
        )
        print(f"\n[CONTEXT] Query ID: {result['query_id']} | Total chunks: {result['total_chunks']}")
        print(f"[CONTEXT] Citations ({len(result['citations'])}):")
        for citation in result["citations"]:
            print(f"  - {citation['id']}: Chunk {citation['chunk_id']} | Team: {citation['team']} | Year: {citation['year']} | Page: {citation['page']}")
        print("-" * 80)
        
        return ContextResponse(
            context=result["context"],
            citations=result["citations"],
            images=result["images"],
            image_map=result.get("image_map", {}),
            query_id=result["query_id"],
            total_chunks=result["total_chunks"],
            images_skipped=result.get("images_skipped", False),
        )
        
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/debug_query")
async def debug_query(
    body: QueryRequest,
    processor: QueryProcessor = Depends(get_processor),
    _: Optional[str] = Depends(verify_api_key),
):
    """Debug endpoint returning how the backend classifies a query.

    Returns: extracted user message, visual/non-visual flags, and whether images would be fetched.
    """
    try:
        info = processor.classify_query(body.query, include_images=body.include_images)
        return info
    except Exception as e:
        logger.error(f"Debug query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/validate-citations", response_model=CitationValidationResponse)
async def validate_citations(
    body: CitationValidationRequest,
    processor: QueryProcessor = Depends(get_processor),
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Validate that citation IDs exist in the database.
    
    Used to verify LLM-generated citations are valid.
    """
    try:
        results = processor.validate_citations(body.chunk_ids)
        return CitationValidationResponse(results=results)
    except Exception as e:
        logger.error(f"Citation validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/chunk/{chunk_id}")
async def get_chunk(
    chunk_id: str,
    db: VectorDatabase = Depends(get_db),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get a specific chunk by ID."""
    chunk = db.get_chunk_by_id(chunk_id)
    
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    
    return chunk


@app.get("/api/v1/stats")
async def get_stats(
    db: VectorDatabase = Depends(get_db),
    _: Optional[str] = Depends(verify_api_key),
):
    """Get database and query statistics."""
    return {
        "database": db.get_stats(),
        "queries": metrics.get_query_stats(),
        "ingestion": metrics.get_ingestion_stats(),
    }


@app.get("/api/v1/public-url")
async def get_public_url_endpoint(public_url: Optional[str] = Depends(get_public_url)):
    """Get public URL if ngrok tunnel is active."""
    if public_url:
        return {"url": public_url}
    return {"url": None, "message": "No tunnel active"}


# ============================================================================
# User Document Endpoints
# ============================================================================


@app.post("/api/v1/user-documents/upsert", response_model=UserDocumentUpsertResponse)
async def upsert_user_documents(
    request: Request,
    body: UserDocumentUpsertRequest,
    db: VectorDatabase = Depends(get_db),
    processor: QueryProcessor = Depends(get_processor),
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Upsert user documents to the vector store.
    
    Chunks and embeds each document using the existing text embedder,
    then stores in the user_docs collection with user_id filtering.
    """
    from .ingestion.text_chunker import TextChunker
    
    logger.info(
        "User document upsert request",
        user_id=body.user_id,
        num_documents=len(body.documents),
    )
    
    # Get chunking config
    chunk_config = body.chunking or ChunkingConfig()
    chunker = TextChunker(
        chunk_size=chunk_config.chunk_size,
        chunk_overlap=chunk_config.chunk_overlap,
    )
    
    upserted = []
    failed = []
    
    for doc in body.documents:
        try:
            # Chunk the document
            chunks = chunker.chunk_text(
                text=doc.text,
                doc_id=doc.doc_id,
                user_id=body.user_id,
                title=doc.title,
                source_type=doc.source.type,
                source_uri=doc.source.uri,
                metadata=doc.metadata,
            )
            
            if not chunks:
                failed.append(FailedDocument(
                    doc_id=doc.doc_id,
                    error={"code": "NO_CHUNKS", "message": "Document produced no chunks"},
                ))
                continue
            
            # Generate embeddings for all chunks
            chunk_texts = [c.text for c in chunks]
            embeddings = []
            
            # Get embedder from processor
            embedder = processor._get_text_embedder()
            
            for text in chunk_texts:
                embedding = embedder.embed_text(text)
                embeddings.append(embedding)
            
            # Prepare chunks for database
            db_chunks = []
            total_bytes = 0
            
            for chunk, embedding in zip(chunks, embeddings):
                total_bytes += len(chunk.text.encode('utf-8'))
                
                db_chunks.append({
                    "id": chunk.id,
                    "embedding": embedding,
                    "text": chunk.text,
                    "user_id": chunk.user_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "source_type": chunk.source_type,
                    "source_uri": chunk.source_uri,
                    **(chunk.metadata or {}),
                })
            
            # Upsert to database
            num_upserted = db.upsert_user_docs(db_chunks)
            
            upserted.append(UpsertedDocument(
                doc_id=doc.doc_id,
                chunks_created=num_upserted,
                bytes_indexed=total_bytes,
            ))
            
            logger.info(
                "User document upserted",
                user_id=body.user_id,
                doc_id=doc.doc_id,
                chunks=num_upserted,
            )
            
        except Exception as e:
            logger.error(f"Failed to upsert document {doc.doc_id}: {e}", exc_info=True)
            failed.append(FailedDocument(
                doc_id=doc.doc_id,
                error={"code": "EMBEDDING_FAILED", "message": str(e)},
            ))
    
    return UserDocumentUpsertResponse(
        user_id=body.user_id,
        upserted=upserted,
        failed=failed,
    )


@app.post("/api/v1/user-documents/delete", response_model=UserDocumentDeleteResponse)
async def delete_user_documents(
    body: UserDocumentDeleteRequest,
    db: VectorDatabase = Depends(get_db),
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Delete user documents from the vector store.
    
    Requires both user_id and doc_ids for multi-tenant security.
    """
    logger.info(
        "User document delete request",
        user_id=body.user_id,
        doc_ids=body.doc_ids,
    )
    
    try:
        result = db.delete_user_docs(body.user_id, body.doc_ids)
        
        return UserDocumentDeleteResponse(
            user_id=body.user_id,
            deleted=result["deleted"],
            not_found=result["not_found"],
        )
        
    except Exception as e:
        logger.error(f"Failed to delete user documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files for images
if settings.images_path.exists():
    app.mount(
        "/images",
        StaticFiles(directory=str(settings.images_path)),
        name="images",
    )
    logger.info(f"Mounted static images from {settings.images_path}")


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc),
        exc_info=True,
    )
    
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": type(exc).__name__},
        )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


def run_server():
    """Run the server using uvicorn."""
    import signal
    import uvicorn
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("Received interrupt signal, shutting down...")
        raise KeyboardInterrupt
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        uvicorn.run(
            "src.app:app",
            host=settings.server_host,
            port=settings.server_port,
            reload=settings.is_development,
            log_level=settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        raise


if __name__ == "__main__":
    run_server()
