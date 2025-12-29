"""
Query processing module.

Handles:
- Query preprocessing and normalization
- Hybrid search (text + image)
- Late fusion scoring
- Result formatting with pagination
- Confidence filtering
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from rank_bm25 import BM25Okapi

from src.ingestion.colpali import ColPaliIngester
from .database_setup import VectorDatabase, get_database
from .ingestion.embedder import TextEmbedder, ImageEmbedder
from .utils.config import settings
from .utils.logger import get_logger
from .utils.metrics import metrics

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Individual search result."""
    
    chunk_id: str
    text: str
    score: float
    page_number: int
    team: str
    year: str
    binder: str
    subsystem: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    source: str = "text"  # 'text' or 'image'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "page_number": self.page_number,
            "team": self.team,
            "year": self.year,
            "binder": self.binder,
            "subsystem": self.subsystem,
            "headers": self.headers,
            "image_ids": self.image_ids,
            "source": self.source,
        }


@dataclass
class ImageResult:
    """Image search result."""
    
    image_id: str
    score: float
    caption: Optional[str] = None
    url: Optional[str] = None
    page: int = 0
    team: str = ""
    year: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "score": self.score,
            "caption": self.caption,
            "url": self.url,
            "page": self.page,
            "team": self.team,
            "year": self.year,
        }


@dataclass
class QueryResponse:
    """Complete query response."""
    
    query_id: str
    query: str
    chunks: List[SearchResult]
    images: List[ImageResult]
    total_chunks: int
    total_images: int
    latency_ms: float
    visual_pages: List[SearchResult] = field(default_factory=list)
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "images": [i.to_dict() for i in self.images],
            "total_chunks": self.total_chunks,
            "total_images": self.total_images,
            "visual_pages": [v.to_dict() for v in self.visual_pages],
            "latency_ms": self.latency_ms,
            "filters_applied": self.filters_applied,
        }


class QueryProcessor:
    """
    Query processor for hybrid retrieval.
    
    Features:
    - Text preprocessing and normalization
    - Dense retrieval with embeddings
    - Late fusion of text and image results
    - Confidence filtering
    - Result pagination
    """

    def __init__(
        self,
        db: Optional[VectorDatabase] = None,
        text_embedder: Optional[TextEmbedder] = None,
        image_embedder: Optional[ImageEmbedder] = None,
        text_weight: float = settings.text_weight,
        image_weight: float = settings.image_weight,
        top_k: int = settings.retrieval_top_k,
    ):
        """
        Initialize query processor.
        
        Args:
            db: Vector database instance
            text_embedder: Text embedding model
            image_embedder: Image embedding model
            text_weight: Weight for text scores in fusion
            image_weight: Weight for image scores in fusion
            top_k: Default number of results
        """
        self.db = db or get_database()
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.top_k = top_k
        
        # Lazy-loaded embedders
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        
        # Cache for captions if they're not in DB
        self._captions_cache: Dict[str, str] = {}
        self._load_captions_cache()

        # Initialize BM25
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_documents: List[Dict[str, Any]] = []
        self._init_bm25()
        
        # Initialize Image Map for strict ID validation
        self._image_path_map: Dict[str, str] = {}
        self._load_image_map()

        # Initialize ColPali if visual retrieval is enabled (lazy load to save VRAM on startup)
        # We assume it's enabled if the collection exists, but we won't load the model until needed
        self.colpali: Optional[ColPaliIngester] = None
        self.visual_retrieval_enabled = False # Will check on first query

    def _load_image_map(self):
        """Load map of image_id -> relative_url for all valid images on disk."""
        try:
            images_path = Path(settings.images_path)
            if not images_path.exists():
                logger.warning(f"Images path does not exist: {images_path}")
                return

            logger.info("Indexing existing images for validation...")
            count = 0
            # Scan all files recursively
            for file_path in images_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    # Extract ID from stem (assuming filename is id.ext)
                    img_id = file_path.stem
                    # Create relative URL
                    try:
                        rel_path = file_path.relative_to(images_path)
                        url = f"/images/{rel_path}"
                        self._image_path_map[img_id] = url
                        count += 1
                    except Exception:
                        pass
            
            logger.info(f"Indexed {count} valid images for strict lookup.")
        except Exception as e:
            logger.error(f"Failed to load image map: {e}")

    def _init_bm25(self):
        """Initialize BM25 index from database chunks."""
        try:
            logger.info("Initializing BM25 index...")
            chunks = self.db.get_all_text_chunks()
            if not chunks:
                logger.warning("No chunks found for BM25 index")
                return
            
            tokenized_corpus = [
                self._normalize_query(chunk["text"]).split() 
                for chunk in chunks
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_documents = chunks
            
            logger.info(f"BM25 index initialized with {len(chunks)} documents")
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")

    def _load_captions_cache(self):
        """Load captions from local storage as backup."""
        try:
            captions_path = Path("data/output/captions.json")
            if captions_path.exists():
                import json
                with open(captions_path, "r") as f:
                    data = json.load(f)
                    for item in data:
                        if isinstance(item, dict) and "image_id" in item:
                            self._captions_cache[item["image_id"]] = item.get("final_caption", "")
                logger.info(f"Loaded {len(self._captions_cache)} captions into query processor cache.")
        except Exception as e:
            logger.warning(f"Failed to load captions cache: {e}")

    def _get_text_embedder(self) -> TextEmbedder:
        """Get or create text embedder (CPU mode for serving)."""
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(device="cpu")
        return self._text_embedder

    def _get_image_embedder(self) -> ImageEmbedder:
        """Get or create image embedder (CPU mode for serving)."""
        if self._image_embedder is None:
            self._image_embedder = ImageEmbedder(device="cpu")
        return self._image_embedder

    def _get_valid_image_url(self, image_id: str) -> Optional[str]:
        """
        Get valid URL for image ID using strict lookup.
        
        Args:
            image_id: Image identifier
            
        Returns:
            URL if known valid, None otherwise
        """
        return self._image_path_map.get(image_id)

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text.
        
        - Strip whitespace
        - Normalize case for matching
        - Expand common abbreviations
        """
        query = query.strip()
        
        # Expand common FRC abbreviations
        expansions = {
            r"\bneo\b": "NEO motor",
            r"\bfalcon\b": "Falcon 500 motor",
            r"\bdt\b": "drivetrain",
            r"\bcad\b": "CAD design",
            r"\bgb\b": "gearbox",
            r"\bpdp\b": "Power Distribution Panel",
            r"\bpdh\b": "Power Distribution Hub",
            r"\brio\b": "roboRIO",
        }
        
        query_lower = query.lower()
        for pattern, expansion in expansions.items():
            if re.search(pattern, query_lower):
                # Don't actually replace, just for better matching
                pass
        
        return query

    def _embed_query(self, query: str) -> List[float]:
        """Embed query text using text embedder."""
        embedder = self._get_text_embedder()
        return embedder.embed_text(query)

    def _search_text_collection(
        self,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search text collection."""
        results = self.db.search_text(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
        
        search_results = []
        seen_ids = set()
        
        for result in results:
            chunk_id = result.get("id", "")
            if not chunk_id:
                continue
                
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            
            # Handle metadata from different formats (prefer flat, fallback to nested)
            if "text" in result:
                chunk_data = result
            elif "metadata" in result and isinstance(result["metadata"], dict):
                chunk_data = result["metadata"]
            else:
                chunk_data = result
            
            search_results.append(SearchResult(
                chunk_id=chunk_id,
                text=chunk_data.get("text", ""),
                score=result.get("score", 0.0),
                page_number=chunk_data.get("page_number", 0),
                team=chunk_data.get("team", ""),
                year=chunk_data.get("year", ""),
                binder=chunk_data.get("binder", ""),
                subsystem=chunk_data.get("subsystem"),
                headers=chunk_data.get("headers", []),
                image_ids=chunk_data.get("image_ids", []),
                source="text",
            ))
        
        return search_results
    
    def _search_bm25(self, query: str, limit: int) -> List[SearchResult]:
        """Search using BM25."""
        if not self.bm25:
            return []
            
        tokenized_query = self._normalize_query(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        import numpy as np
        top_n = np.argsort(scores)[::-1][:limit]
        
        results = []
        for idx in top_n:
            if scores[idx] <= 0:
                continue
                
            chunk_data = self.bm25_documents[idx]
            
            results.append(SearchResult(
                chunk_id=chunk_data.get("id", ""),
                text=chunk_data.get("text", ""),
                score=float(scores[idx]), # Raw BM25 score
                page_number=chunk_data.get("page_number", 0),
                team=chunk_data.get("team", ""),
                year=chunk_data.get("year", ""),
                binder=chunk_data.get("binder", ""),
                subsystem=chunk_data.get("subsystem"),
                headers=chunk_data.get("headers", []),
                image_ids=chunk_data.get("image_ids", []),
                source="bm25",
            ))
            
        return results

    def _search_image_collection(
        self,
        query_vector: List[float],
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[ImageResult]:
        """Search image collection."""
        results = self.db.search_images(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
        
        image_results = []
        seen_ids = set()
        
        for result in results:
            image_id = result.get("id", "")
            if not image_id:
                continue
                
            # Skip duplicates (results are sorted by score)
            if image_id in seen_ids:
                continue
            seen_ids.add(image_id)
            
            # Handle metadata from different formats (prefer flat, fallback to nested)
            if "caption" in result:
                img_data = result
            elif "metadata" in result and isinstance(result["metadata"], dict):
                img_data = result["metadata"]
            else:
                img_data = result
            
            # Construct URL
            saved_path = img_data.get("saved_path")
            url = img_data.get("url")
            
            # Fallback to team/year structure if map lookup failed (unlikely if map is complete)
            if not url:
                url = self._get_valid_image_url(image_id)
            
            # If duplicate, use the original image's URL
            is_duplicate = img_data.get("is_duplicate", False)
            duplicate_of = img_data.get("duplicate_of")
            if not url and is_duplicate and duplicate_of:
                url = self._get_valid_image_url(duplicate_of)
                if url:
                    image_id = duplicate_of
            
            # Skip if we still don't have a valid URL
            if not url:
                continue

            # Get caption from database (prefer final_caption, then caption)
            # Filter out the prompt text if it's stored as caption
            prompt_text = "Describe this engineering image in detail. Focus on visible components, labels, and spatial relationships."
            db_caption = img_data.get("final_caption") or img_data.get("caption")
            
            # Filter out prompt text if it matches
            if db_caption and db_caption.strip() == prompt_text:
                db_caption = None
            
            # Use database caption, fallback to cache
            caption = db_caption or self._captions_cache.get(image_id)
            
            # Final check: filter out prompt text from cache too
            if caption and caption.strip() == prompt_text:
                caption = None
            
            image_results.append(ImageResult(
                image_id=image_id,
                score=result.get("score", 0.0),
                caption=caption,
                url=url,
                page=img_data.get("page", 0),
                team=str(img_data.get("team", "")),
                year=str(img_data.get("year", "")),
            ))
        
        return image_results

    def _fuse_results(
        self,
        text_results: List[SearchResult],
        bm25_results: List[SearchResult],
        image_results: List[ImageResult],
    ) -> Tuple[List[SearchResult], List[ImageResult]]:
        """
        Fuse Vector, BM25, and Image results using Reciprocal Rank Fusion (RRF).
        
        RRF Score = 1 / (k + rank)
        """
        k = 60
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        
        # Helper to process results
        def process_rankings(results: List[SearchResult], weight: float = 1.0):
            for rank, result in enumerate(results):
                if result.chunk_id not in chunk_map:
                    chunk_map[result.chunk_id] = result
                    fused_scores[result.chunk_id] = 0.0
                
                # RRF score accumulation
                fused_scores[result.chunk_id] += (weight * (1.0 / (k + rank)))

        # Process Vector results
        process_rankings(text_results)
        
        # Process BM25 results
        process_rankings(bm25_results)

        # Image scores map (for boosting)
        image_scores: Dict[str, float] = {
            img.image_id: img.score for img in image_results
        }
        
        # Create final list (with normalization)
        scaling_factor = 15.0  # Bring RRF scores (~0.03-0.06) to ~0.5-0.9 range
        
        final_results = []
        for chunk_id, score in fused_scores.items():
            result = chunk_map[chunk_id]
            
            # Boost based on relevant images (Visual Reranking)
            image_boost = 0.0
            for image_id in result.image_ids:
                if image_id in image_scores:
                    image_boost = max(image_boost, image_scores[image_id])
            
            if image_boost > 0:
                score += (self.image_weight * image_boost * 0.1)  # Scale down boost for RRF range
            
            # Normalize score
            result.score = score * scaling_factor
            
            final_results.append(result)
        
        # Sort by fused results
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results, image_results

    def _filter_low_confidence(
        self,
        results: List[SearchResult],
        min_score: float = 0.1,  # Normalized score threshold
        max_score_ratio: float = 0.3, # Relaxed from 0.5 to allow more results
    ) -> List[SearchResult]:
        """
        Filter out low confidence results.
        
        Rules:
        - Remove results below absolute threshold
        - Remove results that are much worse than the best
        """
        if not results:
            return results
        
        max_score = results[0].score
        
        filtered = [
            r for r in results
            if r.score >= min_score and r.score >= max_score * max_score_ratio
        ]
        
        # If filtering removed too many, keep at least top few
        if len(filtered) < 3 and len(results) >= 3:
            return results[:3]
        
        return filtered

    def _visual_search_colpali(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Perform visual search using ColPali.
        """
        if self.colpali is None:
            # Check if enabled
            try:
                db = get_database()
                cols = [c.name for c in db.client.get_collections().collections]
                if "frc_colpali" not in cols:
                     logger.debug("ColPali collection not found, visual search disabled")
                     return []
                
                # Load model
                logger.info("Loading ColPali for query...")
                self.colpali = ColPaliIngester(device="cuda") # Assume CUDA for serving if available
                self.colpali.load_model()
            except Exception as e:
                logger.error(f"Failed to init ColPali: {e}")
                return []
                
        try:
            # Embed query text
            query_vectors = self.colpali.embed_query(query)
            
            if not query_vectors:
                 return []
                 
            # Search in Qdrant
            db = get_database()
            results = db.search_colpali(query_vectors, limit=limit)
            
            search_results = []
            for res in results:
                # Convert to SearchResult
                # ColPali results are pages
                search_results.append(SearchResult(
                    chunk_id=res["id"],
                    score=res["score"],
                    text=f"[Visual Match] Page {res.get('page_number')} from {res.get('binder')}",
                    page_number=res.get("page_number", 0),
                    team=res.get("team", ""),
                    year=res.get("year", ""),
                    binder=res.get("binder", ""),
                    subsystem=res.get("subsystem", ""),
                    source="visual_colpali"
                ))
            
            return search_results
            
        except Exception as e:
             logger.error(f"Visual search failed: {e}")
             return []

    def search(
        self,
        query: str,
        limit: int = None,
        team: Optional[str] = None,
        year: Optional[str] = None,
        subsystem: Optional[str] = None,
        binder: Optional[str] = None,
        include_images: bool = True,
        min_score: float = 0.0,
        offset: int = 0,
    ) -> QueryResponse:
        """
        Perform hybrid search.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            team: Filter by team number
            year: Filter by year
            subsystem: Filter by subsystem
            binder: Filter by binder name
            include_images: Include image results
            min_score: Minimum score threshold
            offset: Result offset for pagination
            
        Returns:
            QueryResponse with chunks and images
        """
        query_id = str(uuid.uuid4())[:8]
        limit = limit or self.top_k
        start_time = time.perf_counter()
        
        logger.info(
            "Processing query",
            query_id=query_id,
            query=query[:100],
            limit=limit,
        )
        
        # Normalize query
        normalized_query = self._normalize_query(query)

        # Build filters
        filters = {}
        if team:
            filters["team"] = team
        if year:
            filters["year"] = year
        if subsystem:
            filters["subsystem"] = subsystem
        if binder:
            filters["binder"] = binder
        
        # Embed query
        query_vector = self._embed_query(normalized_query)
        
        # Search text collection
        text_results = self._search_text_collection(
            query_vector=query_vector,
            limit=limit + offset + 10,  # Get extra for filtering
            filters=filters if filters else None,
            score_threshold=min_score if min_score > 0 else None,
        )
        
        # Optionally search images
        image_results = []
        if include_images:
            # Check if collection uses combined embeddings
            try:
                collection_info = self.db.get_collection_info("frc_image_chunks")
                is_combined = collection_info and collection_info.vector_size > 768
                
                if is_combined:
                    # Create combined query vector (CLIP + BGE)
                    clip_embedder = self._get_image_embedder()
                    clip_query_vector = clip_embedder.embed_text(normalized_query)
                    bge_query_vector = self._embed_query(normalized_query)
                    
                    # Concatenate to match combined embedding structure
                    combined_query_vector = list(clip_query_vector) + list(bge_query_vector)
                    
                    logger.debug(
                        "Using combined embedding search",
                        clip_dim=len(clip_query_vector),
                        bge_dim=len(bge_query_vector),
                        combined_dim=len(combined_query_vector),
                    )
                    
                    image_results = self._search_image_collection(
                        query_vector=combined_query_vector,
                        limit=20,
                        filters=filters if filters else None,
                        score_threshold=0.2,
                    )
                else:
                    # Use CLIP only for standard image embeddings
                    clip_embedder = self._get_image_embedder()
                    image_query_vector = clip_embedder.embed_text(normalized_query)
                    
                    image_results = self._search_image_collection(
                        query_vector=image_query_vector,
                        limit=20,  # Fixed limit for direct image search
                        filters=filters if filters else None,
                        score_threshold=0.2, # Lower threshold for CLIP
                    )
            except Exception as e:
                logger.error(f"Image search failed: {e}")
                # Fallback: associated images
                seen_images = set()
                for result in text_results[:5]:
                    for image_id in result.image_ids:
                        if image_id not in seen_images:
                            seen_images.add(image_id)
                            image_results.append(ImageResult(
                                image_id=image_id,
                                score=result.score * 0.9,
                                page=result.page_number,
                                team=result.team,
                                year=result.year,
                            ))
        
        # Search BM25
        bm25_results = self._search_bm25(query=normalized_query, limit=limit + offset + 10)

        # ColPali Visual Search (Plan B)
        visual_results = []
        if include_images:
            visual_results = self._visual_search_colpali(normalized_query, limit=5)
            if visual_results:
                logger.info(f"Found {len(visual_results)} visual matches via ColPali")

        # Fuse results
        text_results, image_results = self._fuse_results(text_results, bm25_results, image_results)
        
        # Collect images from ALL text chunks (before pagination)
        # This ensures we get images from all matching documents, not just top-k
        seen_image_ids = {img.image_id for img in image_results}
        chunk_images = []
        
        for chunk in text_results:  # All results before pagination
            for image_id in chunk.image_ids:
                if image_id and image_id not in seen_image_ids:
                    seen_image_ids.add(image_id)
                    # Try to find the image file
                    url = self._get_valid_image_url(image_id)
                    if url:  # Only add if we can find the file
                        # Get caption from cache, filter out prompt text
                        prompt_text = "Describe this engineering image in detail. Focus on visible components, labels, and spatial relationships."
                        caption = self._captions_cache.get(image_id)
                        if caption and caption.strip() == prompt_text:
                            caption = None
                        
                        chunk_images.append(ImageResult(
                            image_id=image_id,
                            score=chunk.score * 0.85,  # Slightly lower than direct match
                            page=chunk.page_number,
                            team=chunk.team,
                            year=chunk.year,
                            url=url,
                            caption=caption,
                        ))
        
        # Merge chunk images with direct image search results
        image_results.extend(chunk_images)
        
        # Deduplicate by image_id (keep highest score)
        image_dict = {}
        for img in image_results:
            if img.image_id not in image_dict or img.score > image_dict[img.image_id].score:
                image_dict[img.image_id] = img
        
        # Sort by score and convert back to list
        image_results = sorted(image_dict.values(), key=lambda x: x.score, reverse=True)
        
        # Filter low confidence (adjust thresholds for RRF)
        # Scores are now normalized to ~0.5-1.0 range
        text_results = self._filter_low_confidence(text_results, min_score=0.05)
        
        # Apply pagination
        total_chunks = len(text_results)
        text_results = text_results[offset:offset + limit]
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        metrics.record_query_result(
            query_id=query_id,
            chunks_retrieved=len(text_results),
            images_retrieved=len(image_results),
            latency_ms=latency_ms,
        )
        
        logger.info(
            "Query completed",
            query_id=query_id,
            chunks=len(text_results),
            images=len(image_results),
            latency_ms=round(latency_ms, 2),
        )
        
        return QueryResponse(
            query_id=query_id,
            query=query,
            chunks=text_results,
            images=image_results[:20],  # Increased limit for more comprehensive results
            total_chunks=total_chunks,
            total_images=len(image_results),
            visual_pages=visual_results,
            latency_ms=latency_ms,
            filters_applied=filters,
        )

    def _search_user_docs(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 50,
    ) -> List[SearchResult]:
        """
        Search user documents collection.
        
        Args:
            query_vector: Query embedding vector
            user_id: User identifier for filtering
            limit: Maximum results
            
        Returns:
            List of SearchResult from user documents
        """
        try:
            results = self.db.search_user_docs(
                query_vector=query_vector,
                user_id=user_id,
                limit=limit,
            )
            
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    chunk_id=result.get("id", ""),
                    text=result.get("text", ""),
                    score=result.get("score", 0.0),
                    page_number=result.get("chunk_index", 0),  # Use chunk_index as page
                    team="",  # User docs don't have team
                    year="",  # User docs don't have year
                    binder=result.get("title", ""),  # Use title as binder
                    subsystem=None,
                    headers=[],
                    image_ids=[],
                    source="user_doc",
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"User doc search failed: {e}")
            return []

    def _fuse_with_user_docs(
        self,
        frc_results: List[SearchResult],
        user_doc_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Fuse FRC corpus results with user document results using RRF.
        
        Args:
            frc_results: Results from FRC corpus (already fused text+bm25)
            user_doc_results: Results from user documents
            
        Returns:
            Combined and re-ranked results with source_type preserved
        """
        if not user_doc_results:
            # Mark FRC results with source_type
            for r in frc_results:
                if r.source != "user_doc":
                    r.source = "frc_corpus"
            return frc_results
        
        k = 60
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        
        # Process FRC results
        for rank, result in enumerate(frc_results):
            chunk_map[result.chunk_id] = result
            result.source = "frc_corpus"
            fused_scores[result.chunk_id] = 1.0 / (k + rank)
        
        # Process user doc results
        for rank, result in enumerate(user_doc_results):
            if result.chunk_id not in chunk_map:
                chunk_map[result.chunk_id] = result
                fused_scores[result.chunk_id] = 0.0
            result.source = "user_doc"
            fused_scores[result.chunk_id] += 1.0 / (k + rank)
        
        # Create final sorted list
        scaling_factor = 15.0
        final_results = []
        
        for chunk_id, score in fused_scores.items():
            result = chunk_map[chunk_id]
            result.score = score * scaling_factor
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results

    def get_context_for_llm(
        self,
        query: str,
        max_chunks: int = 5,
        max_context_length: int = 4000,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get formatted context for LLM consumption.
        
        Args:
            query: Search query
            max_chunks: Maximum chunks to include
            max_context_length: Maximum total context length
            user_id: Optional user ID to include user documents in search
            **kwargs: Additional search parameters
            
        Returns:
            Dict with formatted context and metadata
        """
        # Get FRC corpus results
        response = self.search(query, limit=max_chunks * 2, **kwargs)
        
        # Get user doc results if user_id provided
        user_doc_results = []
        if user_id:
            query_vector = self._embed_query(query)
            user_doc_results = self._search_user_docs(
                query_vector=query_vector,
                user_id=user_id,
                limit=max_chunks,
            )
            logger.info(
                "User documents searched",
                user_id=user_id,
                num_results=len(user_doc_results),
            )
        
        # Fuse FRC + user docs
        all_chunks = self._fuse_with_user_docs(
            frc_results=response.chunks,
            user_doc_results=user_doc_results,
        )
        
        # Take top max_chunks
        all_chunks = all_chunks[:max_chunks]
        
        # Log chunks being used for context
        logger.info(
            "Chunks retrieved for context",
            query_id=response.query_id,
            num_chunks=len(all_chunks),
            frc_chunks=len([c for c in all_chunks if c.source == "frc_corpus"]),
            user_chunks=len([c for c in all_chunks if c.source == "user_doc"]),
        )
        print(f"\n[CONTEXT] Query ID: {response.query_id} | Chunks retrieved: {len(all_chunks)}")
        for i, chunk in enumerate(all_chunks):
            print(f"[CHUNK {i+1}] ID: {chunk.chunk_id} | Score: {chunk.score:.4f} | Source: {chunk.source} | Team: {chunk.team} | Year: {chunk.year} | Page: {chunk.page_number}")
            print(f"[CHUNK {i+1}] Text preview: {chunk.text[:200]}...")
            print("-" * 80)
        
        # Format chunks for LLM
        context_parts = []
        citations = []
        current_length = 0
        
        for i, chunk in enumerate(all_chunks):
            chunk_text = chunk.text
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_context_length:
                # Truncate if needed
                remaining = max_context_length - current_length
                if remaining > 100:
                    chunk_text = chunk_text[:remaining] + "..."
                else:
                    break
            
            # Insert inline image placeholders for images associated with this chunk
            image_placeholders = []
            # Deduplicate image IDs to avoid repeated placeholders
            unique_image_ids = sorted(list(set(chunk.image_ids)))
            for img_id in unique_image_ids:
                image_placeholders.append(f"[img:{img_id}]")
            
            if image_placeholders:
                chunk_text = chunk_text + "\n" + " ".join(image_placeholders)
            
            # Format with citation marker
            citation_id = f"[{i+1}]"
            context_parts.append(f"{citation_id} {chunk_text}")
            
            # Build citation with source_type
            citation = {
                "id": citation_id,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_number,
                "team": chunk.team,
                "year": chunk.year,
                "binder": chunk.binder,
                "source_type": chunk.source,
            }
            
            # Add user doc specific fields
            if chunk.source == "user_doc":
                citation["title"] = chunk.binder  # binder holds title for user docs
                citation["doc_id"] = "_".join(chunk.chunk_id.split("_")[:-2]) if "_chunk_" in chunk.chunk_id else chunk.chunk_id
            
            citations.append(citation)
            
            current_length += len(chunk_text) + len(citation_id) + 2
        
        context = "\n\n".join(context_parts)
        
        # Build image_map: placeholder -> {image_id, url, caption}
        image_map = {}
        for chunk in all_chunks:
            for img_id in chunk.image_ids:
                # Avoid re-checking already-added images
                if img_id in image_map:
                    continue

                # Construct team/year from pattern if available
                parts = img_id.split("_")
                team = parts[0] if parts else "unknown"
                year = parts[1] if len(parts) > 1 else "unknown"

                # Only include images that we can verify on disk
                url = self._get_valid_image_url(img_id)
                if not url:
                    logger.debug(
                        "Image file not found for chunk reference; skipping",
                        image_id=img_id,
                        team=team,
                        year=year,
                    )
                    continue

                # Get caption from cache, filter out prompt text
                prompt_text = "Describe this engineering image in detail. Focus on visible components, labels, and spatial relationships."
                caption = self._captions_cache.get(img_id)
                if caption and caption.strip() == prompt_text:
                    caption = None

                image_map[f"[img:{img_id}]"] = {
                    "image_id": img_id,
                    "url": url,
                    "caption": caption,
                }
        
        return {
            "context": context,
            "citations": citations,
            "images": [img.to_dict() for img in response.images],
            "image_map": image_map,
            "query_id": response.query_id,
            "total_chunks": len(all_chunks),
            "user_id": user_id,
        }

    def validate_citation(self, chunk_id: str) -> bool:
        """
        Validate that a citation exists.
        
        Args:
            chunk_id: Chunk identifier to validate
            
        Returns:
            True if chunk exists
        """
        chunk = self.db.get_chunk_by_id(chunk_id)
        return chunk is not None

    def validate_citations(self, chunk_ids: List[str]) -> Dict[str, bool]:
        """
        Validate multiple citations.
        
        Args:
            chunk_ids: List of chunk identifiers
            
        Returns:
            Dict mapping chunk_id to existence
        """
        return {cid: self.validate_citation(cid) for cid in chunk_ids}


# Convenience function
_processor_instance: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """Get or create query processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = QueryProcessor()
    return _processor_instance
