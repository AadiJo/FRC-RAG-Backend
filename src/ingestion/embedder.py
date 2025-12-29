"""
Embedding generation module.

Handles:
- Text embeddings with sentence-transformers (bge-large-en-v1.5)
- Image embeddings with CLIP (ViT-L/14)
- Batch processing with GPU memory management
- Export to Parquet format
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics

from .chunker import Chunk
from .image_processor import ProcessedImage

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    id: str
    embedding: List[float]
    model: str
    model_version: str
    embedding_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "embedding": self.embedding,
            "model": self.model,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "metadata": self.metadata,
        }


class TextEmbedder:
    """
    Text embedding generator using sentence-transformers.
    
    Default model: BAAI/bge-large-en-v1.5
    """

    def __init__(
        self,
        model_name: str = settings.text_embedding_model,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize text embedder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._model_version = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            logger.info(
                "Loading text embedding model",
                model=self.model_name,
                device=self.device,
            )
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            
            # Get model version from config (safely)
            try:
                first_module = self._model._first_module()
                if hasattr(first_module, "auto_model") and hasattr(first_module.auto_model, "config"):
                    self._model_version = str(first_module.auto_model.config._name_or_path)
                elif hasattr(first_module, "config"):
                    self._model_version = str(first_module.config.name_or_path)
                else:
                    self._model_version = self.model_name
            except Exception:
                self._model_version = self.model_name
            
            logger.info(
                "Text embedding model loaded",
                model=self.model_name,
                version=self._model_version,
                embedding_dim=self._model.get_sentence_embedding_dimension(),
            )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_model()
        
        # Normalize text
        text = text.strip()
        if not text:
            return [0.0] * self.embedding_dim
        
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        return embedding.tolist()

    def embed_texts(
        self, texts: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple texts with batching.
        
        Args:
            texts: List of input texts
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        # Normalize texts
        texts = [t.strip() if t else "" for t in texts]
        
        logger.info(
            "Generating text embeddings",
            count=len(texts),
            batch_size=self.batch_size,
        )
        
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        
        metrics.record_embeddings_generated(len(texts))
        
        return embeddings.tolist()

    def embed_chunks(
        self, chunks: List[Chunk], show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Embed chunks and return structured results.
        
        Args:
            chunks: List of chunks to embed
            show_progress: Show progress bar
            
        Returns:
            List of embedding results with metadata
        """
        # Flatten visual facts into text embeddings while keeping structured metadata
        texts = []
        for chunk in chunks:
            text = chunk.text or ""
            vf = getattr(chunk, "visual_facts", []) or []
            if vf:
                # Append a short, flattened representation of visual facts
                vf_flat = "; ".join(vf)
                text = f"{text}\n\nVisualFacts: {vf_flat}"
            texts.append(text)
        embeddings = self.embed_texts(texts, show_progress)
        
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append(EmbeddingResult(
                id=chunk.chunk_id,
                embedding=embedding,
                model=self.model_name,
                model_version=self._model_version or "unknown",
                embedding_dim=len(embedding),
                metadata=chunk.to_dict(),
            ))
        
        return results


class ImageEmbedder:
    """
    Image embedding generator using CLIP.
    
    Default model: ViT-L/14 (OpenAI pretrained)
    """

    def __init__(
        self,
        model_name: str = settings.image_embedding_model,
        pretrained: str = settings.image_embedding_pretrained,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        """
        Initialize image embedder.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights source
            device: Device to use
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the CLIP model."""
        if self._model is None:
            import open_clip
            
            logger.info(
                "Loading image embedding model",
                model=self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self._model.eval()
            
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            logger.info(
                "Image embedding model loaded",
                model=self.model_name,
            )

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        # CLIP ViT-L/14 has 768-dim embeddings
        return 768

    def embed_image(self, image: Union[Image.Image, Path, str]) -> List[float]:
        """
        Embed a single image.
        
        Args:
            image: PIL Image, or path to image
            
        Returns:
            Embedding vector
        """
        self._load_model()
        
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess and encode
        image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0].tolist()

    def embed_text(self, text: str) -> List[float]:
        """
        Embed text using CLIP text encoder for multimodal search.
        
        Args:
            text: Input query text
            
        Returns:
            Normalized embedding vector
        """
        self._load_model()
        
        # Tokenize and encode
        text_tokens = self._tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            embedding = self._model.encode_text(text_tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        return embedding.cpu().numpy()[0].tolist()

    def embed_images(
        self, images: List[Union[Image.Image, Path, str]], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple images with batching.
        
        Args:
            images: List of images or paths
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        logger.info(
            "Generating image embeddings",
            count=len(images),
            batch_size=self.batch_size,
        )
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Load and preprocess images
            processed = []
            for img in batch:
                try:
                    if isinstance(img, (str, Path)):
                        img = Image.open(img).convert("RGB")
                    elif img.mode != "RGB":
                        img = img.convert("RGB")
                    processed.append(self._preprocess(img))
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")
                    # Use zero vector for failed images
                    processed.append(torch.zeros(3, 224, 224))
            
            # Stack and encode
            batch_tensor = torch.stack(processed).to(self.device)
            
            with torch.no_grad():
                batch_embeddings = self._model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(
                    dim=-1, keepdim=True
                )
            
            embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            
            if show_progress and (i + self.batch_size) % 100 == 0:
                logger.debug(f"Processed {i + self.batch_size}/{len(images)} images")
        
        metrics.record_embeddings_generated(len(embeddings))
        
        return embeddings

    def embed_processed_images(
        self, images: List[ProcessedImage], show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Embed processed images and return structured results.
        
        Args:
            images: List of processed images
            show_progress: Show progress bar
            
        Returns:
            List of embedding results with metadata
        """
        # Filter out duplicates and missing files
        valid_images = [
            img for img in images
            if not img.is_duplicate and img.saved_path and img.saved_path.exists()
        ]
        
        if not valid_images:
            return []
        
        paths = [img.saved_path for img in valid_images]
        embeddings = self.embed_images(paths, show_progress)
        
        results = []
        for img, embedding in zip(valid_images, embeddings):
            results.append(EmbeddingResult(
                id=img.image_id,
                embedding=embedding,
                model=f"{self.model_name}_{self.pretrained}",
                model_version=self.pretrained,
                embedding_dim=len(embedding),
                metadata=img.to_dict(),
            ))
        
        return results

    def embed_processed_images_with_context(
        self,
        images: List[ProcessedImage],
        text_embedder: TextEmbedder,
        chunks: List[Chunk],
        captions: Optional[Dict[str, str]] = None,
        show_progress: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Embed processed images with surrounding text and captions combined.
        
        Creates a combined embedding by concatenating:
        - Image pixel embedding (CLIP)
        - Text embedding of surrounding text + caption (BGE)
        
        Args:
            images: List of processed images
            text_embedder: Text embedder for encoding surrounding text and captions
            chunks: List of chunks to find surrounding text
            captions: Optional dict mapping image_id to caption text
            show_progress: Show progress bar
            
        Returns:
            List of embedding results with combined embeddings
        """
        # Filter out duplicates and missing files
        valid_images = [
            img for img in images
            if not img.is_duplicate and img.saved_path and img.saved_path.exists()
        ]
        
        if not valid_images:
            return []
        
        # Build image_id -> chunk mapping
        image_to_chunk: Dict[str, Chunk] = {}
        for chunk in chunks:
            for image_id in chunk.image_ids:
                if image_id not in image_to_chunk:
                    image_to_chunk[image_id] = chunk
        
        # Get image embeddings (CLIP)
        paths = [img.saved_path for img in valid_images]
        image_embeddings = self.embed_images(paths, show_progress)
        
        # Get text embeddings for surrounding text + captions
        text_inputs = []
        for img in valid_images:
            # Get surrounding text from chunk
            surrounding_text = ""
            if img.image_id in image_to_chunk:
                chunk = image_to_chunk[img.image_id]
                # Extract original text (remove context prefix if present)
                text = chunk.text
                if text.startswith("[") and "\n" in text:
                    newline_idx = text.find("\n")
                    if newline_idx > 0 and text[newline_idx - 1] == "]":
                        text = text[newline_idx + 1:].strip()
                surrounding_text = text[:1000]  # Limit to first 1000 chars
            
            # Get caption
            caption = ""
            if captions and img.image_id in captions:
                caption = captions[img.image_id]
            
            # Combine surrounding text and caption
            combined_text = f"{surrounding_text}\n\nCaption: {caption}".strip()
            if not combined_text:
                combined_text = "Image"  # Fallback if no text available
            
            text_inputs.append(combined_text)
        
        # Generate text embeddings
        text_embeddings = text_embedder.embed_texts(text_inputs, show_progress=False)
        
        # Combine embeddings by concatenation
        image_dim = len(image_embeddings[0]) if image_embeddings else 768
        text_dim = len(text_embeddings[0]) if text_embeddings else text_embedder.embedding_dim
        
        results = []
        for img, img_emb, text_emb in zip(valid_images, image_embeddings, text_embeddings):
            # Concatenate image and text embeddings
            combined_embedding = list(img_emb) + list(text_emb)
            
            results.append(EmbeddingResult(
                id=img.image_id,
                embedding=combined_embedding,
                model=f"{self.model_name}_{self.pretrained}+{text_embedder.model_name}",
                model_version=f"{self.pretrained}+combined",
                embedding_dim=len(combined_embedding),
                metadata={
                    **img.to_dict(),
                    "embedding_type": "combined",
                    "image_dim": image_dim,
                    "text_dim": text_dim,
                },
            ))
        
        logger.info(
            "Generated combined embeddings",
            count=len(results),
            image_dim=image_dim,
            text_dim=text_dim,
            combined_dim=len(combined_embedding) if results else 0,
        )
        
        return results


class EmbeddingExporter:
    """Export embeddings to various formats."""

    @staticmethod
    def to_parquet(
        results: List[EmbeddingResult],
        output_path: Path,
        collection_name: str = "embeddings",
    ) -> Path:
        """
        Export embeddings to Parquet format.
        
        Args:
            results: List of embedding results
            output_path: Output file path
            collection_name: Name for the collection
            
        Returns:
            Path to created file
        """
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Convert to records
        records = []
        for result in results:
            record = {
                "id": result.id,
                "embedding": result.embedding,
                "model": result.model,
                "model_version": result.model_version,
                "embedding_dim": result.embedding_dim,
                **{f"meta_{k}": v for k, v in result.metadata.items() if not isinstance(v, (list, dict))},
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        
        logger.info(
            "Embeddings exported to Parquet",
            path=str(output_path),
            count=len(results),
        )
        
        return output_path

    @staticmethod
    def to_jsonl(
        results: List[EmbeddingResult],
        output_path: Path,
    ) -> Path:
        """
        Export embeddings to JSONL format.
        
        Args:
            results: List of embedding results
            output_path: Output file path
            
        Returns:
            Path to created file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")
        
        logger.info(
            "Embeddings exported to JSONL",
            path=str(output_path),
            count=len(results),
        )
        
        return output_path
