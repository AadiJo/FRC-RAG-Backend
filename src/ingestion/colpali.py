
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
import numpy as np
from colpali_engine.models import ColQwen2, ColQwen2Processor
from .image_processor import ProcessedImage
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

class ColPaliIngester:
    """
    Ingester for ColPali visual retrieval model.
    Generates multi-vector embeddings for page images.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.processor = None
        self.is_loaded = False
        
    def load_model(self):
        """Load ColPali model and processor."""
        if self.is_loaded:
            return

        try:
            logger.info(f"Loading ColPali model on {self.device}...")
            
            # Use ColQwen2 model with matching processor
            model_name = "vidore/colqwen2-v1.0"
            
            self.model = ColQwen2.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            ).eval()
            
            # Must use ColQwen2Processor (not ColPaliProcessor) for image_grid_thw support
            self.processor = ColQwen2Processor.from_pretrained(model_name)
            
            self.is_loaded = True
            logger.info("ColPali model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ColPali model: {e}")
            raise

    def embed_page(self, image: Image.Image) -> List[List[float]]:
        """
        Generate multi-vector embedding for a single page image.
        
        Returns:
            List of vectors (list of floats)
        """
        if not self.is_loaded:
            self.load_model()
            
        try:
            # Prepare image
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Process input
            batch_images = self.processor.process_images([image]).to(self.device)
            
            # Inference
            with torch.no_grad():
                # [1, num_patches, dim]
                image_embeddings = self.model(**batch_images)
                
            # Convert to list of lists (bag of vectors)
            # Remove batch dimension, convert bfloat16 to float32
            vectors = image_embeddings[0].cpu().float().numpy().tolist()
            return vectors
            
        except Exception as e:
            logger.error(f"Failed to embed page with ColPali: {e}")
            return []

    def embed_pages_batch(self, images: List[Image.Image]) -> List[List[List[float]]]:
        """
        Generate embeddings for a batch of images.
        """
        if not self.is_loaded:
            self.load_model()
            
        if not images:
            return []
            
        try:
            # Convert all to RGB
            rgb_images = [img.convert("RGB") for img in images]
            
            # Process inputs
            # Note: process_images can handle a list
            batch_images = self.processor.process_images(rgb_images).to(self.device)
            
            with torch.no_grad():
                # [batch_size, num_patches, dim]
                # Note: ColQwen might output class with .last_hidden_state or similar, 
                # strictly it returns the embeddings directly in v1.2 implementation usually
                embeddings = self.model(**batch_images)
            
            # Convert to CPU list structure
            batch_vectors = []
            for i in range(len(images)):
                # vectors for image i
                vecs = embeddings[i].cpu().float().numpy().tolist()
                batch_vectors.append(vecs)
                
            return batch_vectors
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return [[] for _ in images] # Return empty lists on failure

    def embed_query(self, query: str) -> List[List[float]]:
        """
        Generate multi-vector embedding for a text query.
        """
        if not self.is_loaded:
            self.load_model()
            
        try:
            # Process query
            batch_queries = self.processor.process_queries([query]).to(self.device)
            
            with torch.no_grad():
                # [1, num_patches, dim]
                query_embeddings = self.model(**batch_queries)
                
            # Convert to list of lists
            vectors = query_embeddings[0].cpu().float().numpy().tolist()
            return vectors
            
        except Exception as e:
            logger.error(f"Failed to embed query with ColPali: {e}")
            return []

    def unload(self):
        """Unload model to free VRAM."""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("ColPali model unloaded")
