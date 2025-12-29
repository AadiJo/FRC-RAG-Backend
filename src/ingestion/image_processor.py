"""
Image processing module.

Handles:
- Image extraction from parsed documents
- Deduplication with perceptual hashing
- Resizing and format optimization
- Storage with organized directory structure
"""

import hashlib
import io
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
import imagehash
from PIL import Image

from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics

logger = get_logger(__name__)


@dataclass
class ProcessedImage:
    """Metadata for a processed image."""
    
    image_id: str
    original_path: Optional[Path] = None
    saved_path: Optional[Path] = None
    team: str = ""
    year: str = ""
    page: int = 0
    width: int = 0
    height: int = 0
    format: str = "png"
    size_bytes: int = 0
    perceptual_hash: str = ""
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "saved_path": str(self.saved_path) if self.saved_path else None,
            "team": self.team,
            "year": self.year,
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "perceptual_hash": self.perceptual_hash,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
        }


class ImageProcessor:
    """
    Image processor for FRC binder images.
    
    Features:
    - Extract images from PDFs
    - Perceptual hashing for deduplication
    - Automatic format selection (PNG/JPEG)
    - Resize to max dimensions
    - Organized storage structure
    """

    # Maximum image dimension
    MAX_SIZE = 1600
    
    # JPEG quality for photos
    JPEG_QUALITY = 85
    
    # Threshold for considering images as duplicates
    # Hash distance of 5 or less = likely duplicate
    HASH_THRESHOLD = 5

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        max_size: int = MAX_SIZE,
        deduplicate: bool = True,
    ):
        """
        Initialize image processor.
        
        Args:
            output_dir: Base directory for saving images
            max_size: Maximum dimension (width or height)
            deduplicate: Enable perceptual hash deduplication
        """
        self.output_dir = Path(output_dir or settings.images_path)
        self.max_size = max_size
        self.deduplicate = deduplicate
        
        # Hash registry for deduplication
        self._hash_registry: Dict[str, str] = {}  # hash -> image_id
        self._processed_images: Dict[str, ProcessedImage] = {}

    def _get_output_path(
        self, image_id: str, team: str, year: str, format: str
    ) -> Path:
        """
        Get output path following folder structure.
        
        Structure: data/images/frc/<team>/<year>/<image_id>.<ext>
        """
        ext = "jpg" if format.lower() in ("jpeg", "jpg") else "png"
        path = self.output_dir / "frc" / team / year / f"{image_id}.{ext}"
        return path

    def _compute_perceptual_hash(self, img: Image.Image) -> str:
        """
        Compute perceptual hash for deduplication.
        
        Uses average hash which is faster and good enough for our needs.
        """
        try:
            phash = imagehash.average_hash(img)
            return str(phash)
        except Exception as e:
            logger.warning("Failed to compute perceptual hash", error=str(e))
            # Fall back to content hash
            return ""

    def _is_duplicate(self, phash: str) -> Tuple[bool, Optional[str]]:
        """
        Check if image is a duplicate based on perceptual hash.
        
        Returns:
            Tuple of (is_duplicate, original_image_id if duplicate)
        """
        if not self.deduplicate or not phash:
            return False, None
        
        # Check against all existing hashes
        for existing_hash, image_id in self._hash_registry.items():
            try:
                existing = imagehash.hex_to_hash(existing_hash)
                current = imagehash.hex_to_hash(phash)
                distance = existing - current
                
                if distance <= self.HASH_THRESHOLD:
                    logger.debug(
                        "Duplicate image detected",
                        hash_distance=distance,
                        original=image_id,
                    )
                    return True, image_id
                    
            except Exception:
                continue
        
        return False, None

    def _select_format(self, img: Image.Image) -> str:
        """
        Select optimal format based on image content.
        
        - PNG for images with transparency or few colors (diagrams)
        - JPEG for photos (many colors, no transparency)
        """
        # Check for transparency
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            return "png"
        
        # Check color diversity (rough heuristic)
        # Diagrams tend to have fewer unique colors
        small = img.copy()
        small.thumbnail((100, 100))
        colors = small.convert("RGB").getcolors(maxcolors=1000)
        
        if colors is not None and len(colors) < 256:
            return "png"  # Likely a diagram
        
        return "jpeg"  # Likely a photo

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image if exceeds max dimensions."""
        width, height = img.size
        
        if width <= self.max_size and height <= self.max_size:
            return img
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = self.max_size
            new_height = int(height * (self.max_size / width))
        else:
            new_height = self.max_size
            new_width = int(width * (self.max_size / height))
        
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        logger.debug(
            "Image resized",
            original=f"{width}x{height}",
            new=f"{new_width}x{new_height}",
        )
        
        return resized

    def _strip_metadata(self, img: Image.Image) -> Image.Image:
        """Remove EXIF and other metadata from image."""
        # Create new image without metadata
        data = list(img.getdata())
        img_clean = Image.new(img.mode, img.size)
        img_clean.putdata(data)
        return img_clean

    def _save_image(
        self,
        img: Image.Image,
        output_path: Path,
        format: str,
    ) -> int:
        """
        Save image to disk.
        
        Returns:
            File size in bytes
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't overwrite existing images
        if output_path.exists():
            logger.debug("Image already exists", path=str(output_path))
            return output_path.stat().st_size
        
        save_kwargs = {}
        
        if format.lower() in ("jpeg", "jpg"):
            # Convert to RGB if needed (JPEG doesn't support transparency)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            save_kwargs["quality"] = self.JPEG_QUALITY
            save_kwargs["optimize"] = True
        else:
            save_kwargs["optimize"] = True
        
        img.save(output_path, format=format.upper(), **save_kwargs)
        
        return output_path.stat().st_size

    def process_image_bytes(
        self,
        image_bytes: bytes,
        image_id: str,
        team: str,
        year: str,
        page: int = 0,
    ) -> ProcessedImage:
        """
        Process raw image bytes.
        
        Args:
            image_bytes: Raw image data
            image_id: Unique identifier for the image
            team: Team number
            year: Year
            page: Source page number
            
        Returns:
            ProcessedImage with metadata
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            
            # Compute perceptual hash before processing
            phash = self._compute_perceptual_hash(img)
            
            # Check for duplicates
            is_dup, dup_of = self._is_duplicate(phash)
            
            if is_dup:
                return ProcessedImage(
                    image_id=image_id,
                    team=team,
                    year=year,
                    page=page,
                    width=img.width,
                    height=img.height,
                    perceptual_hash=phash,
                    is_duplicate=True,
                    duplicate_of=dup_of,
                )
            
            # Process image
            img = self._resize_image(img)
            img = self._strip_metadata(img)
            
            # Select format
            format = self._select_format(img)
            
            # Save
            output_path = self._get_output_path(image_id, team, year, format)
            size_bytes = self._save_image(img, output_path, format)
            
            # Register hash
            if phash:
                self._hash_registry[phash] = image_id
            
            result = ProcessedImage(
                image_id=image_id,
                saved_path=output_path,
                team=team,
                year=year,
                page=page,
                width=img.width,
                height=img.height,
                format=format,
                size_bytes=size_bytes,
                perceptual_hash=phash,
                is_duplicate=False,
            )
            
            self._processed_images[image_id] = result
            
            logger.debug(
                "Image processed",
                image_id=image_id,
                size=f"{img.width}x{img.height}",
                format=format,
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Image processing failed",
                image_id=image_id,
                error=str(e),
            )
            metrics.record_ingestion_error(
                "image_processing_error",
                str(e),
                image_id,
            )
            raise

    def extract_from_pdf(
        self,
        pdf_path: Path,
        team: Optional[str] = None,
        year: Optional[str] = None,
    ) -> List[ProcessedImage]:
        """
        Extract and process all images from a PDF.
        
        Args:
            pdf_path: Path to PDF file
            team: Team number (extracted from filename if not provided)
            year: Year (extracted from filename if not provided)
            
        Returns:
            List of processed images
        """
        pdf_path = Path(pdf_path)
        
        # Parse team/year from filename if not provided
        if not team or not year:
            name = pdf_path.stem
            parts = name.split("-")
            team = team or parts[0]
            year = year or (parts[-1] if len(parts) > 1 else "unknown")
        
        logger.info(
            "Extracting images from PDF",
            filename=pdf_path.name,
            team=team,
            year=year,
        )
        
        doc = fitz.open(pdf_path)
        images: List[ProcessedImage] = []
        extracted_count = 0
        deduplicated_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    
                    if not base_image:
                        continue
                    
                    image_bytes = base_image["image"]
                    
                    # Generate image ID
                    content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    image_id = f"{team}_{year}_p{page_num}_i{img_index}_{content_hash}"
                    
                    # Process image
                    processed = self.process_image_bytes(
                        image_bytes=image_bytes,
                        image_id=image_id,
                        team=team,
                        year=year,
                        page=page_num,
                    )
                    
                    images.append(processed)
                    extracted_count += 1
                    
                    if processed.is_duplicate:
                        deduplicated_count += 1
                        
                except Exception as e:
                    logger.warning(
                        "Failed to extract image",
                        page=page_num,
                        image_index=img_index,
                        error=str(e),
                    )
        
        doc.close()
        
        # Record metrics
        unique_count = extracted_count - deduplicated_count
        metrics.record_images_extracted(unique_count, deduplicated_count)
        
        logger.info(
            "Image extraction complete",
            filename=pdf_path.name,
            total_extracted=extracted_count,
            unique=unique_count,
            deduplicated=deduplicated_count,
        )
        
        return images

    def extract_from_all_pdfs(
        self, input_dir: Path
    ) -> Dict[str, List[ProcessedImage]]:
        """
        Extract images from all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            
        Returns:
            Dictionary mapping filename to list of processed images
        """
        input_dir = Path(input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))
        
        logger.info(
            "Starting batch image extraction",
            input_dir=str(input_dir),
            file_count=len(pdf_files),
        )
        
        results: Dict[str, List[ProcessedImage]] = {}
        
        for pdf_path in pdf_files:
            try:
                images = self.extract_from_pdf(pdf_path)
                results[pdf_path.name] = images
            except Exception as e:
                logger.error(
                    "Failed to extract images from PDF",
                    filename=pdf_path.name,
                    error=str(e),
                )
                results[pdf_path.name] = []
        
        # Summary
        total_images = sum(len(imgs) for imgs in results.values())
        unique_images = sum(
            1 for imgs in results.values()
            for img in imgs
            if not img.is_duplicate
        )
        
        logger.info(
            "Batch extraction complete",
            total_images=total_images,
            unique_images=unique_images,
            documents=len(results),
        )
        
        return results

    def get_image_url(
        self, image_id: str, base_url: Optional[str] = None
    ) -> Optional[str]:
        """
        Get public URL for an image.
        
        Args:
            image_id: Image identifier
            base_url: Base URL for image hosting
            
        Returns:
            Public URL or None if image not found
        """
        processed = self._processed_images.get(image_id)
        
        if not processed or not processed.saved_path:
            return None
        
        # Get relative path from images directory
        try:
            rel_path = processed.saved_path.relative_to(self.output_dir)
        except ValueError:
            rel_path = processed.saved_path.name
        
        base = base_url or settings.get_image_url_base()
        return f"{base}/{rel_path}"

    def get_all_images(self) -> List[ProcessedImage]:
        """Get all processed images."""
        return list(self._processed_images.values())

    def load_existing_images(self) -> List[ProcessedImage]:
        """
        Load existing images from disk without extracting new ones.
        
        Scans the images directory and reconstructs ProcessedImage objects
        from existing files.
        
        Returns:
            List of ProcessedImage objects for existing images
        """
        images = []
        images_dir = self.output_dir / "frc"
        
        if not images_dir.exists():
            logger.info("No existing images directory found")
            return []
        
        logger.info("Loading existing images from disk", base_dir=str(images_dir))
        
        # Scan directory structure: frc/<team>/<year>/*.jpg, *.png
        for team_dir in images_dir.iterdir():
            if not team_dir.is_dir():
                continue
            
            team = team_dir.name
            
            for year_dir in team_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                year = year_dir.name
                
                # Find all image files
                for img_file in year_dir.glob("*.jpg"):
                    images.append(self._load_image_from_path(img_file, team, year, "jpg"))
                for img_file in year_dir.glob("*.png"):
                    images.append(self._load_image_from_path(img_file, team, year, "png"))
        
        # Filter out duplicates (images without saved_path are duplicates)
        valid_images = [img for img in images if img.saved_path and img.saved_path.exists()]
        
        logger.info(
            "Loaded existing images",
            total=len(images),
            valid=len(valid_images),
        )
        
        return valid_images
    
    def _load_image_from_path(
        self, img_path: Path, team: str, year: str, format: str
    ) -> ProcessedImage:
        """
        Reconstruct ProcessedImage from file path.
        
        Args:
            img_path: Path to image file
            team: Team number
            year: Year
            format: Image format (jpg/png)
            
        Returns:
            ProcessedImage object
        """
        # Extract image_id from filename (remove extension)
        image_id = img_path.stem
        
        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            logger.warning(f"Failed to read image dimensions for {img_path}: {e}")
            width, height = 0, 0
        
        # Get file size
        size_bytes = img_path.stat().st_size if img_path.exists() else 0
        
        # Try to extract page number from image_id if it follows the pattern
        # Pattern: <team>_<year>_p<page>_i<index>_<hash>
        page = 0
        try:
            parts = image_id.split("_")
            for part in parts:
                if part.startswith("p") and part[1:].isdigit():
                    page = int(part[1:])
                    break
        except Exception:
            pass
        
        return ProcessedImage(
            image_id=image_id,
            saved_path=img_path,
            team=team,
            year=year,
            page=page,
            width=width,
            height=height,
            format=format,
            size_bytes=size_bytes,
            perceptual_hash="",  # Not computed for existing images
            is_duplicate=False,
        )

    def clear_cache(self) -> None:
        """Clear the hash registry and processed images cache."""
        self._hash_registry.clear()
        self._processed_images.clear()
        logger.info("Image processor cache cleared")
