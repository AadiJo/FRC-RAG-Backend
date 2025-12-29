"""
Ingestion pipeline script.

Orchestrates the full ingestion process:
1. Parse PDF documents
2. Extract and process images
3. Chunk documents
4. Generate captions for images
5. Generate embeddings
6. Ingest into vector database
"""

import argparse
import json
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database_setup import VectorDatabase
from src.ingestion import (
    DocumentParser,
    DocumentChunker,
    ImageProcessor,
    ImageCaptioner,
    TextEmbedder,
    ImageEmbedder,
    EmbeddingExporter,
)
from src.ingestion.colpali import ColPaliIngester
from pdf2image import convert_from_path
from src.utils.config import settings
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import metrics

# Initialize logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
    json_format=False,
    is_development=True,
)

logger = get_logger(__name__)



def parse_worker(
    pdf_path: Path, 
    output_dir: Path, 
    use_ocr: bool, 
    extract_tables: bool, 
    extract_images: bool
):
    """Worker function for parallel PDF parsing."""
    try:
        # Check cache
        output_path = output_dir / "parsed" / f"{pdf_path.stem}.json"
        if output_path.exists():
            with open(output_path, "r") as f:
                doc_dict = json.load(f)
                from src.ingestion.parser import ParsedDocument
                return ParsedDocument.from_dict(doc_dict), True  # True = cached
        
        # Parse
        from src.ingestion.parser import DocumentParser
        parser = DocumentParser(
            use_ocr=use_ocr,
            extract_tables=extract_tables,
            extract_images=extract_images,
        )
        doc = parser.parse(pdf_path)
        
        # Save
        with open(output_path, "w") as f:
            json.dump(doc.to_dict(), f, indent=2)
            
        return doc, False  # False = newly parsed
        
    except Exception as e:
        return None, str(e)

def format_error(e):
    return f"{type(e).__name__}: {str(e)}"

def extract_images_worker(pdf_path: Path, output_dir: Path):
    """Worker function for parallel image extraction."""
    try:
        from src.ingestion.image_processor import ImageProcessor
        processor = ImageProcessor(output_dir=output_dir)
        return processor.extract_from_pdf(pdf_path)
    except Exception as e:
        return str(e)


class IngestionPipeline:
    """
    Full ingestion pipeline for FRC binders.
    
    Orchestrates all ingestion steps from PDF to vector database.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        use_gpu: bool = True,
        skip_captions: bool = False,
        skip_images: bool = False,
        skip_extraction: bool = False,
        ignore_caption_validation: bool = False,
        use_combined_embeddings: bool = False,
        use_colpali: bool = False,
        resume: bool = False,
    ):
        """
        Initialize pipeline.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for intermediate outputs
            use_gpu: Use GPU for embeddings (if available)
            skip_captions: Skip caption generation (faster)
            skip_images: Skip image processing entirely
            skip_extraction: Skip image extraction but still process existing images
            ignore_caption_validation: Embed all images even if captions failed validation
            use_combined_embeddings: Embed images with surrounding text + caption + image pixels
            use_colpali: Enable ColPali visual retrieval indexing
            resume: Try to pick up from a previous partially completed run
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir or self.input_dir / "output")
        self.use_gpu = use_gpu
        self.skip_captions = skip_captions
        self.skip_images = skip_images
        self.skip_extraction = skip_extraction
        self.ignore_caption_validation = ignore_caption_validation
        self.use_combined_embeddings = use_combined_embeddings
        self.use_colpali = use_colpali
        self.resume = resume
        
        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "parsed").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        
        # Initialize components
        device = "cuda" if use_gpu else "cpu"
        
        self.parser = DocumentParser(
            use_ocr=True,
            extract_tables=True,
            extract_images=not skip_images and not skip_extraction,
        )
        
        self.chunker = DocumentChunker()
        
        if not skip_images:
            self.image_processor = ImageProcessor(
                output_dir=settings.images_path,
            )
            
            if not skip_captions:
                self.captioner = ImageCaptioner(device=device)
            else:
                self.captioner = None
        else:
            self.image_processor = None
            self.captioner = None
        
        self.text_embedder = TextEmbedder(device=device)
        
        if not skip_images:
            self.image_embedder = ImageEmbedder(device=device)
        else:
            self.image_embedder = None
        
        self.db = VectorDatabase()
        
        if self.use_colpali:
            self.colpali = ColPaliIngester(device=device)
        else:
            self.colpali = None

    def _backup_existing_database(self) -> Optional[Path]:
        """
        Backup existing database if it exists.
        
        Returns:
            Path to backup if created, None if no backup needed
        """
        db_path = Path(settings.db_path)
        
        # Check if database directory exists and has content
        if not db_path.exists():
            logger.debug("No existing database found, skipping backup")
            return None
        
        if self.resume:
            logger.info("Resume mode active: skipping database backup/deletion.")
            return None
        
        # Check if directory has any Qdrant collections (has subdirectories or files)
        try:
            # Qdrant creates a .qdrant directory or collection files
            has_content = any(db_path.iterdir())
            if not has_content:
                logger.debug("Database directory exists but is empty, skipping backup")
                return None
        except Exception:
            # If we can't check, assume it has content
            has_content = True
        
        if not has_content:
            return None
        
        # Create backups directory
        backups_dir = db_path.parent / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"db_backup_{timestamp}"
        backup_path = backups_dir / backup_name
        
        try:
            # Copy entire database directory to backup location
            logger.info(
                "Backing up existing database",
                source=str(db_path),
                backup=str(backup_path),
            )
            
            shutil.copytree(db_path, backup_path, dirs_exist_ok=False)
            
            logger.info(
                "Database backup created successfully",
                backup_path=str(backup_path),
            )
            
            # Remove the existing database directory so a fresh one can be created
            logger.info("Removing existing database directory for fresh ingestion")
            shutil.rmtree(db_path)
            
            return backup_path
            
        except Exception as e:
            logger.error(
                f"Failed to backup database: {e}",
                exc_info=True,
            )
            # Don't fail the entire ingestion if backup fails
            # But warn the user
            logger.warning(
                "Continuing without backup. Existing database may be overwritten."
            )
            return None

    def run(self) -> Dict[str, int]:
        """
        Run the full ingestion pipeline.
        
        Returns:
            Statistics about the ingestion
        """
        run_id = f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Starting ingestion pipeline",
            run_id=run_id,
            input_dir=str(self.input_dir),
        )
        
        # Start metrics tracking
        metrics.start_ingestion_run(
            run_id=run_id,
            model_info={
                "text_embedding": settings.text_embedding_model,
                "image_embedding": settings.image_embedding_model,
                "vision_model": settings.vision_model if not self.skip_captions else "skipped",
            },
        )
        
        try:
            # Step 0: Backup existing database if it exists
            logger.info("Step 0: Checking for existing database")
            backup_path = self._backup_existing_database()
            if backup_path:
                logger.info(f"Existing database backed up to: {backup_path}")
            
            # Step 1: Parse documents
            logger.info("Step 1: Parsing documents")
            parsed_docs = self._parse_documents()
            
            # Step 2: Process images
            if not self.skip_images:
                if self.skip_extraction:
                    logger.info("Step 2: Loading existing images (skipping extraction)")
                    all_images = self._load_existing_images()
                else:
                    logger.info("Step 2: Processing images")
                    all_images = self._process_images()
            else:
                logger.info("Step 2: Skipping image processing")
                all_images = []
            
            # Step 3: Chunk documents
            logger.info("Step 3: Chunking documents")
            all_chunks = self._chunk_documents(parsed_docs)
            
            # Step 4: Generate captions
            if self.captioner and all_images:
                logger.info("Step 4: Generating captions")
                captions = self._generate_captions(all_images, parsed_docs)
            else:
                logger.info("Step 4: Skipping caption generation")
                captions = []
            
            # Step 5: Generate embeddings
            logger.info("Step 5: Generating embeddings")
            text_embeddings, image_embeddings = self._generate_embeddings(
                all_chunks, all_images, captions
            )
            
            # Step 6: Ingest into database
            logger.info("Step 6: Ingesting into database")
            self._ingest_to_database(text_embeddings, image_embeddings)
            
            # Step 7: ColPali Processing
            if self.colpali:
                logger.info("Step 7: Processing ColPali visual embeddings")
                self._process_colpali(parsed_docs)
            
            # Complete metrics
            run = metrics.end_ingestion_run(success=True)
            
            stats = {
                "documents_processed": len(parsed_docs),
                "chunks_created": len(all_chunks),
                "images_processed": len(all_images),
                "text_embeddings": len(text_embeddings),
                "image_embeddings": len(image_embeddings),
                "duration_ms": run.total_duration_ms if run else 0,
            }
            
            logger.info(
                "Ingestion complete",
                **stats,
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            metrics.record_ingestion_error("pipeline_error", str(e))
            metrics.end_ingestion_run(success=False)
            raise

    def _parse_documents(self) -> List:
        """Parse all PDF documents in parallel."""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        parsed = []
        
        # Determine max workers for CPU-bound parsing
        # Leave some cores free for system/other tasks
        max_workers = max(1, (multiprocessing.cpu_count() or 2) - 1)
        
        logger.info(f"Parsing documents with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    parse_worker,
                    pdf_path,
                    self.output_dir,
                    True, # use_ocr
                    True, # extract_tables
                    not self.skip_images and not self.skip_extraction # extract_images
                ): pdf_path
                for pdf_path in pdf_files
            }
            
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    doc, is_cached_or_error = future.result()
                    
                    if isinstance(is_cached_or_error, str):
                        # It's an error message
                        logger.error(f"Failed to parse {pdf_path.name}: {is_cached_or_error}")
                        metrics.record_document_processed(success=False)
                    else:
                        # Success
                        parsed.append(doc)
                        metrics.record_document_processed(success=True)
                        is_cached = is_cached_or_error
                        if is_cached:
                            logger.info(f"Loaded cached parsed data for {pdf_path.name}")
                        else:
                            logger.info(
                                f"Parsed {pdf_path.name}",
                                pages=doc.total_pages,
                                errors=len(doc.parse_errors),
                            )
                except Exception as e:
                    logger.error(f"Worker failed for {pdf_path.name}: {e}")
                    metrics.record_document_processed(success=False)
        
        return parsed

    def _process_images(self) -> List:
        """Extract and process images from all PDFs in parallel."""
        # Use settings image path if valid, otherwise fallback
        images_output_dir = Path(settings.images_path)
        
        all_images = []
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        max_workers = max(1, (multiprocessing.cpu_count() or 2) - 1)
        logger.info(f"Extracting images with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    extract_images_worker,
                    pdf,
                    images_output_dir
                ): pdf
                for pdf in pdf_files
            }
            
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    result = future.result()
                    if isinstance(result, str):
                        # Error
                        logger.error(f"Image extraction failed for {pdf.name}: {result}")
                    else:
                        # Success (list of images)
                        all_images.extend(result)
                except Exception as e:
                    logger.error(f"Worker failed for {pdf.name}: {e}")
        
        # Filter unique images
        unique_images_map = {}
        for img in all_images:
            if not img.is_duplicate:
                unique_images_map[img.image_id] = img
        
        unique_images = list(unique_images_map.values())
        
        logger.info(
            f"Processed {len(all_images)} images, {len(unique_images)} unique"
        )
        
        return unique_images
    
    def _load_existing_images(self) -> List:
        """Load existing images from disk without extraction."""
        if not self.image_processor:
            return []
        
        images = self.image_processor.load_existing_images()
        
        logger.info(f"Loaded {len(images)} existing images from disk")
        
        return images

    def _chunk_documents(self, parsed_docs: List) -> List:
        """Chunk all parsed documents."""
        all_chunks = []
        
        for doc in parsed_docs:
            try:
                chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Chunking failed for {doc.filename}: {e}")
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        return all_chunks

    def _generate_captions(self, images: List, parsed_docs: List) -> List:
        """Generate captions for images with resumability."""
        if not self.captioner:
            return []
        
        # Build context map from parsed docs
        context_map = {}
        for doc in parsed_docs:
            for page in doc.pages:
                for img_ref in page.images:
                    context_map[img_ref.image_id] = page.raw_text[:1000]
        
        captions_path = self.output_dir / "captions.json"
        existing_captions = []
        existing_ids = set()
        
        if captions_path.exists():
            try:
                logger.info(f"Checking for existing captions in {captions_path}")
                with open(captions_path, "r") as f:
                    cached_data = json.load(f)
                    from src.ingestion.captioner import ImageCaption
                    existing_captions = [ImageCaption(**c) for c in cached_data]
                    existing_ids = {c.image_id for c in existing_captions}
                logger.info(f"Loaded {len(existing_captions)} existing captions from cache.")
            except Exception as e:
                logger.warning(f"Failed to load existing captions, starting fresh: {e}")

        # Identify missing images
        missing_images = [img for img in images if img.image_id not in existing_ids]
        
        if not missing_images:
            logger.info("All images already have captions in cache.")
            return [c for c in existing_captions if c.image_id in {img.image_id for img in images}]

        logger.info(f"Need to generate captions for {len(missing_images)} / {len(images)} total images.")
        
        # Process in chunks to allow incremental saving (checkpoints)
        # 100 images per save point
        checkpoint_size = 100
        new_captions = []
        
        try:
            for i in range(0, len(missing_images), checkpoint_size):
                chunk = missing_images[i:i + checkpoint_size]
                logger.info(f"Processing caption batch: {i//checkpoint_size + 1} (images {i} to {min(i+checkpoint_size, len(missing_images))})")
                
                chunk_results = self.captioner.caption_processed_images(
                    chunk,
                    context_map=context_map,
                    show_progress=True
                )
                
                new_captions.extend(chunk_results)
                
                # Save checkpoint
                combined = existing_captions + new_captions
                with open(captions_path, "w") as f:
                    json.dump([c.to_dict() for c in combined], f, indent=2)
                
                logger.info(f"Checkpoint saved: {len(combined)} total captions stored.")
            
            return existing_captions + new_captions
            
        except Exception as e:
            logger.error(f"Caption generation failed during processing: {e}")
            # Even if it fails mid-run, return what we have so far
            return existing_captions + new_captions

    def _generate_embeddings(self, chunks: List, images: List, captions: List = None):
        """Generate embeddings for chunks and images."""
        # If captions provided, merge caption visual facts into chunks deterministically
        if captions:
            # Build image_id -> caption text mapping
            caption_map = {}
            caption_uncert_map = {}
            for cap in captions:
                # Prefer final_caption, fall back to raw_visual_facts
                cap_text = getattr(cap, 'final_caption', None) or getattr(cap, 'raw_visual_facts', None) or ""
                if cap_text:
                    caption_map[getattr(cap, 'image_id')] = cap_text
                # capture validation notes as uncertainties
                caption_uncert_map[getattr(cap, 'image_id')] = getattr(cap, 'validation_notes', []) or []

            # Attach captions to chunks that reference images
            try:
                for chunk in chunks:
                    vf = getattr(chunk, 'visual_facts', None)
                    if vf is None:
                        chunk.visual_facts = []
                    unc = getattr(chunk, 'uncertainties', None)
                    if unc is None:
                        chunk.uncertainties = []
                    for image_id in getattr(chunk, 'image_ids', []) or []:
                        if image_id in caption_map:
                            # Append caption (shortened) as a visual_fact
                            snippet = caption_map[image_id]
                            # keep snippet reasonably short
                            if len(snippet) > 512:
                                snippet = snippet[:512]
                            if snippet not in chunk.visual_facts:
                                chunk.visual_facts.append(snippet)
                        # attach uncertainties
                        notes = caption_uncert_map.get(image_id, [])
                        for n in notes:
                            if n and n not in chunk.uncertainties:
                                chunk.uncertainties.append(n)
            except Exception:
                logger.warning("Failed to merge captions into chunks, continuing without merging")

        # Text embeddings
        logger.info(f"Generating text embeddings for {len(chunks)} chunks")
        text_results = self.text_embedder.embed_chunks(chunks)
        
        # Export text embeddings
        text_path = self.output_dir / "embeddings" / "text_embeddings.parquet"
        EmbeddingExporter.to_parquet(text_results, text_path)
        
        # Image embeddings
        image_results = []
        if self.image_embedder and images:
            # Filter images based on captions validation if provided and validation not ignored
            images_to_embed = images
            if captions and not self.ignore_caption_validation:
                valid_ids = {c.image_id for c in captions if c.validation_passed}
                images_to_embed = [img for img in images if img.image_id in valid_ids]
                logger.info(f"Filtered {len(images) - len(images_to_embed)} rejected images (validation failed)")
            elif captions and self.ignore_caption_validation:
                logger.info("Ignoring caption validation - embedding all images")
            
            if images_to_embed:
                if self.use_combined_embeddings:
                    # Use combined embeddings (image + surrounding text + caption)
                    logger.info(f"Generating combined embeddings for {len(images_to_embed)} images")
                    
                    # Build caption dict
                    caption_dict = {}
                    if captions:
                        for caption in captions:
                            # Filter out prompt text if it matches
                            prompt_text = "Describe this engineering image in detail. Focus on visible components, labels, and spatial relationships."
                            final_caption = caption.final_caption
                            if final_caption and final_caption.strip() == prompt_text:
                                final_caption = None
                            if final_caption:
                                caption_dict[caption.image_id] = final_caption
                    
                    image_results = self.image_embedder.embed_processed_images_with_context(
                        images=images_to_embed,
                        text_embedder=self.text_embedder,
                        chunks=chunks,
                        captions=caption_dict if caption_dict else None,
                    )
                else:
                    # Use standard image-only embeddings
                    logger.info(f"Generating image embeddings for {len(images_to_embed)} images")
                    image_results = self.image_embedder.embed_processed_images(images_to_embed)
            
            # Export image embeddings
            image_path = self.output_dir / "embeddings" / "image_embeddings.parquet"
            EmbeddingExporter.to_parquet(image_results, image_path)
        
        return text_results, image_results

    def _ingest_to_database(self, text_embeddings: List, image_embeddings: List):
        """Ingest embeddings into vector database."""
        # Determine image embedding dimension from results
        image_embedding_dim = None
        if image_embeddings:
            image_embedding_dim = len(image_embeddings[0].embedding)
            logger.info(
                "Detected image embedding dimension",
                dimension=image_embedding_dim,
            )
        
        # Initialize database with correct dimension
        self.db.initialize(image_embedding_dim=image_embedding_dim)
        
        # Convert to dict format for ingestion
        text_records = [
            {
                "id": e.id,
                "embedding": e.embedding,
                # Preserve lists for headers, image ids, visual_facts, and uncertainties
                **{k: v for k, v in e.metadata.items() if not isinstance(v, (list, dict)) or k in ["headers", "image_ids", "visual_facts", "uncertainties"]},
            }
            for e in text_embeddings
        ]
        
        image_records = [
            {
                "id": e.id,
                "embedding": e.embedding,
                **{k: v for k, v in e.metadata.items() if not isinstance(v, (list, dict))},
            }
            for e in image_embeddings
        ]
        
        # Upsert
        text_count = self.db.upsert_text_chunks(text_records)
        image_count = self.db.upsert_image_chunks(image_records) if image_records else 0
        
        logger.info(
            "Database ingestion complete",
            text_chunks=text_count,
            image_chunks=image_count,
        )

    def _process_colpali(self, parsed_docs: List):
        """Process documents with ColPali."""
        # Process each document
        total_upserted = 0
        
        for doc in parsed_docs:
            try:
                # Skip if already in DB and resume is on
                if self.resume and self.db.check_colpali_pdf_exists(doc.filename):
                    logger.info(f"Skipping ColPali for {doc.filename} (already indexed)")
                    continue

                pdf_path = self.input_dir / doc.filename
                if not pdf_path.exists():
                    logger.warning(f"PDF not found for ColPali: {pdf_path}")
                    continue
                    
                logger.info(f"Rendering pages for {doc.filename}...")
                # Convert PDF pages to images
                # 150 DPI is usually sufficient for ColPali and faster than 300
                images = convert_from_path(str(pdf_path), dpi=150)
                
                logger.info(f"Generating ColPali embeddings for {len(images)} pages...")
                
                # Batch process pages
                batch_size = 4 # Keep small for VRAM safety
                pages_data = []
                
                for i in range(0, len(images), batch_size):
                    batch_imgs = images[i:i+batch_size]
                    batch_embeddings = self.colpali.embed_pages_batch(batch_imgs)
                    
                    for j, embedding in enumerate(batch_embeddings):
                        if not embedding:
                            continue
                            
                        page_num = i + j + 1
                        page_id = f"{doc.filename}_p{page_num}"
                        
                        pages_data.append({
                            "id": page_id,
                            "multivector": embedding,
                            "binder": doc.filename,
                            "page_number": page_num,
                            "team": doc.metadata.get("team", ""),
                            "year": doc.metadata.get("year", ""),
                        })
                
                # Upsert to DB
                if pages_data:
                    count = self.db.upsert_colpali_pages(pages_data)
                    total_upserted += count
                    logger.info(f"Upserted {count} pages for {doc.filename}")
                    
            except Exception as e:
                logger.error(f"ColPali processing failed for {doc.filename}: {e}")
                
        logger.info(f"ColPali processing complete. Total pages: {total_upserted}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest FRC binder PDFs into the RAG system"
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Directory containing PDF files (default: data)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for intermediate files",
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="Skip image caption generation (faster)",
    )
    
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip all image processing",
    )
    
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip image extraction but still process embeddings for existing images",
    )
    
    parser.add_argument(
        "--ignore-caption-validation",
        action="store_true",
        help="Embed all images even if captions failed validation",
    )
    
    parser.add_argument(
        "--use-combined-embeddings",
        action="store_true",
        help="Embed images with surrounding text + caption + image pixels (combined embedding)",
    )
    
    parser.add_argument(
        "--colpali",
        action="store_true",
        help="Enable ColPali visual retrieval indexing (requires GPU)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume ingestion from a previous partially completed run",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        setup_logging(log_level="DEBUG", json_format=False, is_development=True)
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Validate flags
    if args.skip_images and args.skip_extraction:
        logger.warning("--skip-images and --skip-extraction both set. --skip-images takes precedence.")
        args.skip_extraction = False
    
    # Run pipeline
    pipeline = IngestionPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        skip_captions=args.skip_captions,
        skip_images=args.skip_images,
        skip_extraction=args.skip_extraction,
        ignore_caption_validation=args.ignore_caption_validation,
        use_combined_embeddings=args.use_combined_embeddings,
        use_colpali=args.colpali,
        resume=args.resume,
    )
    
    try:
        stats = pipeline.run()
        
        print("\n" + "=" * 50)
        print("Ingestion Complete!")
        print("=" * 50)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
