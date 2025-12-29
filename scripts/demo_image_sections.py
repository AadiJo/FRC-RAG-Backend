#!/usr/bin/env python3
"""
Demonstration script for image embedding with section context.

Shows:
1. How images are extracted from PDF
2. How chunks are created with section boundaries
3. How images are associated with chunks/sections
4. What context information is used for image captions
5. What metadata is stored with image embeddings
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    DocumentParser,
    DocumentChunker,
    ImageProcessor,
    ImageCaptioner,
    ImageEmbedder,
)
from src.utils.config import settings
from src.utils.logger import get_logger, setup_logging

# Initialize logging
setup_logging(
    log_level="INFO",
    log_file=None,
    json_format=False,
    is_development=True,
)

logger = get_logger(__name__)


def find_first_pdf(data_dir: Path) -> Path:
    """Find the first PDF file in the data directory."""
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    return pdf_files[0]


def extract_team_year_from_filename(filename: str) -> tuple:
    """Extract team and year from filename if possible."""
    # Try common patterns: team_year.pdf, team-year.pdf, etc.
    parts = filename.replace(".pdf", "").replace("_", "-").split("-")
    if len(parts) >= 2:
        try:
            team = parts[0]
            year = parts[1]
            return team, year
        except:
            pass
    return "unknown", "unknown"


def demonstrate_image_sections(pdf_path: Path):
    """Demonstrate how images are associated with sections."""
    
    print("=" * 80)
    print("IMAGE EMBEDDING WITH SECTION CONTEXT DEMONSTRATION")
    print("=" * 80)
    print(f"\nPDF: {pdf_path.name}")
    
    # Extract team/year from filename
    team, year = extract_team_year_from_filename(pdf_path.name)
    print(f"Team: {team}, Year: {year}\n")
    
    # Step 1: Parse document
    print("=" * 80)
    print("STEP 1: PARSING DOCUMENT")
    print("=" * 80)
    
    parser = DocumentParser(
        use_ocr=True,
        extract_tables=True,
        extract_images=True,
    )
    
    doc = parser.parse(pdf_path)
    print(f"\n✓ Parsed {doc.total_pages} pages")
    print(f"✓ Found {len(doc.pages)} pages with content")
    
    # Count images
    total_images = sum(len(page.images) for page in doc.pages)
    print(f"✓ Found {total_images} image references")
    
    # Step 2: Process images
    print("\n" + "=" * 80)
    print("STEP 2: PROCESSING IMAGES")
    print("=" * 80)
    
    image_processor = ImageProcessor(
        output_dir=settings.images_path,
    )
    
    processed_images = image_processor.extract_from_pdf(pdf_path)
    unique_images = [img for img in processed_images if not img.is_duplicate]
    
    print(f"\n✓ Processed {len(processed_images)} images")
    print(f"✓ {len(unique_images)} unique images (after deduplication)")
    print(f"✓ {len(processed_images) - len(unique_images)} duplicates removed")
    
    # Step 3: Chunk document with sections
    print("\n" + "=" * 80)
    print("STEP 3: CHUNKING WITH SECTION BOUNDARIES")
    print("=" * 80)
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(doc)
    
    print(f"\n✓ Created {len(chunks)} chunks")
    
    # Analyze chunk-image associations
    print("\n" + "-" * 80)
    print("CHUNK-IMAGE ASSOCIATIONS:")
    print("-" * 80)
    
    chunks_with_images = [c for c in chunks if c.image_ids]
    print(f"\n✓ {len(chunks_with_images)} chunks have associated images")
    print(f"✓ {len(chunks) - len(chunks_with_images)} chunks have no images")
    
    # Show first few chunks with images
    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS WITH IMAGES:")
    print("=" * 80)
    
    for i, chunk in enumerate(chunks_with_images[:5], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Section Index: {chunk.section_index}")
        print(f"Headers: {chunk.headers}")
        print(f"Associated Image IDs: {chunk.image_ids}")
        print(f"Subsystem: {chunk.subsystem or 'None'}")
        
        # Show context injection
        context_line = chunk.text.split("\n")[0] if "\n" in chunk.text else ""
        print(f"\nContext Prefix: {context_line}")
        
        # Show chunk text preview
        text_preview = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else chunk.text
        print(f"\nText Preview (first 200 chars):")
        print(f"  {text_preview[:200]}...")
    
    # Step 4: Build context map for captions
    print("\n" + "=" * 80)
    print("STEP 4: BUILDING CONTEXT MAP FOR IMAGE CAPTIONS")
    print("=" * 80)
    
    # Build context map (same as in ingest.py)
    context_map = {}
    for page in doc.pages:
        for img_ref in page.images:
            # Use page text as context (first 1000 chars)
            context_map[img_ref.image_id] = page.raw_text[:1000]
    
    print(f"\n✓ Built context map for {len(context_map)} images")
    
    # Show context for first few images
    print("\n" + "=" * 80)
    print("SAMPLE IMAGE CONTEXT INFORMATION:")
    print("=" * 80)
    
    shown = 0
    for img in unique_images[:5]:
        if img.image_id in context_map:
            shown += 1
            print(f"\n--- Image {shown} ---")
            print(f"Image ID: {img.image_id}")
            print(f"Page: {img.page}")
            print(f"Team: {img.team}, Year: {img.year}")
            print(f"Saved Path: {img.saved_path}")
            print(f"Format: {img.format}")
            print(f"Is Duplicate: {img.is_duplicate}")
            
            # Find which chunk(s) this image belongs to
            associated_chunks = [
                c for c in chunks 
                if img.image_id in c.image_ids
            ]
            
            if associated_chunks:
                print(f"\nAssociated with {len(associated_chunks)} chunk(s):")
                for chunk in associated_chunks:
                    print(f"  - Chunk {chunk.chunk_id}")
                    print(f"    Section: {' > '.join(chunk.headers) if chunk.headers else 'None'}")
                    print(f"    Page: {chunk.page_number}")
            
            # Show context text
            context = context_map[img.image_id]
            print(f"\nContext Text (first 300 chars):")
            print(f"  {context[:300]}...")
            
            # Find section headers for this page
            page_obj = doc.pages[img.page - 1] if img.page <= len(doc.pages) else None
            if page_obj and page_obj.headers:
                print(f"\nSection Headers on Page {img.page}:")
                for header in page_obj.headers:
                    print(f"  - {header}")
    
    # Step 5: Show what gets embedded
    print("\n" + "=" * 80)
    print("STEP 5: IMAGE EMBEDDING METADATA")
    print("=" * 80)
    
    print("\nWhat gets embedded for images:")
    print("  - Visual content only (CLIP ViT-L/14)")
    print("  - NO text context in the embedding vector itself")
    print("  - Metadata stored separately in database payload:")
    
    if unique_images:
        sample_img = unique_images[0]
        print(f"\nSample Image Metadata (for {sample_img.image_id}):")
        img_dict = sample_img.to_dict()
        
        # Show what metadata is stored
        print("\nMetadata stored with embedding:")
        print(f"  - image_id: {img_dict.get('image_id')}")
        print(f"  - team: {img_dict.get('team')}")
        print(f"  - year: {img_dict.get('year')}")
        print(f"  - page: {img_dict.get('page')}")
        print(f"  - format: {img_dict.get('format')}")
        print(f"  - saved_path: {Path(img_dict.get('saved_path', '')).name if img_dict.get('saved_path') else 'None'}")
        
        # Find associated chunk to show section info
        associated_chunks = [
            c for c in chunks 
            if sample_img.image_id in c.image_ids
        ]
        
        if associated_chunks:
            chunk = associated_chunks[0]
            print(f"\nSection Information (from associated chunk):")
            print(f"  - Section Headers: {chunk.headers}")
            print(f"  - Subsystem: {chunk.subsystem or 'None'}")
            print(f"  - Binder: {chunk.binder}")
            print(f"  - Chunk ID: {chunk.chunk_id}")
            
            # Show that section info is NOT in embedding, but available via chunk
            print(f"\n⚠️  IMPORTANT: Section headers are NOT in the image embedding")
            print(f"   They are stored in the associated TEXT chunk")
            print(f"   Images are linked to chunks via image_ids")
        
        # Show what would be stored in database
        print("\n" + "-" * 80)
        print("What gets stored in vector database for this image:")
        print("-" * 80)
        print("""
Embedding Vector (768 dimensions):
  - Pure visual features from CLIP ViT-L/14
  - No text, no context, just visual representation

Metadata Payload:
  - image_id: Unique identifier
  - team: Team number
  - year: Year
  - page: Page number
  - saved_path: File path
  - format: Image format (jpg/png)
  - width, height: Image dimensions
  - perceptual_hash: For deduplication
  
Section Context (via chunk association):
  - Stored in TEXT chunk, not image embedding
  - Accessible via chunk.image_ids lookup
  - Used for retrieval and context injection
        """)
    
    # Step 6: Show caption context (if available)
    print("\n" + "=" * 80)
    print("STEP 6: CAPTION GENERATION CONTEXT")
    print("=" * 80)
    
    print("\nWhat context is used for captions:")
    print("  1. Section header (if available)")
    print("  2. Nearby page text (first 1000 chars)")
    print("  3. OCR text from image")
    print("  4. Visual facts from vision model")
    
    print("\nCaption generation process:")
    print("  1. Vision model describes what it sees")
    print("  2. OCR extracts text from image")
    print("  3. Context text provides naming/grounding")
    print("  4. Caption synthesized from all sources")
    print("  5. Validation ensures accuracy")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nDocument: {pdf_path.name}")
    print(f"Pages: {doc.total_pages}")
    print(f"Chunks: {len(chunks)}")
    print(f"  - Chunks with images: {len(chunks_with_images)}")
    print(f"  - Chunks without images: {len(chunks) - len(chunks_with_images)}")
    print(f"\nImages: {len(processed_images)} total")
    print(f"  - Unique: {len(unique_images)}")
    print(f"  - Duplicates: {len(processed_images) - len(unique_images)}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("""
1. Images are embedded using CLIP (visual content only)
   - The embedding vector captures visual features
   - No text context is included in the embedding itself

2. Images are associated with chunks based on:
   - Page number (images on same page as chunk)
   - Section boundaries (images belong to section chunks)

3. Context information is used for:
   - Caption generation (section headers, nearby text)
   - Metadata storage (team, year, page, subsystem)
   - Retrieval (images can be found via associated chunks)

4. Smart section splitting means:
   - Images are grouped with their section's text chunks
   - This allows semantic search to find relevant images
   - Images inherit section context for better retrieval
    """)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demonstrate image embedding with section context"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing PDF files (default: data)",
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    try:
        pdf_path = find_first_pdf(args.data_dir)
        demonstrate_image_sections(pdf_path)
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

