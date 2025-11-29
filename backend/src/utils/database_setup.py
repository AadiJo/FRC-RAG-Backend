import os
import shutil
import glob
import json
from typing import List, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
import io
import hashlib
import base64
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import time
from xml.sax.saxutils import escape
import re
import sys
import chromadb

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

# Configure paths
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_PATH) # Add backend root to path to allow imports from src

from src.core.image_embedder import ImageEmbedder
DATA_PATH = os.path.join(BASE_PATH, "data")
IMAGES_PATH = os.path.join(DATA_PATH, "images")
REJECTED_IMAGES_PATH = os.path.join(DATA_PATH, "rejected_images")
CHROMA_PATH = os.path.join(BASE_PATH, "db")
IMAGE_CONTEXTS_PDF_PATH = os.path.join(DATA_PATH, "image_contexts.pdf")

# Parallelization settings
MAX_PDF_WORKERS = None  # None = auto-detect based on CPU count
MAX_IMAGE_WORKERS = 4   # Limit image processing workers to avoid overwhelming system
ENABLE_PARALLEL_PROCESSING = True  # Set to False to disable parallel processing

# Image filtering settings
MIN_IMAGE_SIZE = (80, 80)  # Minimum width and height
MAX_ASPECT_RATIO = 20  # Filter out extremely wide/tall images
MIN_OCR_CHARS = 3  # Minimum OCR characters to consider image useful
MIN_FILE_SIZE = 1000  # Minimum file size in bytes

# Enhanced filtering settings for relevant technical content
IRRELEVANT_KEYWORDS = [
    # Social/meme content
    'meme', 'funny', 'lol', 'lmao', 'joke', 'humor', 'reaction', 'face', 'emoji',
    'twitter', 'instagram', 'social media', 'facebook', 'tiktok', 'snapchat',
    
    # Game reveal/watching content
    'game reveal', 'watching', 'stream', 'livestream', 'audience', 'watching party',
    'reveal reaction', 'first look', 'initial reaction', 'team watching',
    
    # Non-technical team content
    'team photo', 'group photo', 'team picture', 'awards ceremony', 'celebration',
    'banner', 'poster', 'logo only', 'team logo', 'sponsor logo', 'title page',
    'cover page', 'intro page', 'welcome', 'introduction', 'about us',
    
    # Logo and branding content
    'logo', 'brand', 'branding', 'emblem', 'badge', 'trademark', 'copyright',
    'company logo', 'organization logo', 'first logo', 'frc logo', 'competition logo',
    'sponsor badge', 'team badge', 'team emblem', 'corporate logo', 'brand mark',
    
    # Generic/decorative content
    'decorative', 'background', 'pattern', 'texture', 'gradient', 'abstract',
    'clip art', 'stock photo', 'generic image', 'filler image'
]

TECHNICAL_KEYWORDS = [
    # Robot components and mechanisms
    'robot', 'mechanism', 'drivetrain', 'chassis', 'frame', 'gearbox', 'motor',
    'actuator', 'pneumatic', 'hydraulic', 'sensor', 'encoder', 'gyro', 'accelerometer',
    'intake', 'shooter', 'climber', 'elevator', 'arm', 'gripper', 'manipulator',
    
    # CAD and design
    'cad', 'solidworks', 'inventor', 'fusion', 'onshape', 'design', 'model',
    '3d model', 'assembly', 'part', 'drawing', 'blueprint', 'schematic',
    'dimensions', 'tolerances', 'material', 'aluminum', 'steel', 'plastic',
    
    # Programming and electronics
    'code', 'programming', 'software', 'algorithm', 'autonomous', 'teleop',
    'wiring', 'circuit', 'pcb', 'rio', 'roborio', 'can bus', 'pwm',
    'electronics', 'voltage', 'current', 'power distribution',
    
    # Build and manufacturing
    'machining', 'fabrication', 'welding', 'cutting', 'drilling', 'milling',
    'lathe', 'cnc', 'tools', 'workshop', 'build process', 'assembly process',
    'prototype', 'iteration', 'testing', 'troubleshooting',
    
    # Game elements and strategy
    'field', 'game piece', 'scoring', 'strategy', 'alliance', 'match',
    'autonomous period', 'endgame', 'points', 'ranking'
]

# Captioning model setup
CAPTIONING_ENABLED = True  # Set to False to disable image captioning
CAPTIONING_MODEL_ID = "Salesforce/blip-image-captioning-large"
captioning_processor = None
captioning_model = None


# ========================================================================
# ENHANCED CONTEXT EXTRACTION FUNCTIONS
# ========================================================================

def extract_enhanced_text_structure(page) -> Dict[str, Any]:
    """
    Extract text with better structure preservation using multiple methods
    """
    results = {
        "structured_content": "",
        "bullet_points": [],
        "headers": [],
        "technical_specs": [],
        "layout_preserved": False
    }
    
    try:
        # Method 1: Extract text blocks with positioning
        blocks = page.get_text("dict")
        sorted_blocks = []
        
        for block in blocks["blocks"]:
            if block["type"] == 0:  # Text block
                block_text = ""
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    if line_text.strip():
                        block_text += line_text + "\n"
                
                if block_text.strip():
                    sorted_blocks.append({
                        "text": block_text.strip(),
                        "bbox": block["bbox"],  # [x0, y0, x1, y1]
                        "position": (block["bbox"][1], block["bbox"][0])  # (y, x) for sorting
                    })
        
        # Sort blocks by position (top to bottom, left to right)
        sorted_blocks.sort(key=lambda x: (x["position"][0], x["position"][1]))
        
        # Process each block for structure
        structured_parts = []
        for block in sorted_blocks:
            text = block["text"]
            
            # Detect headers (ALL CAPS or specific patterns)
            if detect_header(text):
                results["headers"].append(text)
                structured_parts.append(f"## {text}")
                
            # Detect bullet points and lists
            elif detect_list_item(text):
                results["bullet_points"].append(text)
                structured_parts.append(f"• {text}")
                
            # Detect technical specifications
            elif detect_technical_spec(text):
                results["technical_specs"].append(text)
                structured_parts.append(f"SPEC: {text}")
                
            else:
                structured_parts.append(text)
        
        results["structured_content"] = "\n".join(structured_parts)
        results["layout_preserved"] = True
        
        # Method 2: Fallback to regular extraction if positioning fails
        if not results["structured_content"].strip():
            results["structured_content"] = page.get_text()
            results["layout_preserved"] = False
            
    except Exception as e:
        print(f"Warning: Enhanced extraction failed: {e}")
        results["structured_content"] = page.get_text()
        results["layout_preserved"] = False
    
    return results

def detect_header(text: str) -> bool:
    """Detect header text patterns"""
    text = text.strip()
    
    # Common header patterns in FRC docs
    header_patterns = [
        r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS headers
        r'^[A-Z][A-Za-z\s]+:$',  # Title case with colon
        r'^\d+\.\s*[A-Z]',       # Numbered headers
    ]
    
    # Short, distinctive text is likely a header
    if len(text) < 100 and any(re.match(pattern, text) for pattern in header_patterns):
        return True
        
    # Single line, all caps
    if '\n' not in text and text.isupper() and 3 <= len(text) <= 50:
        return True
    
    return False

def detect_list_item(text: str) -> bool:
    """Detect list items and bullet points"""
    text = text.strip()
    
    # Look for bullet-like patterns at the start
    bullet_patterns = [
        r'^\s*[•·▪▫‣⁃]\s+',           # Unicode bullets
        r'^\s*[-*+]\s+',               # ASCII bullets
        r'^\s*\d+[\.)]\s+',            # Numbered lists
        r'^\s*[a-zA-Z][\.)]\s+',       # Lettered lists
        r'^Built with\s+',              # Technical specs starting with "Built with"
        r'^\d+[\-\s]stage\s+',         # Multi-stage descriptions
        r'^Casca[ded|ding]',           # Specific robotics terms
        r'^Tube\s+',                   # Component descriptions
        r'^Main\s+structure',          # Structure descriptions
        r'^Structure\s+is',            # Structure descriptions
    ]
    
    return any(re.match(pattern, text, re.IGNORECASE) for pattern in bullet_patterns)

def detect_technical_spec(text: str) -> bool:
    """Detect technical specifications"""
    text = text.lower()
    
    # Technical specification keywords
    tech_keywords = [
        'aluminum', 'steel', 'motor', 'gear', 'ratio', 'tube', 'bearing',
        'shaft', 'wheel', 'encoder', 'sensor', 'pneumatic', 'hydraulic',
        'voltage', 'current', 'torque', 'speed', 'diameter', 'thickness',
        'weight', 'cg', 'center of gravity', 'rigging', 'trusses'
    ]
    
    # Measurement patterns
    measurement_patterns = [
        r'\d+["\']\s*x\s*\d+["\']\s*',  # Dimensions like 2" x 1"
        r'\d+\s*(inch|in|mm|cm|ft)',     # Measurements
        r'\d+:\d+',                      # Ratios
        r'\d+\s*(lb|kg|oz)',            # Weights
        r'\d+\s*(rpm|fps|mph)',         # Speeds
    ]
    
    # Check for technical keywords
    keyword_count = sum(1 for keyword in tech_keywords if keyword in text)
    
    # Check for measurement patterns
    measurement_found = any(re.search(pattern, text) for pattern in measurement_patterns)
    
    return keyword_count >= 2 or measurement_found

def create_enhanced_document_content(page_text_structure: Dict[str, Any], 
                                   page_num: int, 
                                   image_info: List[Dict[str, Any]], 
                                   pdf_name: str) -> str:
    """
    Create enhanced document content that preserves context and relationships
    """
    content_parts = []
    
    # Document metadata
    content_parts.append(f"=== DOCUMENT: {pdf_name} | PAGE: {page_num + 1} ===")
    
    # Add headers as primary structure
    if page_text_structure["headers"]:
        content_parts.append("\n📋 PRIMARY SECTIONS:")
        for header in page_text_structure["headers"]:
            content_parts.append(f"▸ {header}")
    
    # Add bullet points/list items as structured content
    if page_text_structure["bullet_points"]:
        content_parts.append("\n🔧 DETAILED SPECIFICATIONS:")
        for i, bullet in enumerate(page_text_structure["bullet_points"], 1):
            content_parts.append(f"{i}. {bullet}")
    
    # Add technical specifications
    if page_text_structure["technical_specs"]:
        content_parts.append("\n⚙️ TECHNICAL DETAILS:")
        for spec in page_text_structure["technical_specs"]:
            content_parts.append(f"• {spec}")
    
    # Add full structured content
    content_parts.append("\n📄 COMPLETE CONTENT:")
    content_parts.append(page_text_structure["structured_content"])
    
    # Add image context with better relationship mapping
    if image_info:
        content_parts.append("\n🖼️ ASSOCIATED VISUAL CONTENT:")
        for img in image_info:
            img_text = img.get("ocr_text", "").strip()
            if img_text and len(img_text) > 10:
                content_parts.append(f"📷 {img['filename']}: {img_text}")
            else:
                # For images without OCR text, try to relate them to the page content
                content_parts.append(f"📷 {img['filename']}: [Technical diagram showing components described above]")
                
                # Try to infer relationships based on proximity and content
                if any(keyword in page_text_structure["structured_content"].lower() 
                      for keyword in ['elevator', 'lift', 'vertical']):
                    content_parts.append(f"   → Likely shows: Elevator mechanism and components")
                elif any(keyword in page_text_structure["structured_content"].lower() 
                        for keyword in ['drive', 'wheel', 'motion']):
                    content_parts.append(f"   → Likely shows: Drivetrain components and assembly")
                elif any(keyword in page_text_structure["structured_content"].lower() 
                        for keyword in ['intake', 'gripper', 'pickup']):
                    content_parts.append(f"   → Likely shows: Intake mechanism design")
    
    # Add context preservation note
    layout_status = "✅ Layout preserved" if page_text_structure["layout_preserved"] else "⚠️ Basic extraction"
    content_parts.append(f"\n📊 Extraction quality: {layout_status}")
    
    return "\n".join(content_parts)

def create_enhanced_image_context(img_info: Dict[str, Any], pdf_name: str, page_num: int, 
                                page_structure: Dict[str, Any], context_excerpt: str) -> str:
    """
    Create much richer image context using page structure information
    """
    image_text = (img_info.get("ocr_text") or "").strip()
    filename = img_info.get("filename", "Unknown")
    
    context_parts = [
        f"📄 **DOCUMENT**: {pdf_name} | Page {page_num + 1}",
        f"🖼️ **IMAGE**: {filename}",
        ""
    ]
    
    # Add page headers for context
    if page_structure.get("headers"):
        context_parts.append("📋 **PAGE SECTIONS**:")
        for header in page_structure["headers"][:3]:  # Top 3 headers
            context_parts.append(f"  ▸ {header}")
        context_parts.append("")
    
    # Add extracted image text
    if image_text and len(image_text) > 10:
        context_parts.extend([
            "🔍 **EXTRACTED TEXT**:",
            image_text,
            ""
        ])
    
    # Add related specifications
    if page_structure.get("bullet_points"):
        context_parts.append("🔧 **RELATED SPECIFICATIONS**:")
        for spec in page_structure["bullet_points"][:3]:  # Top 3 most relevant specs
            # Truncate long specs
            spec_text = spec if len(spec) <= 100 else spec[:97] + "..."
            context_parts.append(f"  • {spec_text}")
        context_parts.append("")
    
    # Add inferred content based on context
    context_parts.append("🎯 **CONTENT ANALYSIS**:")
    
    page_content_lower = page_structure["structured_content"].lower()
    
    # Infer what the image likely shows based on page content
    if any(term in page_content_lower for term in ['elevator', 'lift', 'vertical', '3-stage']):
        context_parts.append("  → Technical diagram: Elevator mechanism and vertical movement system")
        context_parts.append("  → Components: Rails, carriages, rigging, support structure")
        
    elif any(term in page_content_lower for term in ['drivetrain', 'wheel', 'motor', 'chassis']):
        context_parts.append("  → Technical diagram: Drivetrain and mobility system")
        context_parts.append("  → Components: Motors, gears, wheels, chassis framework")
        
    elif any(term in page_content_lower for term in ['intake', 'gripper', 'manipulator']):
        context_parts.append("  → Technical diagram: Game piece manipulation system")
        context_parts.append("  → Components: Intake mechanism, gripper, actuators")
        
    else:
        context_parts.append("  → Technical diagram: Robot component or subsystem")
        context_parts.append("  → Engineering drawing with specifications and dimensions")
    
    # Add materials and construction details if available
    if any(term in page_content_lower for term in ['aluminum', 'tube', 'steel', 'material']):
        context_parts.append("  → Materials: Metal fabrication with precise specifications")
    
    if any(term in page_content_lower for term in ['rigging', 'bearing', 'coupling']):
        context_parts.append("  → Mechanics: Precision mechanical components and connections")
    
    context_parts.extend([
        "",
        "📝 **FULL PAGE CONTEXT**:",
        context_excerpt
    ])
    
    return "\n".join(context_parts)

# ========================================================================
# END ENHANCED EXTRACTION FUNCTIONS
# ========================================================================


def create_context_excerpt(text: str, max_length: int = 1500) -> str:
    """Return a trimmed version of context text for storage."""
    if not text:
        return ""

    cleaned_text = text.strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text

    return cleaned_text[: max_length - 3].rstrip() + "..."

def convert_to_relative_path(absolute_path: str, base_path: str) -> str:
    """Convert absolute path to relative path from project root for web compatibility."""
    try:
        rel_path = os.path.relpath(absolute_path, base_path)
        return rel_path
    except (ValueError, OSError):
        # Fallback: try to extract the data/images/... part
        if "data/images/" in absolute_path:
            idx = absolute_path.find("data/images/")
            return absolute_path[idx:]
        return absolute_path

def format_image_context(img_info: Dict[str, Any], pdf_name: str, page_num: int, page_context: str) -> str:
    """
    Format image context with consistent styling and comprehensive information.
    """
    image_text = (img_info.get("ocr_text") or "").strip()
    filename = img_info.get("filename", "Unknown")
    
    # Create a standardized context format
    context_parts = [
        f"📄 **Document**: {pdf_name}",
        f"📖 **Page**: {page_num + 1}",
        f"🖼️ **Image**: {filename}",
        f"📏 **Dimensions**: {img_info.get('size', 'Unknown')}",
        ""
    ]
    
    # Add extracted content section
    if image_text:
        context_parts.extend([
            "🔍 **Extracted Content**:",
            image_text,
            ""
        ])
    else:
        context_parts.extend([
            "🔍 **Extracted Content**: No readable text detected",
            ""
        ])
    
    # Add page context section
    if page_context:
        context_parts.extend([
            "📝 **Page Context**:",
            page_context,
            ""
        ])
    
    # Add technical classification
    context_parts.extend([
        "🔧 **Classification**: Technical diagram/image relevant to robotics",
        f"⚙️ **Content Type**: {'Text-based' if image_text else 'Visual-only'}"
    ])
    
    return "\n".join(context_parts)

def load_captioning_model():
    """
    Load the image captioning model and processor from Hugging Face.
    """
    global captioning_processor, captioning_model, CAPTIONING_ENABLED
    if not CAPTIONING_ENABLED:
        print("Image captioning is disabled.")
        return
    
    if captioning_model is None:
        try:
            print(f"Loading image captioning model: {CAPTIONING_MODEL_ID}...")
            captioning_processor = BlipProcessor.from_pretrained(CAPTIONING_MODEL_ID)
            captioning_model = BlipForConditionalGeneration.from_pretrained(CAPTIONING_MODEL_ID)
            print("Image captioning model loaded successfully.")
        except Exception as e:
            print(f"Error loading captioning model: {e}")
            print("Please ensure you have 'transformers' and 'torch' installed.")
            CAPTIONING_ENABLED = False

def generate_caption(pil_image: Image.Image) -> str:
    """
    Generate a caption for an image using the pre-trained model.
    """
    if not CAPTIONING_ENABLED or captioning_model is None or captioning_processor is None:
        return ""
    
    try:
        # Prepare image for the model
        inputs = captioning_processor(images=pil_image, return_tensors="pt")
        
        # Generate caption
        outputs = captioning_model.generate(**inputs, max_length=75)
        
        # Decode caption
        caption = captioning_processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption.strip()
    except Exception as e:
        print(f"Error during image captioning: {e}")
        return ""

def main():
    print("Starting database creation...")
    start_time = time.time()
    
    # Load the captioning model
    load_captioning_model()
    
    # Clear existing database and rejected images
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    if os.path.exists(REJECTED_IMAGES_PATH):
        print(f"Removing existing rejected images at {REJECTED_IMAGES_PATH}")
        shutil.rmtree(REJECTED_IMAGES_PATH)
    os.makedirs(REJECTED_IMAGES_PATH)

    if os.path.exists(IMAGE_CONTEXTS_PDF_PATH):
        print(f"Removing existing image context manifest at {IMAGE_CONTEXTS_PDF_PATH}")
        os.remove(IMAGE_CONTEXTS_PDF_PATH)
    
    # Find all PDF files in the data directory
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_PATH}")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {[os.path.basename(f) for f in pdf_files]}")
    
    # Process PDFs in parallel
    all_documents = process_pdfs_parallel(pdf_files)
    
    if not all_documents:
        print("No documents were successfully processed.")
        return
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total documents created: {len(all_documents)}")

    # Persist image-to-context mapping for downstream inspection
    write_image_context_manifest(all_documents)
    
    # Split text into chunks
    chunks = split_text(all_documents)
    
    # Save to vector database
    save_to_chroma(chunks)
    
    end_time = time.time()
    print(f"Enhanced database creation completed in {end_time - start_time:.2f} seconds!")

def process_pdfs_parallel(pdf_files: List[str], max_workers: int = None) -> List[Document]:
    """
    Process multiple PDFs in parallel using ThreadPoolExecutor
    """
    if not ENABLE_PARALLEL_PROCESSING or len(pdf_files) == 1:
        # Fall back to sequential processing
        print("Processing PDFs sequentially...")
        all_documents = []
        for pdf_path in pdf_files:
            print(f"\nProcessing: {os.path.basename(pdf_path)}")
            
            # Create PDF-specific image folder
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_images_path = os.path.join(IMAGES_PATH, pdf_name)
            
            # Clear existing images for this PDF
            if os.path.exists(pdf_images_path):
                print(f"Removing existing images at {pdf_images_path}")
                shutil.rmtree(pdf_images_path)
            
            # Ensure directories exist
            os.makedirs(pdf_images_path, exist_ok=True)
            
            try:
                documents = process_pdf_with_images(pdf_path, pdf_images_path)
                all_documents.extend(documents)
                print(f"✓ Completed {os.path.basename(pdf_path)}: {len(documents)} documents")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(pdf_path)}: {e}")
        
        return all_documents
    
    if max_workers is None:
        max_workers = MAX_PDF_WORKERS or min(len(pdf_files), mp.cpu_count())
    
    print(f"Processing {len(pdf_files)} PDFs using {max_workers} workers...")
    
    all_documents = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {}
        for pdf_path in pdf_files:
            # Create PDF-specific image folder
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_images_path = os.path.join(IMAGES_PATH, pdf_name)
            
            # Clear existing images for this PDF
            if os.path.exists(pdf_images_path):
                print(f"Removing existing images at {pdf_images_path}")
                shutil.rmtree(pdf_images_path)
            
            # Ensure directories exist
            os.makedirs(pdf_images_path, exist_ok=True)
            
            future = executor.submit(process_pdf_with_images, pdf_path, pdf_images_path)
            future_to_pdf[future] = pdf_path
        
        # Collect results as they complete
        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                documents = future.result()
                all_documents.extend(documents)
                print(f"✓ Completed {os.path.basename(pdf_path)}: {len(documents)} documents")
            except Exception as e:
                print(f"✗ Error processing {os.path.basename(pdf_path)}: {e}")
    
    return all_documents

def process_pdf_with_images(pdf_path: str, pdf_images_path: str) -> List[Document]:
    """
    ENHANCED: Process PDF to extract both text and images with structure preservation
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    documents = []
    pdf_name = os.path.basename(pdf_path)
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    print(f"[{pdf_name}] ENHANCED processing PDF with {total_pages} pages...")
    
    pages_processed = 0
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        
        # ENHANCED: Extract text with structure preservation
        page_structure = extract_enhanced_text_structure(page)
        
        if page_structure["structured_content"].strip():  # Only process pages with content
            
            # Analyze page context for better image filtering (use structured content)
            page_context = analyze_page_context(page_structure["structured_content"], page_num)
            
            # Extract images from this page, providing structured content for context
            image_info = extract_images_from_page(page, page_num, pdf_images_path, page_context, page_structure["structured_content"])
            
            # ENHANCED: Create document with much richer context
            enhanced_content = create_enhanced_document_content(
                page_structure, page_num, image_info, pdf_name
            )
            
            metadata = {
                "source": pdf_path,
                "page": page_num + 1,
                "type": "enhanced_text_with_images",
                "image_count": len(image_info),
                "image_filenames": json.dumps([img["filename"] for img in image_info]) if image_info else "[]",
                "headers_count": len(page_structure["headers"]),
                "bullet_points_count": len(page_structure["bullet_points"]),
                "layout_preserved": page_structure["layout_preserved"],
                "extraction_method": "enhanced_structure_aware"
            }
            
            document = Document(
                page_content=enhanced_content,  # Much richer content
                metadata=metadata
            )
            documents.append(document)
            
            # ENHANCED: Create enhanced image context documents
            if image_info:
                context_excerpt = create_context_excerpt(page_structure["structured_content"])

                for img_info in image_info:
                    image_text = (img_info.get("ocr_text") or "").strip()
                    
                    # ENHANCED: Create much richer image context using page structure
                    enhanced_image_context = create_enhanced_image_context(
                        img_info, pdf_name, page_num, page_structure, context_excerpt
                    )

                    image_context_doc = Document(
                        page_content=enhanced_image_context,
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "type": "enhanced_image_context",
                            "image_file": img_info["filename"],
                            "image_path": img_info["file_path"],
                            "image_text": image_text,
                            "page_context_excerpt": context_excerpt,
                            "related_headers": json.dumps(page_structure["headers"]),
                            "related_specs": json.dumps(page_structure["bullet_points"][:3]),  # Top 3 specs
                            "extraction_method": "enhanced_context_aware"
                        }
                    )
                    documents.append(image_context_doc)

                # Create enhanced metadata entries for images
                for img_info in image_info:
                    img_metadata = {
                        "source": pdf_path,
                        "page": page_num + 1,
                        "type": "enhanced_image_info",
                        "image_file": img_info["filename"],
                        "image_path": img_info["file_path"],
                        "has_ocr_text": bool(img_info.get("ocr_text", "").strip()),
                        "related_headers_count": len(page_structure["headers"]),
                        "related_specs_count": len(page_structure["bullet_points"])
                    }
                    
                    # Create enhanced image metadata document
                    img_doc = Document(
                        page_content=f"Enhanced image metadata for {img_info['filename']} on page {page_num + 1} with {len(page_structure['headers'])} headers and {len(page_structure['bullet_points'])} specifications",
                        metadata=img_metadata
                    )
                    documents.append(img_doc)
            
            # Create enhanced OCR documents with better context
            for img_info in image_info:
                if img_info.get("ocr_text") and img_info["ocr_text"].strip():
                    # Enhanced OCR content with page structure context
                    enhanced_ocr_content = f"""Image OCR Content: {img_info['ocr_text']}

Page Context: This image appears on page {page_num + 1} of {pdf_name}

Related Sections: {', '.join(page_structure['headers'][:3])}

Technical Specifications Present: {len(page_structure['bullet_points'])} detailed specs

Full Context: This image is part of a technical document containing structured information about {', '.join(page_structure['headers'][:2]) if page_structure['headers'] else 'robotics components'}."""
                    
                    img_document = Document(
                        page_content=enhanced_ocr_content,
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "type": "enhanced_image_text",
                            "image_file": img_info["filename"],
                            "image_path": img_info["file_path"],
                            "extraction_method": "enhanced_ocr_with_context"
                        }
                    )
                    documents.append(img_document)
        
        pages_processed += 1
        if pages_processed % 5 == 0 or pages_processed == total_pages:
            extraction_quality = "✅ Enhanced" if page_structure.get("layout_preserved") else "⚠️ Basic"
            print(f"[{pdf_name}] {extraction_quality} processing: {pages_processed}/{total_pages} pages...")
    
    pdf_document.close()
    print(f"[{pdf_name}] ENHANCED extraction completed: {len(documents)} documents created")
    return documents

def analyze_page_context(page_text: str, page_num: int) -> Dict[str, Any]:
    """
    Analyze the text content of a page to provide context for image filtering
    """
    text_lower = page_text.lower()
    
    context = {
        'is_intro_page': False,
        'is_team_page': False,
        'is_technical_page': False,
        'is_social_page': False,
        'technical_score': 0,
        'social_score': 0
    }
    
    # Detect intro/welcome pages
    intro_indicators = ['welcome', 'introduction', 'about us', 'team overview', 
                       'mission statement', 'who we are', 'our story']
    intro_score = sum(1 for indicator in intro_indicators if indicator in text_lower)
    if intro_score >= 2 or (page_num <= 3 and intro_score >= 1):
        context['is_intro_page'] = True
    
    # Detect team/social pages
    team_indicators = ['team members', 'our team', 'meet the team', 'student list',
                      'mentors', 'coaches', 'sponsors', 'thank you', 'acknowledgments',
                      'awards', 'recognition', 'competition results']
    team_score = sum(1 for indicator in team_indicators if indicator in text_lower)
    if team_score >= 2:
        context['is_team_page'] = True
    
    # Detect social/fun content
    social_indicators = ['game reveal', 'watching', 'stream', 'party', 'fun',
                        'meme', 'joke', 'funny', 'reaction', 'social media']
    social_score = sum(1 for indicator in social_indicators if indicator in text_lower)
    context['social_score'] = social_score
    if social_score >= 2:
        context['is_social_page'] = True
    
    # Detect technical content
    technical_indicators = ['design', 'build', 'programming', 'software', 'hardware',
                           'mechanism', 'robot', 'autonomous', 'control system',
                           'sensors', 'actuators', 'drivetrain', 'cad', 'fabrication',
                           'testing', 'troubleshooting', 'strategy', 'game analysis']
    technical_score = sum(1 for indicator in technical_indicators if indicator in text_lower)
    context['technical_score'] = technical_score
    if technical_score >= 3:
        context['is_technical_page'] = True
    
    return context

def extract_images_from_page(page, page_num: int, pdf_images_path: str, page_context: Dict[str, Any] = None, page_text: str = "") -> List[Dict[str, Any]]:
    """
    Extract images from a specific PDF page with filtering and parallel OCR
    """
    image_list = page.get_images()
    
    if not image_list:
        return []
    
    # First pass: extract and save all valid images
    valid_images = []
    for img_index, img in enumerate(image_list):
        try:
            # Get image data
            xref = img[0]
            pix = fitz.Pixmap(page.parent, xref)
            
            # Convert to PIL Image for processing
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Generate filename and path
                filename = f"page{page_num}_img{img_index}.png"
                file_path = os.path.join(pdf_images_path, filename)
                
                # Save image temporarily
                pil_image.save(file_path)

                # Filter out useless images based on size and aspect ratio
                if not is_useful_image(pil_image):
                    reason = f"Initial filter: size {pil_image.size}, aspect ratio"
                    move_to_rejected(file_path, reason)
                    pix = None
                    continue
                
                # Check file size after saving
                if os.path.getsize(file_path) < MIN_FILE_SIZE:
                    reason = f"File size too small ({os.path.getsize(file_path)} bytes)"
                    move_to_rejected(file_path, reason)
                    pix = None
                    continue
                
                valid_images.append({
                    "filename": filename,
                    "file_path": file_path,
                    "pil_image": pil_image,
                    "page_num": page_num,
                    "img_index": img_index
                })
            
            pix = None  # Free memory
            
        except Exception as e:
            print(f"Error extracting image {img_index} from page {page_num}: {e}")
    
    if not valid_images:
        return []
    
    # Second pass: perform OCR in parallel and apply filters
    return process_images_parallel(valid_images, page_context, page_text)

def process_single_image_ocr(image_data: Dict[str, Any], page_context: Dict[str, Any] = None, page_text: str = "") -> Dict[str, Any]:
    """
    Process a single image for OCR, with captioning fallback, and filtering.
    """
    filename = image_data["filename"]
    file_path = image_data["file_path"]
    pil_image = image_data["pil_image"]
    page_num = image_data["page_num"]
    
    try:
        # Perform OCR on the image
        ocr_text = ""
        try:
            ocr_text = pytesseract.image_to_string(pil_image).strip()
        except Exception as e:
            print(f"OCR failed for {filename}: {e}")

        # Fallback to image captioning if OCR is weak
        caption_text = ""
        if CAPTIONING_ENABLED and len(ocr_text) < MIN_OCR_CHARS:
            caption_text = generate_caption(pil_image)
            if caption_text:
                print(f"Generated caption for {filename}: '{caption_text}'")
        
        # Use OCR text if it's substantial, otherwise prioritize caption
        image_text = ocr_text if len(ocr_text) >= MIN_OCR_CHARS else caption_text
        
        # If still no text, use a generic descriptor if content is meaningful
        if not image_text.strip() and has_meaningful_content(pil_image):
            image_text = "Meaningful visual content without readable text."

        # Check for logo content - reject if detected
        if is_likely_logo(pil_image, image_text):
            reason = f"OCR text indicates logo content: {image_text[:100]}..."
            move_to_rejected(file_path, reason)
            return None

        # Filter based on OCR/caption content
        if len(image_text) < MIN_OCR_CHARS:
            reason = "No meaningful text content from OCR or captioning"
            move_to_rejected(file_path, reason)
            return None
        
        # Enhanced content filtering for technical relevance
        if not is_technically_relevant(image_text, page_text, pil_image, page_num, page_context):
            reason = f"Not technically relevant based on combined text context."
            move_to_rejected(file_path, reason)
            return None
        
        # Create image info with relative path for web compatibility
        relative_path = convert_to_relative_path(file_path, BASE_PATH)
        img_info = {
            "filename": filename,
            "file_path": relative_path,
            "page": page_num + 1,
            "index": image_data["img_index"],
            "ocr_text": image_text,  # Now contains either OCR or caption
            "size": pil_image.size
        }
        
        print(f"Extracted image: {filename} (Text: {len(image_text)} chars, Size: {pil_image.size})")
        return img_info
        
    except Exception as e:
        reason = f"Error during processing: {e}"
        move_to_rejected(file_path, reason)
        return None

def process_images_parallel(valid_images: List[Dict[str, Any]], page_context: Dict[str, Any] = None, page_text: str = "") -> List[Dict[str, Any]]:
    """
    Process multiple images in parallel for OCR and filtering
    """
    if not ENABLE_PARALLEL_PROCESSING or len(valid_images) <= 1:
        # For single image or disabled parallel processing, process directly
        image_info = []
        for img_data in valid_images:
            result = process_single_image_ocr(img_data, page_context, page_text)
            if result:
                image_info.append(result)
        return image_info
    
    image_info = []
    max_workers = min(len(valid_images), MAX_IMAGE_WORKERS)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit OCR tasks
        ocr_func = partial(process_single_image_ocr, page_context=page_context, page_text=page_text)
        future_to_image = {executor.submit(ocr_func, img_data): img_data 
                          for img_data in valid_images}
        
        # Collect results
        for future in as_completed(future_to_image):
            try:
                result = future.result()
                if result:
                    image_info.append(result)
            except Exception as e:
                img_data = future_to_image[future]
                print(f"Error in parallel OCR for {img_data['filename']}: {e}")
    
    return image_info

def is_useful_image(pil_image: Image.Image) -> bool:
    """
    Filter out useless images based on size and aspect ratio
    """
    width, height = pil_image.size
    
    # Filter by minimum size
    if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
        return False
    
    # Filter by aspect ratio (avoid extremely wide or tall images)
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False
    
    return True

def is_likely_logo(pil_image: Image.Image, ocr_text: str = "") -> bool:
    """
    Detect if an image is likely a logo based on OCR text mentioning logo explicitly
    """
    if not ocr_text:
        return False
    
    ocr_lower = ocr_text.lower()
    
    # Check if the OCR text explicitly mentions it's a logo
    logo_phrases = [
        'logo', 'company logo', 'team logo', 'sponsor logo', 'brand logo',
        'official logo', 'organization logo', 'corporate logo', 'logo design',
        'logo image', 'brand mark', 'trademark', 'registered trademark'
    ]
    
    # Return True if any logo phrase is found in the OCR text
    return any(phrase in ocr_lower for phrase in logo_phrases)

def has_meaningful_content(pil_image: Image.Image) -> bool:
    """
    Check if image has meaningful visual content beyond simple shapes/logos
    Enhanced to better detect technical diagrams and mechanical drawings
    """
    import numpy as np
    
    # Convert to numpy array for analysis
    img_array = np.array(pil_image.convert('L'))  # Convert to grayscale
    
    # Calculate variance - low variance suggests solid colors or simple content
    variance = np.var(img_array)
    
    # Calculate edge detection using simple gradient
    grad_x = np.gradient(img_array, axis=1)
    grad_y = np.gradient(img_array, axis=0)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_density = np.mean(edge_magnitude > 10)  # Threshold for edge detection
    
    # Enhanced analysis for technical content
    
    # 1. Detect line-based content (technical drawings often have many straight lines)
    # Simple line detection using Hough-like approach
    height, width = img_array.shape
    line_score = 0
    
    # Check for horizontal and vertical line patterns
    for row in img_array[::5]:  # Sample every 5th row
        consecutive_pixels = 0
        for pixel in row:
            if pixel < 128:  # Dark pixels (assuming lines are dark)
                consecutive_pixels += 1
            else:
                if consecutive_pixels > width * 0.1:  # Line spans >10% of width
                    line_score += 1
                consecutive_pixels = 0
    
    for col_idx in range(0, width, 5):  # Sample every 5th column
        col = img_array[:, col_idx]
        consecutive_pixels = 0
        for pixel in col:
            if pixel < 128:
                consecutive_pixels += 1
            else:
                if consecutive_pixels > height * 0.1:  # Line spans >10% of height
                    line_score += 1
                consecutive_pixels = 0
    
    # 2. Detect geometric shapes (circles, rectangles) common in technical drawings
    shape_score = 0
    
    # Simple circle detection - look for curved edges
    kernel_size = min(20, min(height, width) // 10)
    if kernel_size > 5:
        # Check for circular patterns by looking at pixel intensity around centers
        center_y, center_x = height // 2, width // 2
        for radius in range(kernel_size, min(height, width) // 4, kernel_size):
            if radius < min(height, width) // 2:
                circle_pixels = []
                for angle in np.linspace(0, 2*np.pi, 16):
                    y = int(center_y + radius * np.sin(angle))
                    x = int(center_x + radius * np.cos(angle))
                    if 0 <= y < height and 0 <= x < width:
                        circle_pixels.append(img_array[y, x])
                
                if len(circle_pixels) > 8:
                    circle_variance = np.var(circle_pixels)
                    if circle_variance > 500:  # High variance suggests edge
                        shape_score += 1
    
    # 3. Text density analysis (technical drawings often have annotations)
    # Look for text-like patterns (small clustered dark regions)
    text_score = 0
    block_size = max(5, min(height, width) // 50)
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = img_array[y:y+block_size, x:x+block_size]
            if np.mean(block) < 200 and np.var(block) > 100:  # Dark area with variation
                text_score += 1
    
    # Enhanced decision criteria
    min_variance = 100
    min_edge_density = 0.02
    min_line_score = 3
    min_shape_score = 1
    min_text_score = 5
    
    # Technical drawing indicators
    is_technical = (line_score >= min_line_score or 
                   shape_score >= min_shape_score or
                   text_score >= min_text_score)
    
    # Basic content indicators
    has_basic_content = (variance > min_variance or edge_density > min_edge_density)
    
    return has_basic_content or is_technical

def is_technically_relevant(ocr_text: str, page_text: str, pil_image: Image.Image, page_num: int, page_context: Dict[str, Any] = None) -> bool:
    """
    Enhanced filtering to determine if an image is technically relevant to robotics.
    Now considers the full page text for context.
    """
    # Convert all text to lowercase for case-insensitive matching
    ocr_lower = ocr_text.lower()
    page_lower = page_text.lower()
    combined_text = ocr_lower + " " + page_lower

    # Initialize context if not provided
    if page_context is None:
        page_context = {'is_intro_page': False, 'is_team_page': False, 
                       'is_technical_page': False, 'is_social_page': False,
                       'technical_score': 0, 'social_score': 0}
    
    # Strong filters based on page context
    if page_context.get('is_social_page', False):
        return False  # Exclude all images from social/fun pages
    
    # On intro or team pages, require strong technical keywords in the combined text
    if page_context.get('is_intro_page', False) or page_context.get('is_team_page', False):
        if not any(keyword in combined_text for keyword in TECHNICAL_KEYWORDS):
            return False

    # Check for irrelevant keywords that suggest non-technical content
    irrelevant_score = 0
    for keyword in IRRELEVANT_KEYWORDS:
        if keyword in ocr_lower:  # Check only image text for irrelevant keywords
            irrelevant_score += 1
    
    # Check for technical keywords in the combined text
    technical_score = 0
    for keyword in TECHNICAL_KEYWORDS:
        if keyword in combined_text:
            technical_score += 1
    
    # Boost score for technical diagrams and CAD images based on image text
    technical_visual_indicators = ['dimension', 'measurement', 'scale', 'view',
                                  'section', 'detail', 'assembly', 'part']
    for indicator in technical_visual_indicators:
        if indicator in ocr_lower:
            technical_score += 2
            
    # Decision logic
    # If technical score is high, it's likely relevant
    if technical_score >= 5:
        return True
    
    # If irrelevant score from image is high, it's likely not relevant
    if irrelevant_score >= 3:
        return False
    
    # If page has strong technical context, be more lenient with image text
    if page_context.get('is_technical_page', False) and technical_score >= 2:
        return True

    # For borderline cases, prefer inclusion if there's some technical content
    if technical_score > 1 and irrelevant_score <= 1:
        return True
    
    # If no clear technical content and some irrelevant indicators, exclude
    if technical_score <= 1 and irrelevant_score > 0:
        return False
    
    # Default to inclusion if uncertain, especially if the page has technical context
    if page_context.get('technical_score', 0) > 2:
        return True

    return False

def split_text(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks for better retrieval
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Larger chunks to maintain context
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Show sample chunk
    if chunks:
        print("\nSample chunk:")
        print(f"Content: {chunks[10].page_content[:200]}...")
        print(f"Metadata: {chunks[10].metadata}")
    
    return chunks

def save_to_chroma(chunks: List[Document]):
    """
    Save chunks to Chroma vector database
    """
    # Clear existing database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Filter complex metadata manually
    filtered_chunks = []
    for chunk in chunks:
        # Create a new document with simple metadata
        simple_metadata = {}
        for key, value in chunk.metadata.items():
            # Only keep simple types that ChromaDB can handle
            if isinstance(value, (str, int, float, bool)):
                simple_metadata[key] = value
            elif isinstance(value, list):
                # Convert lists to strings
                simple_metadata[key] = str(value)
            else:
                # Convert other types to strings
                simple_metadata[key] = str(value)
        
        filtered_chunk = Document(
            page_content=chunk.page_content,
            metadata=simple_metadata
        )
        filtered_chunks.append(filtered_chunk)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save database
    db = Chroma.from_documents(
        filtered_chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(filtered_chunks)} chunks to {CHROMA_PATH}")

    # --- Image Embeddings ---
    print("Generating image embeddings...")
    try:
        # Initialize SigLIP model
        img_embedder = ImageEmbedder()
        
        # Get Chroma client
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Create or get image collection
        image_collection = client.get_or_create_collection(name="image_embeddings")
        
        # Prepare image data
        ids = []
        embeddings_list = []
        metadatas = []
        documents = [] 
        
        for chunk in chunks:
            if chunk.metadata.get('type') == 'image_context' and chunk.metadata.get('image_path'):
                image_path = chunk.metadata['image_path']
                # Fix path if it's relative or incorrect
                if not os.path.isabs(image_path):
                    image_path = os.path.join(BASE_PATH, image_path)
                    
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path)
                        # Use SigLIP embedder
                        emb = img_embedder.embed_image(image)[0]
                        
                        # Use a unique ID
                        img_id = f"img_{os.path.basename(image_path)}"
                        
                        ids.append(img_id)
                        embeddings_list.append(emb)
                        
                        # Filter metadata for Chroma
                        meta = {k: str(v) for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
                        metadatas.append(meta)
                        
                        documents.append(chunk.page_content) # Store the context/OCR as the document text
                        
                    except Exception as e:
                        print(f"Error embedding image {image_path}: {e}")
        
        # Add to collection in batches
        if ids:
            batch_size = 50 # Smaller batch size for safety
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                image_collection.add(
                    ids=ids[i:end],
                    embeddings=embeddings_list[i:end],
                    metadatas=metadatas[i:end],
                    documents=documents[i:end]
                )
            print(f"Added {len(ids)} image embeddings to 'image_embeddings' collection")
        else:
            print("No images found to embed.")
            
    except Exception as e:
        print(f"Failed to generate image embeddings: {e}")


def write_image_context_manifest(documents: List[Document], manifest_path: str = IMAGE_CONTEXTS_PDF_PATH):
    """Create a PDF manifest capturing each retained image and its associated context."""
    image_entries: List[Dict[str, Any]] = []

    for doc in documents:
        if doc.metadata.get("type") != "image_context":
            continue

        entry = {
            "image_file": doc.metadata.get("image_file"),
            "image_path": doc.metadata.get("image_path"),
            "source_pdf": os.path.basename(doc.metadata.get("source", "")),
            "page": doc.metadata.get("page"),
            "image_text": doc.metadata.get("image_text", ""),
            "page_context_excerpt": doc.metadata.get("page_context_excerpt", ""),
        }
        image_entries.append(entry)

    if not image_entries:
        print("No image context entries to write.")
        return

    image_entries.sort(
        key=lambda entry: (
            entry.get("source_pdf") or "",
            entry.get("page") or 0,
            entry.get("image_file") or "",
        )
    )

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    stylesheet = getSampleStyleSheet()
    heading_style = stylesheet["Heading3"]
    body_style = stylesheet["BodyText"]

    doc_template = SimpleDocTemplate(
        manifest_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=48,
        bottomMargin=48,
    )

    story = []

    for idx, entry in enumerate(image_entries, start=1):
        header_text = (
            f"Image {idx}: {escape((entry.get('image_file') or 'Unknown'))} "
            f"(PDF: {escape((entry.get('source_pdf') or 'Unknown'))}, "
            f"Page: {entry.get('page', 'N/A')})"
        )
        story.append(Paragraph(header_text, heading_style))
        story.append(Spacer(1, 0.15 * inch))

        image_path = entry.get("image_path")
        added_visual = False
        
        # Convert relative path back to absolute path for file access
        if image_path:
            if not os.path.isabs(image_path):
                absolute_image_path = os.path.join(BASE_PATH, image_path)
            else:
                absolute_image_path = image_path
        else:
            absolute_image_path = None
            
        if absolute_image_path and os.path.exists(absolute_image_path):
            try:
                with Image.open(absolute_image_path) as pil_img:
                    width, height = pil_img.size

                max_width = 5.5 * inch
                max_height = 4.5 * inch

                if width and height:
                    display_width = max_width
                    display_height = display_width * (height / width)

                    if display_height > max_height:
                        display_height = max_height
                        display_width = display_height * (width / height)
                else:
                    display_width = max_width
                    display_height = max_height

                story.append(RLImage(absolute_image_path, width=display_width, height=display_height))
                story.append(Spacer(1, 0.15 * inch))
                added_visual = True
            except Exception as exc:
                message = escape(f"Unable to display image (error: {exc}).")
                story.append(Paragraph(message, body_style))
                story.append(Spacer(1, 0.1 * inch))

        if not added_visual:
            story.append(Paragraph("Image file not available for preview.", body_style))
            story.append(Spacer(1, 0.1 * inch))

        text_sections = [
            ("Extracted image text", entry.get("image_text") or "No extracted text available."),
            ("Page context excerpt", entry.get("page_context_excerpt") or "No page context available."),
        ]

        for label, content in text_sections:
            story.append(Paragraph(f"<b>{escape(label)}:</b>", body_style))
            content_text = escape(str(content)).replace("\n", "<br/>")
            story.append(Paragraph(content_text, body_style))
            story.append(Spacer(1, 0.1 * inch))

        if idx != len(image_entries):
            story.append(PageBreak())

    doc_template.build(story)
    print(f"Saved {len(image_entries)} image context entries to {manifest_path}")

def test_database():
    """
    Test the created database with sample queries
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Test query
    test_queries = [
        "tube shaped object",
        "coral picker",
        "CAD design",
        "gripper mechanism"
    ]
    
    print("\n" + "="*50)
    print("Testing database with sample queries:")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = db.similarity_search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {result.page_content[:200]}...")
            print(f"Page: {result.metadata.get('page', 'N/A')}")
            print(f"Type: {result.metadata.get('type', 'N/A')}")
            if 'images' in result.metadata and result.metadata['images']:
                print(f"Associated images: {[img['filename'] for img in result.metadata['images']]}")
            if result.metadata.get('image_file'):
                print(f"Image file: {result.metadata['image_file']}")

def benchmark_performance():
    """
    Benchmark the performance difference between parallel and sequential processing
    """
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Find PDF files
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    if not pdf_files:
        print("No PDF files found for benchmarking")
        return
    
    print(f"Benchmarking with {len(pdf_files)} PDF files...")
    
    # Test with parallel processing enabled
    global ENABLE_PARALLEL_PROCESSING
    original_setting = ENABLE_PARALLEL_PROCESSING
    
    try:
        # Benchmark parallel processing
        ENABLE_PARALLEL_PROCESSING = True
        print("\n1. Testing with parallel processing ENABLED...")
        start_time = time.time()
        
        # Process just one PDF for quick benchmark
        test_pdf = pdf_files[0]
        pdf_name = os.path.splitext(os.path.basename(test_pdf))[0]
        pdf_images_path = os.path.join(IMAGES_PATH, f"{pdf_name}_benchmark_parallel")
        
        if os.path.exists(pdf_images_path):
            shutil.rmtree(pdf_images_path)
        os.makedirs(pdf_images_path, exist_ok=True)
        
        parallel_docs = process_pdf_with_images(test_pdf, pdf_images_path)
        parallel_time = time.time() - start_time
        
        # Benchmark sequential processing
        ENABLE_PARALLEL_PROCESSING = False
        print("\n2. Testing with parallel processing DISABLED...")
        start_time = time.time()
        
        pdf_images_path = os.path.join(IMAGES_PATH, f"{pdf_name}_benchmark_sequential")
        
        if os.path.exists(pdf_images_path):
            shutil.rmtree(pdf_images_path)
        os.makedirs(pdf_images_path, exist_ok=True)
        
        sequential_docs = process_pdf_with_images(test_pdf, pdf_images_path)
        sequential_time = time.time() - start_time
        
        # Results
        print(f"\n{'='*50}")
        print("BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"PDF: {os.path.basename(test_pdf)}")
        print(f"Parallel processing time:   {parallel_time:.2f} seconds")
        print(f"Sequential processing time: {sequential_time:.2f} seconds")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")
        print(f"Documents (parallel):   {len(parallel_docs)}")
        print(f"Documents (sequential): {len(sequential_docs)}")
        
        # Cleanup benchmark directories
        for dir_name in [f"{pdf_name}_benchmark_parallel", f"{pdf_name}_benchmark_sequential"]:
            dir_path = os.path.join(IMAGES_PATH, dir_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
        
    finally:
        # Restore original setting
        ENABLE_PARALLEL_PROCESSING = original_setting

def move_to_rejected(file_path: str, reason: str):
    """Moves a file to the rejected images directory."""
    if not os.path.exists(file_path):
        return
    try:
        # Ensure the base rejected directory exists
        os.makedirs(REJECTED_IMAGES_PATH, exist_ok=True)
        
        # Create a subdirectory for the PDF source
        pdf_name = os.path.basename(os.path.dirname(file_path))
        rejected_subfolder = os.path.join(REJECTED_IMAGES_PATH, pdf_name)
        os.makedirs(rejected_subfolder, exist_ok=True)
        
        # Move the file
        new_path = os.path.join(rejected_subfolder, os.path.basename(file_path))
        shutil.move(file_path, new_path)
        print(f"Rejected image '{os.path.basename(file_path)}' moved to rejected folder. Reason: {reason}")
    except Exception as e:
        print(f"Error moving rejected image {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for benchmark flag
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_performance()
    else:
        main()
        
        # Optionally test the database
        print("\nWould you like to test the database? (y/n)")
        try:
            if input().lower().startswith('y'):
                test_database()
        except KeyboardInterrupt:
            print("\nSkipping database test.")
