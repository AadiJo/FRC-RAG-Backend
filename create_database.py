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

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pytesseract

# Configure paths
DATA_PATH = "data"
IMAGES_PATH = os.path.join(DATA_PATH, "images")
CHROMA_PATH = "db"

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

def main():
    print("Starting database creation...")
    start_time = time.time()
    
    # Clear existing database
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
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
    Process PDF to extract both text and images, linking them together
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    documents = []
    pdf_name = os.path.basename(pdf_path)
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    print(f"[{pdf_name}] Processing PDF with {total_pages} pages...")
    
    pages_processed = 0
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        
        # Extract text from page
        page_text = page.get_text()
        
        if page_text.strip():  # Only process pages with text
            # Analyze page context for better image filtering
            page_context = analyze_page_context(page_text, page_num)
            
            # Extract images from this page
            image_info = extract_images_from_page(page, page_num, pdf_images_path, page_context)
            
            # Create document with text and image references
            metadata = {
                "source": pdf_path,
                "page": page_num + 1,  # 1-indexed for user friendliness
                "type": "text_with_images",
                "image_count": len(image_info),
                "image_filenames": json.dumps([img["filename"] for img in image_info]) if image_info else "[]"
            }
            
            # Add image descriptions to the text content if images exist
            enhanced_content = page_text
            if image_info:
                image_descriptions = []
                for img_info in image_info:
                    if img_info.get("ocr_text"):
                        image_descriptions.append(f"[Image {img_info['filename']}: {img_info['ocr_text']}]")
                    else:
                        image_descriptions.append(f"[Image {img_info['filename']}: Visual content on page {page_num + 1}]")
                
                enhanced_content += "\n\nImages on this page:\n" + "\n".join(image_descriptions)
            
            document = Document(
                page_content=enhanced_content,
                metadata=metadata
            )
            documents.append(document)
            
            # Store image info separately for later retrieval
            if image_info:
                for img_info in image_info:
                    # Create a simple metadata entry for image info
                    img_metadata = {
                        "source": pdf_path,
                        "page": page_num + 1,
                        "type": "image_info",
                        "image_file": img_info["filename"],
                        "image_path": img_info["file_path"],
                        "has_ocr_text": bool(img_info.get("ocr_text", "").strip())
                    }
                    
                    # Create a document just for storing image metadata
                    img_doc = Document(
                        page_content=f"Image metadata for {img_info['filename']} on page {page_num + 1}",
                        metadata=img_metadata
                    )
                    documents.append(img_doc)
            
            # Create separate documents for each image with OCR text
            for img_info in image_info:
                if img_info.get("ocr_text") and img_info["ocr_text"].strip():
                    img_document = Document(
                        page_content=f"Image content: {img_info['ocr_text']}\n\nContext: This image appears on page {page_num + 1} of the document.",
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "type": "image_text",
                            "image_file": img_info["filename"],
                            "image_path": img_info["file_path"]
                        }
                    )
                    documents.append(img_document)
        
        pages_processed += 1
        if pages_processed % 5 == 0 or pages_processed == total_pages:
            print(f"[{pdf_name}] Processed {pages_processed}/{total_pages} pages...")
    
    pdf_document.close()
    print(f"[{pdf_name}] Extracted content from {total_pages} pages, created {len(documents)} documents")
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

def extract_images_from_page(page, page_num: int, pdf_images_path: str, page_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
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
                
                # Filter out useless images
                if not is_useful_image(pil_image):
                    print(f"Filtered out image: page{page_num}_img{img_index}.png (not useful)")
                    pix = None
                    continue
                
                # Generate filename
                filename = f"page{page_num}_img{img_index}.png"
                file_path = os.path.join(pdf_images_path, filename)
                
                # Save image
                pil_image.save(file_path)
                
                # Check file size after saving
                if os.path.getsize(file_path) < MIN_FILE_SIZE:
                    print(f"Filtered out image: {filename} (too small file size)")
                    os.remove(file_path)
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
    return process_images_parallel(valid_images, page_context)

def process_single_image_ocr(image_data: Dict[str, Any], page_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Process a single image for OCR and filtering
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
        
        # Filter based on OCR content
        if len(ocr_text) < MIN_OCR_CHARS and not has_meaningful_content(pil_image):
            print(f"Filtered out image: {filename} (no meaningful content)")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None
        
        # Enhanced content filtering for technical relevance
        if not is_technically_relevant(ocr_text, pil_image, page_num, page_context):
            print(f"Filtered out image: {filename} (not technically relevant)")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None
        
        # Create image info
        img_info = {
            "filename": filename,
            "file_path": file_path,
            "page": page_num + 1,
            "index": image_data["img_index"],
            "ocr_text": ocr_text,
            "size": pil_image.size
        }
        
        print(f"Extracted image: {filename} (OCR: {len(ocr_text)} chars, Size: {pil_image.size})")
        return img_info
        
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return None

def process_images_parallel(valid_images: List[Dict[str, Any]], page_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Process multiple images in parallel for OCR and filtering
    """
    if not ENABLE_PARALLEL_PROCESSING or len(valid_images) <= 1:
        # For single image or disabled parallel processing, process directly
        image_info = []
        for img_data in valid_images:
            result = process_single_image_ocr(img_data, page_context)
            if result:
                image_info.append(result)
        return image_info
    
    image_info = []
    max_workers = min(len(valid_images), MAX_IMAGE_WORKERS)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit OCR tasks
        ocr_func = partial(process_single_image_ocr, page_context=page_context)
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

def is_technically_relevant(ocr_text: str, pil_image: Image.Image, page_num: int, page_context: Dict[str, Any] = None) -> bool:
    """
    Enhanced filtering to determine if an image is technically relevant to robotics
    """
    # Convert OCR text to lowercase for case-insensitive matching
    ocr_lower = ocr_text.lower()
    
    # Initialize context if not provided
    if page_context is None:
        page_context = {'is_intro_page': False, 'is_team_page': False, 
                       'is_technical_page': False, 'is_social_page': False,
                       'technical_score': 0, 'social_score': 0}
    
    # Strong filters based on page context
    if page_context.get('is_social_page', False):
        return False  # Exclude all images from social/fun pages
    
    if page_context.get('is_intro_page', False) and page_num <= 5:
        # Be very strict on intro pages
        technical_in_image = any(keyword in ocr_lower for keyword in TECHNICAL_KEYWORDS[:10])  # Top technical keywords
        if not technical_in_image:
            return False
    
    if page_context.get('is_team_page', False):
        # Only include if image has clear technical content
        technical_in_image = any(keyword in ocr_lower for keyword in TECHNICAL_KEYWORDS[:15])
        if not technical_in_image:
            return False
    
    # Check for irrelevant keywords that suggest non-technical content
    irrelevant_score = 0
    for keyword in IRRELEVANT_KEYWORDS:
        if keyword in ocr_lower:
            irrelevant_score += 1
    
    # Check for technical keywords that suggest relevant content
    technical_score = 0
    for keyword in TECHNICAL_KEYWORDS:
        if keyword in ocr_lower:
            technical_score += 1
    
    # Additional heuristics based on content patterns
    
    # 1. Filter out images with excessive social media language
    social_indicators = ['@', '#hashtag', 'follow us', 'like and subscribe', 
                        'social media', 'post', 'share', 'comment']
    for indicator in social_indicators:
        if indicator in ocr_lower:
            irrelevant_score += 2
    
    # 2. Filter out images that are primarily text with non-technical content
    if len(ocr_text) > 50:  # Substantial text content
        # Check if it's mostly non-technical text
        words = ocr_lower.split()
        non_technical_words = ['welcome', 'introduction', 'about', 'team', 'members',
                              'sponsors', 'thank', 'thanks', 'acknowledgment', 'fun',
                              'joke', 'meme', 'funny', 'laugh', 'smile']
        non_tech_count = sum(1 for word in words if word in non_technical_words)
        if non_tech_count > len(words) * 0.3:  # More than 30% non-technical words
            irrelevant_score += 3
    
    # 3. Filter based on image characteristics for likely memes/social content
    width, height = pil_image.size
    
    # Very wide images often contain memes or banners
    if width / height > 3:
        irrelevant_score += 1
    
    # Square images with minimal text often are logos or memes
    if abs(width - height) < min(width, height) * 0.1 and len(ocr_text) < 20:
        irrelevant_score += 1
    
    # 4. Page-based filtering - first few pages often contain intro/team content
    if page_num <= 3:  # First 3 pages
        intro_keywords = ['welcome', 'introduction', 'about', 'overview', 'team']
        for keyword in intro_keywords:
            if keyword in ocr_lower:
                irrelevant_score += 2
    
    # 5. Filter images with primarily names/titles (often team photos or credits)
    if len(ocr_text) > 20:
        # Look for patterns like "John Smith", "Team Captain", etc.
        import re
        name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last name pattern
            r'\bcaptain\b', r'\bmentor\b', r'\bcoach\b', r'\bstudent\b',
            r'\bpresident\b', r'\bvice\b', r'\bdirector\b'
        ]
        name_matches = sum(len(re.findall(pattern, ocr_text, re.IGNORECASE)) 
                          for pattern in name_patterns)
        if name_matches > 3:  # Multiple name/title patterns
            irrelevant_score += 2
    
    # 6. Boost score for technical diagrams and CAD images
    technical_visual_indicators = ['dimension', 'measurement', 'scale', 'view',
                                  'section', 'detail', 'assembly', 'part']
    for indicator in technical_visual_indicators:
        if indicator in ocr_lower:
            technical_score += 2
    
    # 7. Filter out images with excessive punctuation (often decorative)
    if len(ocr_text) > 10:
        punctuation_ratio = sum(1 for char in ocr_text if not char.isalnum() and not char.isspace()) / len(ocr_text)
        if punctuation_ratio > 0.3:  # More than 30% punctuation
            irrelevant_score += 1
    
    # Decision logic
    # If technical score is high, likely relevant
    if technical_score >= 3:
        return True
    
    # If irrelevant score is high, likely not relevant
    if irrelevant_score >= 3:
        return False
    
    # For borderline cases, prefer inclusion if there's any technical content
    if technical_score > 0 and irrelevant_score <= 1:
        return True
    
    # If no clear technical content and some irrelevant indicators, exclude
    if technical_score == 0 and irrelevant_score > 0:
        return False
    
    # Default to inclusion if uncertain (better to have false positives than miss technical content)
    return True

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
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and save database
    db = Chroma.from_documents(
        filtered_chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(filtered_chunks)} chunks to {CHROMA_PATH}")

def test_database():
    """
    Test the created database with sample queries
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
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
        
        # Optionally run benchmark
        print("\nWould you like to run a performance benchmark? (y/n)")
        try:
            if input().lower().startswith('y'):
                benchmark_performance()
        except KeyboardInterrupt:
            print("\nSkipping benchmark.")
