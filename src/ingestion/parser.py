"""
Document parsing module.

Handles PDF parsing with:
- Layout-aware text extraction (PyMuPDF)
- Automatic scanned page detection
- OCR for scanned pages (Tesseract + PaddleOCR fallback)
- Table extraction (pdfplumber)
- Structured JSON output per document
"""

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics

logger = get_logger(__name__)


@dataclass
class TableData:
    """Extracted table data."""
    
    rows: List[List[str]]
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "page": self.page,
            "bbox": self.bbox,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableData":
        return cls(
            rows=data["rows"],
            page=data["page"],
            bbox=tuple(data["bbox"]),
        )
    
    def to_text(self) -> str:
        """Convert table to markdown format."""
        if not self.rows:
            return ""
        
        lines = []
        for i, row in enumerate(self.rows):
            line = "| " + " | ".join(str(cell or "") for cell in row) + " |"
            lines.append(line)
            if i == 0:
                # Add header separator
                lines.append("| " + " | ".join("---" for _ in row) + " |")
        
        return "\n".join(lines)


@dataclass
class ImageRef:
    """Reference to an extracted image."""
    
    image_id: str
    page: int
    bbox: Tuple[float, float, float, float]
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "page": self.page,
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            "visual_facts": getattr(self, "visual_facts", []),
            "uncertainties": getattr(self, "uncertainties", []),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRef":
        return cls(
            image_id=data["image_id"],
            page=data["page"],
            bbox=tuple(data["bbox"]),
            width=data["width"],
            height=data["height"],
        )


@dataclass
class ParagraphBlock:
    """Structured paragraph block with bbox and metadata."""

    text: str
    bbox: Tuple[float, float, float, float]
    tokens: int = 0
    intent: Optional[str] = None
    anchored_image_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "tokens": self.tokens,
            "intent": self.intent,
            "anchored_image_ids": self.anchored_image_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParagraphBlock":
        return cls(
            text=data.get("text", ""),
            bbox=tuple(data.get("bbox", (0, 0, 0, 0))),
            tokens=int(data.get("tokens", 0)),
            intent=data.get("intent"),
            anchored_image_ids=data.get("anchored_image_ids", []),
        )


@dataclass
class PageContent:
    """Content extracted from a single page."""
    
    page_number: int
    printed_page_number: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    paragraph_blocks: List[ParagraphBlock] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    images: List[ImageRef] = field(default_factory=list)
    raw_text: str = ""
    is_scanned: bool = False
    ocr_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "printed_page_number": self.printed_page_number,
            "headers": self.headers,
            "paragraphs": self.paragraphs,
            "paragraph_blocks": [p.to_dict() for p in self.paragraph_blocks],
            "tables": [t.to_dict() for t in self.tables],
            "images": [i.to_dict() for i in self.images],
            "raw_text": self.raw_text,
            "is_scanned": self.is_scanned,
            "ocr_confidence": self.ocr_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageContent":
        return cls(
            page_number=data["page_number"],
            printed_page_number=data.get("printed_page_number"),
            headers=data.get("headers", []),
            paragraphs=data.get("paragraphs", []),
            paragraph_blocks=[ParagraphBlock.from_dict(p) for p in data.get("paragraph_blocks", [])],
            tables=[TableData.from_dict(t) for t in data.get("tables", [])],
            images=[ImageRef.from_dict(i) for i in data.get("images", [])],
            raw_text=data.get("raw_text", ""),
            is_scanned=data.get("is_scanned", False),
            ocr_confidence=data.get("ocr_confidence"),
        )


@dataclass
class ParsedDocument:
    """Complete parsed document."""
    
    filename: str
    team: str
    year: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "team": self.team,
            "year": self.year,
            "total_pages": self.total_pages,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
            "parse_errors": self.parse_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedDocument":
        return cls(
            filename=data["filename"],
            team=data["team"],
            year=data["year"],
            total_pages=data["total_pages"],
            pages=[PageContent.from_dict(p) for p in data.get("pages", [])],
            metadata=data.get("metadata", {}),
            parse_errors=data.get("parse_errors", []),
        )
    
    def save_json(self, output_dir: Path) -> Path:
        """Save parsed document to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.team}-{self.year}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(
            "Saved parsed document",
            path=str(output_path),
            pages=self.total_pages,
        )
        return output_path


class DocumentParser:
    """
    PDF document parser with OCR support.
    
    Features:
    - Automatic scanned page detection
    - Tesseract OCR with PaddleOCR fallback
    - Table extraction with pdfplumber
    - Header/section detection
    - Image reference extraction
    """

    # Minimum text density to consider a page as digital (not scanned)
    MIN_TEXT_DENSITY = 50  # characters per page
    
    # Common header patterns in FRC binders
    HEADER_PATTERNS = [
        r"^#+\s+",  # Markdown headers
        r"^[A-Z][^.!?]*:$",  # Title case ending with colon
        r"^\d+\.\s+[A-Z]",  # Numbered sections
        r"^(Chapter|Section|Part)\s+\d+",  # Chapter/Section headers
        r"^(Introduction|Overview|Summary|Conclusion|Design|Implementation)",
    ]

    def __init__(
        self,
        use_ocr: bool = True,
        ocr_language: str = "eng",
        extract_tables: bool = True,
        extract_images: bool = True,
    ):
        """
        Initialize document parser.
        
        Args:
            use_ocr: Enable OCR for scanned pages
            ocr_language: OCR language code
            extract_tables: Enable table extraction
            extract_images: Enable image extraction
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        
        # Lazy-loaded OCR engines
        self._tesseract = None
        self._paddle_ocr = None
        
        # Compiled header patterns
        self._header_patterns = [
            re.compile(p, re.MULTILINE) for p in self.HEADER_PATTERNS
        ]

    def _get_tesseract(self):
        """Lazy-load Tesseract OCR."""
        if self._tesseract is None:
            try:
                import pytesseract
                self._tesseract = pytesseract
                logger.debug("Tesseract OCR loaded")
            except ImportError:
                logger.warning("Tesseract not available")
        return self._tesseract

    def _get_paddleocr(self):
        """Lazy-load PaddleOCR."""
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False,
                )
                logger.debug("PaddleOCR loaded")
            except ImportError:
                logger.warning("PaddleOCR not available")
        return self._paddle_ocr

    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Extract team number and year from filename.
        
        Expected formats:
        - 254-2025.pdf
        - 4607-1-2024.pdf (multi-part)
        """
        name = Path(filename).stem
        parts = name.split("-")
        
        team = parts[0] if parts else "unknown"
        year = parts[-1] if len(parts) > 1 else "unknown"
        
        return team, year

    def _is_scanned_page(self, page: fitz.Page) -> bool:
        """
        Detect if a page is scanned (image-only).
        
        A page is considered scanned if:
        - It has very little extractable text
        - It has one or more large images
        """
        text = page.get_text("text").strip()
        text_length = len(text)
        
        # Check text density
        if text_length < self.MIN_TEXT_DENSITY:
            # Check for images
            image_list = page.get_images()
            if image_list:
                return True
        
        return False

    def _ocr_page(self, page: fitz.Page) -> Tuple[str, float]:
        """
        Perform OCR on a scanned page.
        
        Returns:
            Tuple of (extracted_text, confidence)
        """
        # Render page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        
        # Try Tesseract first
        tesseract = self._get_tesseract()
        if tesseract:
            try:
                # Get OCR result with confidence
                data = tesseract.image_to_data(
                    img,
                    lang=self.ocr_language,
                    output_type=tesseract.Output.DICT,
                )
                
                # Calculate average confidence
                confidences = [
                    int(c) for c in data["conf"] if int(c) > 0
                ]
                avg_confidence = (
                    sum(confidences) / len(confidences)
                    if confidences
                    else 0
                )
                
                text = tesseract.image_to_string(img, lang=self.ocr_language)
                
                if avg_confidence > 50:  # Good confidence threshold
                    return text.strip(), avg_confidence / 100.0
                    
            except Exception as e:
                logger.warning(
                    "Tesseract OCR failed",
                    error=str(e),
                    page=page.number,
                )
        
        # Fall back to PaddleOCR for low confidence or failure
        paddle = self._get_paddleocr()
        if paddle:
            try:
                # PaddleOCR expects numpy array
                import numpy as np
                img_array = np.array(img)
                
                result = paddle.ocr(img_array, cls=True)
                
                if result and result[0]:
                    lines = []
                    confidences = []
                    
                    for line in result[0]:
                        text_info = line[1]
                        lines.append(text_info[0])
                        confidences.append(text_info[1])
                    
                    avg_confidence = (
                        sum(confidences) / len(confidences)
                        if confidences
                        else 0
                    )
                    
                    return "\n".join(lines), avg_confidence
                    
            except Exception as e:
                logger.warning(
                    "PaddleOCR failed",
                    error=str(e),
                    page=page.number,
                )
        
        return "", 0.0

    def _extract_headers(self, text: str) -> List[str]:
        """Extract headers/section titles from text."""
        headers = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check against header patterns
            for pattern in self._header_patterns:
                if pattern.match(line):
                    headers.append(line)
                    break
            
            # Also detect ALL CAPS headers (common in technical docs)
            if (
                len(line) > 3
                and len(line) < 100
                and line.isupper()
                and not line.isdigit()
            ):
                headers.append(line)
        
        return headers

    def _extract_paragraphs(self, text: str, headers: List[str]) -> List[str]:
        """Extract paragraphs, excluding headers."""
        paragraphs = []
        current_para = []
        
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            
            # Skip headers
            if line in headers:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                continue
            
            if line:
                current_para.append(line)
            else:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
        
        if current_para:
            paragraphs.append(" ".join(current_para))
        
        # Filter out very short paragraphs (likely noise)
        return [p for p in paragraphs if len(p) > 20]

    def _extract_paragraph_blocks_from_page(self, page: fitz.Page) -> List[ParagraphBlock]:
        """Extract paragraph-like text blocks with bbox from a page's layout dict."""
        blocks = []
        try:
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") == 0:  # text block
                    # Combine spans
                    spans = b.get("lines", [])
                    lines = []
                    for line in spans:
                        for span in line.get("spans", []):
                            txt = span.get("text", "").strip()
                            if txt:
                                lines.append(txt)
                    text = " ".join(lines).strip()
                    if text:
                        bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
                        tokens = max(1, len(text.split()))
                        blocks.append(ParagraphBlock(text=text, bbox=bbox, tokens=tokens))
        except Exception:
            # Fallback: no blocks
            pass

        return blocks

    def _extract_printed_page_number(self, text: str) -> Optional[str]:
        """Extract printed page number from page text."""
        # Look for page numbers at start or end of text
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        if not lines:
            return None
        
        # Check first and last few lines
        candidates = lines[:3] + lines[-3:]
        
        for line in candidates:
            # Match common page number formats
            match = re.match(r"^(?:Page\s+)?(\d+)(?:\s*of\s*\d+)?$", line, re.I)
            if match:
                return match.group(1)
            
            # Just a number alone
            if re.match(r"^\d{1,3}$", line):
                return line
        
        return None

    def _extract_tables_pdfplumber(
        self, pdf_path: Path, page_num: int
    ) -> List[TableData]:
        """Extract tables using pdfplumber."""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    extracted = page.extract_tables()
                    
                    for i, table in enumerate(extracted):
                        if table:
                            # Clean table data
                            cleaned = [
                                [
                                    str(cell).strip() if cell else ""
                                    for cell in row
                                ]
                                for row in table
                            ]
                            
                            tables.append(TableData(
                                rows=cleaned,
                                page=page_num,
                                bbox=(0, 0, 0, 0),  # pdfplumber doesn't give bbox easily
                            ))
                            
        except Exception as e:
            logger.warning(
                "Table extraction failed",
                error=str(e),
                page=page_num,
            )
        
        return tables

    def _extract_images_from_page(
        self, doc: fitz.Document, page: fitz.Page, team: str, year: str
    ) -> List[ImageRef]:
        """Extract images from a page and return references."""
        images: List[ImageRef] = []

        # Build mapping from image xref (if present in layout) to bbox
        xref_bbox: Dict[int, Tuple[float, float, float, float]] = {}
        try:
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") == 1:
                    bbox = tuple(b.get("bbox", (0, 0, 0, 0)))
                    # try to get xref
                    xinfo = b.get("image") or {}
                    xref = xinfo.get("xref") or b.get("xref")
                    try:
                        if xref is not None:
                            xref_bbox[int(xref)] = bbox
                    except Exception:
                        pass
        except Exception:
            pass

        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)

                if base_image:
                    image_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Generate deterministic image ID
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    image_id = f"{team}_{year}_p{page.number}_i{img_index}_{image_hash}"

                    bbox = xref_bbox.get(xref, (0, 0, width, height))

                    img_ref = ImageRef(
                        image_id=image_id,
                        page=page.number,
                        bbox=bbox,
                        width=width,
                        height=height,
                    )

                    # Add deterministic placeholder visual facts
                    setattr(img_ref, "visual_facts", [
                        f"image_on_page_{page.number}",
                        f"size_{width}x{height}",
                    ])
                    setattr(img_ref, "uncertainties", [])

                    images.append(img_ref)

            except Exception as e:
                logger.warning(
                    "Image extraction failed",
                    error=str(e),
                    page=page.number,
                    image_index=img_index,
                )

        return images

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        Parse a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedDocument with structured content
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        team, year = self._parse_filename(filename)
        
        logger.info(
            "Parsing document",
            filename=filename,
            team=team,
            year=year,
        )
        
        doc = fitz.open(pdf_path)
        
        parsed = ParsedDocument(
            filename=filename,
            team=team,
            year=year,
            total_pages=len(doc),
            metadata={
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
            },
        )
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # Check if scanned
                is_scanned = self._is_scanned_page(page)
                
                # Get text (OCR if needed)
                if is_scanned and self.use_ocr:
                    raw_text, ocr_confidence = self._ocr_page(page)
                else:
                    raw_text = page.get_text("text")
                    ocr_confidence = None
                
                # Extract structure
                headers = self._extract_headers(raw_text)
                paragraphs = self._extract_paragraphs(raw_text, headers)
                paragraph_blocks = self._extract_paragraph_blocks_from_page(page)
                printed_page = self._extract_printed_page_number(raw_text)
                
                # Extract tables
                tables = []
                if self.extract_tables:
                    tables = self._extract_tables_pdfplumber(pdf_path, page_num)
                
                # Extract images
                images = []
                if self.extract_images:
                    images = self._extract_images_from_page(doc, page, team, year)

                # Anchor images to nearest paragraph blocks deterministically
                try:
                    for img in images:
                        # compute image center y
                        x0, y0, x1, y1 = img.bbox
                        img_cy = (y0 + y1) / 2.0
                        # find nearest paragraph by vertical distance
                        best_para = None
                        best_dist = None
                        for para in paragraph_blocks:
                            px0, py0, px1, py1 = para.bbox
                            para_cy = (py0 + py1) / 2.0
                            dist = abs(para_cy - img_cy)
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_para = para
                        if best_para is not None and best_dist is not None and best_dist < 300:  # threshold in points
                            best_para.anchored_image_ids.append(img.image_id)
                except Exception:
                    pass

                page_content = PageContent(
                    page_number=page_num,
                    printed_page_number=printed_page,
                    headers=headers,
                    paragraphs=paragraphs,
                    paragraph_blocks=paragraph_blocks,
                    tables=tables,
                    images=images,
                    raw_text=raw_text,
                    is_scanned=is_scanned,
                    ocr_confidence=ocr_confidence,
                )
                
                parsed.pages.append(page_content)
                
            except Exception as e:
                error_msg = f"Error parsing page {page_num}: {str(e)}"
                parsed.parse_errors.append(error_msg)
                logger.error(
                    "Page parsing error",
                    page=page_num,
                    error=str(e),
                )
                metrics.record_ingestion_error(
                    "parse_error",
                    error_msg,
                    filename,
                )
        
        doc.close()
        
        logger.info(
            "Document parsed",
            filename=filename,
            pages=parsed.total_pages,
            errors=len(parsed.parse_errors),
        )
        
        return parsed

    def parse_all(
        self, input_dir: Path, output_dir: Optional[Path] = None
    ) -> List[ParsedDocument]:
        """
        Parse all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Optional directory to save JSON outputs
            
        Returns:
            List of parsed documents
        """
        input_dir = Path(input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))
        
        logger.info(
            "Starting batch parse",
            input_dir=str(input_dir),
            file_count=len(pdf_files),
        )
        
        documents = []
        
        for pdf_path in pdf_files:
            try:
                doc = self.parse(pdf_path)
                documents.append(doc)
                
                if output_dir:
                    doc.save_json(output_dir)
                
                metrics.record_document_processed(success=True)
                
            except Exception as e:
                logger.error(
                    "Document parse failed",
                    filename=pdf_path.name,
                    error=str(e),
                )
                metrics.record_document_processed(success=False)
                metrics.record_ingestion_error(
                    "document_error",
                    str(e),
                    pdf_path.name,
                )
        
        return documents
