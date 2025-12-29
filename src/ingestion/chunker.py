"""
Document chunking module.

Implements section-aware chunking with:
- Preference for section/header boundaries
- Fallback to page boundaries
- Configurable token targets
- Chunk versioning
- Edge case handling (very small/large sections)
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.config import settings
from ..utils.logger import get_logger
from ..utils.metrics import metrics

from .parser import PageContent, ParsedDocument

logger = get_logger(__name__)


# Simple token estimation (words ~= tokens * 0.75)
def estimate_tokens(text: str) -> int:
    """Estimate token count from text."""
    words = len(text.split())
    # Rough approximation: 1 word ≈ 1.3 tokens on average
    return int(words * 1.3)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    
    chunk_id: str
    text: str
    page_number: int
    section_index: int
    team: str
    year: str
    binder: str
    subsystem: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    visual_facts: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    token_count: int = 0
    version: int = 1
    content_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "section_index": self.section_index,
            "team": self.team,
            "year": self.year,
            "binder": self.binder,
            "subsystem": self.subsystem,
            "headers": self.headers,
            "image_ids": self.image_ids,
            "visual_facts": self.visual_facts,
            "uncertainties": self.uncertainties,
            "token_count": self.token_count,
            "version": self.version,
            "content_hash": self.content_hash,
        }


class DocumentChunker:
    """
    Section-aware document chunker.
    
    Features:
    - Prefer section/header boundaries
    - Merge small sections
    - Split large sections
    - Associate images with chunks
    - Generate stable chunk IDs
    """

    # Subsystem keywords for classification
    SUBSYSTEMS = {
        "drivetrain": ["drivetrain", "drive", "chassis", "wheels", "motors", "gearbox"],
        "intake": ["intake", "roller", "grabber", "collector"],
        "shooter": ["shooter", "flywheel", "launch", "catapult"],
        "arm": ["arm", "pivot", "manipulator", "extension"],
        "climber": ["climber", "climb", "hook", "winch"],
        "elevator": ["elevator", "lift", "carriage"],
        "electrical": ["electrical", "wiring", "pdp", "pdh", "canbus"],
        "software": ["software", "code", "autonomous", "vision", "odometry"],
        "strategy": ["strategy", "match", "game", "scoring"],
    }

    def __init__(
        self,
        min_tokens: int = settings.chunk_min_tokens,
        target_tokens: int = settings.chunk_target_tokens,
        max_tokens: int = settings.chunk_max_tokens,
    ):
        """
        Initialize chunker.
        
        Args:
            min_tokens: Minimum tokens per chunk (merge smaller)
            target_tokens: Target tokens per chunk
            max_tokens: Maximum tokens per chunk (split larger)
        """
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens

    def _generate_chunk_id(
        self, year: str, binder: str, page: int, section: int
    ) -> str:
        """
        Generate stable chunk ID.
        
        Format: <year>_<binder>_p<page>_s<section>
        """
        # Clean binder name for ID
        clean_binder = re.sub(r"[^a-zA-Z0-9]", "", binder)[:20]
        return f"{year}_{clean_binder}_p{page}_s{section}"

    def _compute_content_hash(self, text: str) -> str:
        """Compute hash of chunk content for versioning."""
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def _format_context_string(
        self, year: str, binder: str, headers: List[str]
    ) -> str:
        """Format context string for injection."""
        # Clean binder name
        binder_clean = binder.replace(".pdf", "").replace("_", " ")
        
        context_parts = []
        context_parts.append(f"Source: {binder_clean} ({year})")
        
        if headers:
            context_parts.append(f"Section: {' > '.join(headers)}")
            
        return " | ".join(context_parts)

    def _extract_original_text(self, text: str) -> str:
        """
        Extract original text from chunk text that may have context prefix.
        
        Context format: [Source: ... | Section: ...]\n{original_text}
        """
        # Check if text starts with context prefix
        if text.startswith("[") and "\n" in text:
            # Find the first newline after the context bracket
            newline_idx = text.find("\n")
            if newline_idx > 0 and text[newline_idx - 1] == "]":
                # Extract text after context prefix
                return text[newline_idx + 1:].strip()
        # No context prefix, return as-is
        return text

    def _detect_subsystem(self, text: str, headers: List[str]) -> Optional[str]:
        """
        Detect subsystem from text content.
        
        Returns:
            Subsystem name or None if not detected
        """
        combined = " ".join(headers + [text]).lower()
        
        # Check each subsystem's keywords
        for subsystem, keywords in self.SUBSYSTEMS.items():
            for keyword in keywords:
                if keyword in combined:
                    return subsystem
        
        return None

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences for fine-grained chunking."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_small_chunks(
        self, chunks: List[Chunk], min_tokens: int
    ) -> List[Chunk]:
        """
        Merge consecutive chunks that are too small.
        
        Preserves chunk IDs of the first chunk in each merge.
        Properly handles context injection for merged chunks.
        """
        if not chunks:
            return chunks
        
        merged: List[Chunk] = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            # Check if current is too small and can be merged
            if current.token_count < min_tokens:
                # Extract original text from both chunks (remove context prefixes)
                current_original = self._extract_original_text(current.text)
                next_original = self._extract_original_text(next_chunk.text)
                
                # Merge original texts
                merged_original = current_original + "\n\n" + next_original
                
                # Re-inject context using current chunk's metadata
                # Use combined headers for better context
                combined_headers = list(current.headers) + [h for h in next_chunk.headers if h not in current.headers]
                context_str = self._format_context_string(current.year, current.binder, combined_headers)
                merged_text = f"[{context_str}]\n{merged_original}"
                
                current.text = merged_text
                current.token_count = estimate_tokens(merged_text)
                current.content_hash = self._compute_content_hash(merged_text)
                current.headers.extend([h for h in next_chunk.headers if h not in current.headers])
                current.image_ids.extend(next_chunk.image_ids)
                # Keep the page of the merged content
                # (useful to know the chunk spans multiple pages)
            else:
                merged.append(current)
                current = next_chunk
        
        merged.append(current)
        
        return merged

    def _split_large_chunk(
        self, chunk: Chunk, max_tokens: int, target_tokens: int
    ) -> List[Chunk]:
        """
        Split a chunk that exceeds max tokens.
        
        Tries to split at sentence boundaries.
        Properly handles context injection for split chunks.
        """
        if chunk.token_count <= max_tokens:
            return [chunk]
        
        # Extract original text (remove context prefix if present)
        original_text = self._extract_original_text(chunk.text)
        sentences = self._split_by_sentences(original_text)
        
        if not sentences:
            # Can't split, return as-is
            return [chunk]
        
        split_chunks: List[Chunk] = []
        current_text: List[str] = []
        current_tokens = 0
        section_suffix = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > target_tokens and current_text:
                # Create chunk from accumulated text
                original_text_part = " ".join(current_text)
                
                # Re-inject context for this split chunk
                # Use chunk's headers for all splits (they're from the same section)
                context_str = self._format_context_string(chunk.year, chunk.binder, chunk.headers)
                full_text = f"[{context_str}]\n{original_text_part}"
                
                new_id = f"{chunk.chunk_id}_{section_suffix}"
                
                split_chunks.append(Chunk(
                    chunk_id=new_id,
                    text=full_text,
                    page_number=chunk.page_number,
                    section_index=chunk.section_index,
                    team=chunk.team,
                    year=chunk.year,
                    binder=chunk.binder,
                    subsystem=chunk.subsystem,
                    headers=chunk.headers.copy() if section_suffix == 0 else [],
                    image_ids=chunk.image_ids.copy() if section_suffix == 0 else [],
                    token_count=estimate_tokens(full_text),
                    version=chunk.version,
                    content_hash=self._compute_content_hash(full_text),
                ))
                
                current_text = [sentence]
                current_tokens = sentence_tokens
                section_suffix += 1
            else:
                current_text.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining text
        if current_text:
            original_text_part = " ".join(current_text)
            
            # Re-inject context
            context_str = self._format_context_string(chunk.year, chunk.binder, chunk.headers)
            full_text = f"[{context_str}]\n{original_text_part}"
            
            new_id = f"{chunk.chunk_id}_{section_suffix}" if section_suffix > 0 else chunk.chunk_id
            
            split_chunks.append(Chunk(
                chunk_id=new_id,
                text=full_text,
                page_number=chunk.page_number,
                section_index=chunk.section_index,
                team=chunk.team,
                year=chunk.year,
                binder=chunk.binder,
                subsystem=chunk.subsystem,
                headers=chunk.headers.copy() if section_suffix == 0 else [],
                image_ids=chunk.image_ids.copy() if section_suffix == 0 else [],
                token_count=estimate_tokens(full_text),
                version=chunk.version,
                content_hash=self._compute_content_hash(full_text),
            ))
        
        return split_chunks

    def _chunk_page(
        self,
        page: PageContent,
        team: str,
        year: str,
        binder: str,
        base_section: int = 0,
    ) -> Tuple[List[Chunk], int]:
        """
        Chunk a single page.
        
        Returns:
            Tuple of (chunks, next_section_index)
        """
        chunks: List[Chunk] = []
        section_index = base_section
        
        # Get image IDs for this page
        page_image_ids = [img.image_id for img in page.images]
        
        # Strategy 1: Try to chunk by headers/sections
        if page.headers and page.paragraphs:
            current_header: List[str] = []
            current_text: List[str] = []
            current_images: List[str] = []
            
            all_content = page.raw_text.split("\n")
            header_set = set(page.headers)
            
            for line in all_content:
                line = line.strip()
                if not line:
                    continue
                
                if line in header_set:
                    # Save previous section
                    if current_text:
                        text = " ".join(current_text)
                        chunk_id = self._generate_chunk_id(
                            year, binder, page.page_number, section_index
                        )
                        
                        # Inject context into text
                        context_str = self._format_context_string(year, binder, current_header)
                        full_text = f"[{context_str}]\n{text}"
                        
                        chunks.append(Chunk(
                            chunk_id=chunk_id,
                            text=full_text,
                            page_number=page.page_number,
                            section_index=section_index,
                            team=team,
                            year=year,
                            binder=binder,
                            subsystem=self._detect_subsystem(text, current_header),
                            headers=current_header.copy(),
                            image_ids=current_images.copy(),
                            token_count=estimate_tokens(full_text),
                            content_hash=self._compute_content_hash(full_text),
                        ))
                        section_index += 1
                        current_text = []
                        current_images = []
                    
                    current_header = [line]
                else:
                    current_text.append(line)
            
            # Add final section
            if current_text:
                text = " ".join(current_text)
                chunk_id = self._generate_chunk_id(
                    year, binder, page.page_number, section_index
                )
                
                # Assign remaining images to last chunk
                if page_image_ids:
                    current_images = page_image_ids
                
                # Inject context into text
                context_str = self._format_context_string(year, binder, current_header)
                full_text = f"[{context_str}]\n{text}"
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=full_text,
                    page_number=page.page_number,
                    section_index=section_index,
                    team=team,
                    year=year,
                    binder=binder,
                    subsystem=self._detect_subsystem(text, current_header),
                    headers=current_header,
                    image_ids=current_images,
                    token_count=estimate_tokens(full_text),
                    content_hash=self._compute_content_hash(full_text),
                ))
                section_index += 1
        
        # Strategy 2: Fall back to full page as chunk
        elif page.raw_text.strip():
            text = page.raw_text.strip()
            chunk_id = self._generate_chunk_id(
                year, binder, page.page_number, section_index
            )
            
            # Inject context into text
            context_str = self._format_context_string(year, binder, page.headers)
            full_text = f"[{context_str}]\n{text}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=full_text,
                page_number=page.page_number,
                section_index=section_index,
                team=team,
                year=year,
                binder=binder,
                subsystem=self._detect_subsystem(text, page.headers),
                headers=page.headers,
                image_ids=page_image_ids,
                token_count=estimate_tokens(full_text),
                content_hash=self._compute_content_hash(full_text),
            ))
            section_index += 1
        
        # Add tables as separate chunks
        for table_idx, table in enumerate(page.tables):
            table_text = table.to_text()
            if table_text:
                chunk_id = self._generate_chunk_id(
                    year, binder, page.page_number, section_index
                )
                
                # Inject context for table chunks
                table_headers = page.headers + [f"Table {table_idx + 1}"]
                context_str = self._format_context_string(year, binder, table_headers)
                full_table_text = f"[{context_str}]\n{table_text}"
                
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=full_table_text,
                    page_number=page.page_number,
                    section_index=section_index,
                    team=team,
                    year=year,
                    binder=binder,
                    subsystem=None,  # Tables are often cross-subsystem
                    headers=table_headers,
                    image_ids=[],
                    token_count=estimate_tokens(full_table_text),
                    content_hash=self._compute_content_hash(full_table_text),
                ))
                section_index += 1
        
        return chunks, section_index

    def chunk_document(self, doc: ParsedDocument) -> List[Chunk]:
        """
        Chunk an entire parsed document.
        
        Args:
            doc: Parsed document
            
        Returns:
            List of chunks
        """
        logger.info(
            "Chunking document",
            filename=doc.filename,
            pages=doc.total_pages,
        )
        
        all_chunks: List[Chunk] = []
        section_index = 0
        
        for page in doc.pages:
            page_chunks, section_index = self._chunk_page(
                page=page,
                team=doc.team,
                year=doc.year,
                binder=doc.filename,
                base_section=section_index,
            )

            # Attach anchored images and visual facts to chunks deterministically
            try:
                # Build paragraph -> anchored image ids map
                para_map = {}
                for para in getattr(page, "paragraph_blocks", []):
                    para_map.get(para.text)  # ensure attribute access
                    para_map[para.text] = list(para.anchored_image_ids or [])

                # Map image_id -> ImageRef for quick lookup
                id_to_image = {img.image_id: img for img in getattr(page, "images", [])}

                for chunk in page_chunks:
                    original = self._extract_original_text(chunk.text)
                    assigned = set()

                    # If paragraph text appears in chunk, collect its anchored images
                    for para_text, ids in para_map.items():
                        if para_text and para_text in original:
                            assigned.update(ids)

                    # Fallback: if no anchored images found, assign all page images
                    if not assigned and id_to_image:
                        assigned = set(id_to_image.keys())

                    chunk.image_ids = list(assigned)

                    # Collect visual facts and uncertainties from matched images
                    vf = []
                    unc = []
                    for iid in chunk.image_ids:
                        img = id_to_image.get(iid)
                        if img:
                            vf.extend(getattr(img, "visual_facts", []) or [])
                            unc.extend(getattr(img, "uncertainties", []) or [])

                    # Deduplicate while preserving order
                    chunk.visual_facts = list(dict.fromkeys(vf))
                    chunk.uncertainties = list(dict.fromkeys(unc))
            except Exception:
                pass

            all_chunks.extend(page_chunks)
        
        # Post-processing: merge small chunks
        all_chunks = self._merge_small_chunks(all_chunks, self.min_tokens)
        
        # Post-processing: split large chunks
        final_chunks: List[Chunk] = []
        for chunk in all_chunks:
            if chunk.token_count > self.max_tokens:
                split = self._split_large_chunk(
                    chunk, self.max_tokens, self.target_tokens
                )
                final_chunks.extend(split)
            else:
                final_chunks.append(chunk)
        
        # Record metrics
        metrics.record_chunks_created(len(final_chunks))
        
        # Log statistics
        token_counts = [c.token_count for c in final_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        logger.info(
            "Document chunked",
            filename=doc.filename,
            chunks=len(final_chunks),
            avg_tokens=round(avg_tokens, 1),
            min_tokens=min(token_counts) if token_counts else 0,
            max_tokens=max(token_counts) if token_counts else 0,
        )
        
        return final_chunks

    def chunk_all_documents(
        self, documents: List[ParsedDocument]
    ) -> Dict[str, List[Chunk]]:
        """
        Chunk multiple parsed documents.
        
        Args:
            documents: List of parsed documents
            
        Returns:
            Dictionary mapping filename to list of chunks
        """
        results: Dict[str, List[Chunk]] = {}
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                results[doc.filename] = chunks
            except Exception as e:
                logger.error(
                    "Failed to chunk document",
                    filename=doc.filename,
                    error=str(e),
                )
                results[doc.filename] = []
        
        # Summary
        total_chunks = sum(len(chunks) for chunks in results.values())
        
        logger.info(
            "Batch chunking complete",
            documents=len(results),
            total_chunks=total_chunks,
        )
        
        return results
