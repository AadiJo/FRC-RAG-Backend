"""
Text chunking module for user documents.

Provides recursive character text splitting similar to LangChain's
RecursiveCharacterTextSplitter but with no external dependencies.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    text: str
    chunk_index: int
    doc_id: str
    user_id: str
    title: str
    source_type: str
    source_uri: Optional[str]
    metadata: Dict


class TextChunker:
    """
    Recursive text chunker for user documents.
    
    Splits text on hierarchical separators to create semantically
    meaningful chunks while respecting size limits.
    """
    
    # Separators in order of preference (try to split on these first)
    SEPARATORS = [
        "\n\n\n",  # Multiple blank lines (section breaks)
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentences
        ", ",      # Clauses
        " ",       # Words
        "",        # Characters (last resort)
    ]
    
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        user_id: str,
        title: str,
        source_type: str = "manual",
        source_uri: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Full document text
            doc_id: Document identifier
            user_id: User identifier
            title: Document title
            source_type: Source type (e.g., "gdrive", "manual")
            source_uri: Optional source URI
            metadata: Optional additional metadata
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for doc_id={doc_id}")
            return []
        
        # Clean the text
        text = text.strip()
        
        # Split into chunks
        raw_chunks = self._split_text(text, self.SEPARATORS)
        
        # Create Chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            chunks.append(Chunk(
                id=chunk_id,
                text=chunk_text.strip(),
                chunk_index=i,
                doc_id=doc_id,
                user_id=user_id,
                title=title,
                source_type=source_type,
                source_uri=source_uri,
                metadata=metadata or {},
            ))
        
        logger.info(
            "Text chunked",
            doc_id=doc_id,
            total_chars=len(text),
            num_chunks=len(chunks),
            avg_chunk_size=len(text) // max(len(chunks), 1),
        )
        
        return chunks
    
    def _split_text(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """
        Recursively split text using hierarchical separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: split on character boundary
                return self._split_on_size(text)
            
            if sep in text:
                splits = text.split(sep)
                
                # Merge small splits back together
                chunks = []
                current_chunk = ""
                
                for split in splits:
                    # Add separator back except for the first split
                    test_chunk = current_chunk + (sep if current_chunk else "") + split
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk if it has content
                        if current_chunk.strip():
                            # If current chunk is still too big, recurse
                            if len(current_chunk) > self.chunk_size:
                                chunks.extend(self._split_text(current_chunk, separators[i+1:]))
                            else:
                                chunks.append(current_chunk)
                        
                        # Start new chunk with overlap
                        if self.chunk_overlap > 0 and chunks:
                            # Get last part of previous chunk for overlap
                            overlap_text = self._get_overlap(chunks[-1])
                            current_chunk = overlap_text + split
                        else:
                            current_chunk = split
                
                # Don't forget the last chunk
                if current_chunk.strip():
                    if len(current_chunk) > self.chunk_size:
                        chunks.extend(self._split_text(current_chunk, separators[i+1:]))
                    else:
                        chunks.append(current_chunk)
                
                return chunks
        
        # If no separator worked, split on size
        return self._split_on_size(text)
    
    def _split_on_size(self, text: str) -> List[str]:
        """Split text by character count as last resort."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get the overlap portion from the end of a text."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break on word boundary
        overlap = text[-self.chunk_overlap:]
        
        # Find first space to break on word boundary
        space_idx = overlap.find(" ")
        if space_idx > 0:
            overlap = overlap[space_idx + 1:]
        
        return overlap
