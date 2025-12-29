
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.embedder import TextEmbedder, Chunk
import torch

def test_embed():
    print("Initializing TextEmbedder...")
    embedder = TextEmbedder(device="cpu", batch_size=1)
    
    print("Creating dummy chunks...")
    chunks = [
        Chunk(
            chunk_id=f"test_{i}",
            text="This is a test chunk " * 100,
            page_number=1,
            section_index=i,
            team="0",
            year="2025",
            binder="test"
        )
        for i in range(10)
    ]
    
    print(f"Embedding {len(chunks)} chunks...")
    results = embedder.embed_chunks(chunks)
    print(f"Successfully embedded. First result dim: {len(results[0].embedding)}")

if __name__ == "__main__":
    test_embed()
