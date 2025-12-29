
from sentence_transformers import SentenceTransformer
import torch

try:
    print("Loading model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    print("Model loaded.")
    
    print("Encoding...")
    emb = model.encode(["hello world"])
    print(f"Encoded. Shape: {emb.shape}")
except Exception as e:
    print(f"Failed: {e}")
