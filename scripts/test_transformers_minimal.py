
from transformers import AutoTokenizer, AutoModel
import torch

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    print("Loading model...")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").cuda()
    print("Model loaded on CUDA.")
    
    inputs = tokenizer(["hello world"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print("Success.")
except Exception as e:
    print(f"Failed: {e}")
