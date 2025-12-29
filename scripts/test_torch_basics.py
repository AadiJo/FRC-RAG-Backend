
import torch
print("Imported torch")
try:
    a = torch.randn(10, 10)
    print("Created tensor")
    b = a + a
    print("Summed tensor")
    print(f"Result sum: {b.sum()}")
    
    if torch.cuda.is_available():
        print("CUDA available")
        c = a.cuda()
        print("Moved to CUDA")
        d = c + c
        print(f"CUDA Sum: {d.sum()}")
except Exception as e:
    print(f"Failed: {e}")
