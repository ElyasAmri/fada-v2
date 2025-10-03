"""Clear CUDA cache"""
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("CUDA cache cleared")
else:
    print("CUDA not available")
