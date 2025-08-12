# save as check_env.py
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test PyG CUDA
from torch_geometric.nn import GATConv
conv = GATConv(10, 10)
if torch.cuda.is_available():
    conv = conv.cuda()
    x = torch.randn(100, 10).cuda()
    edge_index = torch.randint(0, 100, (2, 500)).cuda()
    out = conv(x, edge_index)
    print("✅ PyG CUDA test passed!")