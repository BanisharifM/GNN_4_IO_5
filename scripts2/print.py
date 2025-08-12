import os
import torch
import joblib
import sys
import inspect

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tabular import TabGNNTabularModel

# === Paths ===
output_dir = "logs/training/all/Experiment5/combined"
gnn_path = os.path.join(output_dir, "tabgnn_part.pt")
tabular_path = os.path.join(output_dir, "tabular_part.joblib")

print("🔍 Checking GNN checkpoint (.pt)")
if os.path.exists(gnn_path):
    try:
        checkpoint = torch.load(gnn_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            print(f"  ✅ Loaded checkpoint dictionary with keys:")
            for k, v in checkpoint.items():
                if hasattr(v, "shape"):
                    print(f"    - {k}: tensor, shape = {v.shape}")
                else:
                    print(f"    - {k}: {type(v)}")
        else:
            print(f"  ✅ Loaded GNN model directly: {type(checkpoint)}")
    except Exception as e:
        print(f"  ❌ Error loading {gnn_path}: {e}")
else:
    print(f"  ❌ GNN checkpoint not found at {gnn_path}")

print("\n🔍 Checking Tabular model (.joblib)")
if os.path.exists(tabular_path):
    try:
        obj = joblib.load(tabular_path)
        if isinstance(obj, dict):
            print(f"  ✅ Loaded dictionary with keys:")
            for k, v in obj.items():
                print(f"    - {k}: {type(v)}")
        else:
            print(f"  ✅ Loaded model object directly: {type(obj)}")
    except Exception as e:
        print(f"  ❌ Error loading {tabular_path}: {e}")
else:
    print(f"  ❌ Tabular model not found at {tabular_path}")

print("\n🔍 Methods in TabGNNTabularModel:")
methods = [f for f in dir(TabGNNTabularModel) if not f.startswith("__")]
for name in methods:
    try:
        sig = inspect.signature(getattr(TabGNNTabularModel, name))
        print(f"  - {name}{sig}")
    except Exception:
        print(f"  - {name} (no signature found)")
