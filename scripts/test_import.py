import sys
import os
from pathlib import Path

# Show current directory and Python path
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Added to path: {project_root}")

# Try import
try:
    from src.darshan import DarshanParser, FeatureExtractor, process_multiple_logs
    print("✓ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
    # Check if files exist
    print("\nChecking files:")
    darshan_dir = project_root / "src" / "darshan"
    print(f"  src/darshan exists: {darshan_dir.exists()}")
    if darshan_dir.exists():
        files = list(darshan_dir.glob("*.py"))
        for f in files:
            print(f"    - {f.name}")
