"""
Configuration settings for IO Bottleneck Analyzer
"""
import os
from pathlib import Path
from typing import Optional

# Base paths - can be overridden by environment variables
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_PATH = os.environ.get(
    'IO_MODEL_PATH', 
    str(PROJECT_ROOT / 'data' / '1M' / 'best_model.pt')
)
DEFAULT_DATA_DIR = os.environ.get(
    'IO_DATA_DIR',
    str(PROJECT_ROOT / 'data' / '1M')
)

# Model configuration
MODEL_DEVICE = os.environ.get('IO_MODEL_DEVICE', 'cpu')  # 'cpu' or 'cuda'

# Graph construction parameters
SIMILARITY_THRESHOLD = float(os.environ.get('IO_SIM_THRESHOLD', '0.75'))
K_NEIGHBORS = int(os.environ.get('IO_K_NEIGHBORS', '100'))
SUBGRAPH_SIZE = int(os.environ.get('IO_SUBGRAPH_SIZE', '500'))

# Interpretation method parameters
ATTENTION_THRESHOLD = float(os.environ.get('IO_ATTENTION_THRESHOLD', '0.001'))
GNN_EXPLAINER_THRESHOLD = float(os.environ.get('IO_GNN_THRESHOLD', '0.001'))
GNN_EXPLAINER_NUM_EPOCHS = int(os.environ.get('IO_GNN_EPOCHS', '200'))

# Random seed for reproducibility
RANDOM_SEED = int(os.environ.get('IO_RANDOM_SEED', '42'))

# Paths for data files
def get_similarity_matrix_path(data_dir: str) -> Optional[str]:
    """Find similarity matrix in data directory"""
    data_path = Path(data_dir)
    possible_paths = [
        data_path / 'similarity_output_0.75' / 'similarity_matrix.npz',
        data_path / 'similarity_matrix.npz',
        data_path / 'similarity_graph.npz',
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None

def get_training_features_path(data_dir: str) -> Optional[str]:
    """Find training features in data directory"""
    data_path = Path(data_dir)
    possible_paths = [
        data_path / 'aiio_sample_1000000_normalized.csv',
        data_path / 'aiio_sample_1000000.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None

# Reporting settings
REPORT_OUTPUT_DIR = os.environ.get('IO_REPORT_DIR', './results')
SAVE_FULL_SCORES = True  # Whether to save full scores report

# Logging configuration
LOG_LEVEL = os.environ.get('IO_LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(message)s'
LOG_DATE_FORMAT = '%H:%M:%S'