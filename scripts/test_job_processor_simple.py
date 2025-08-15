#!/usr/bin/env python
"""
Simplified test for job processing - focusing on getting predictions working
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import NeighborFinder, SubgraphBuilder
from src.models.gat import create_gat_model

def load_model(checkpoint_path, config_path):
    """Load the trained GAT model"""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model_config = config.get('model', {})
    model = create_gat_model(
        num_features=49,
        model_type=model_config.get('type', 'standard'),
        hidden_channels=model_config.get('hidden_channels', 256),
        num_layers=model_config.get('num_layers', 3),
        heads=model_config.get('heads', [8, 8, 1]),
        dropout=model_config.get('dropout', 0.2),
        residual=model_config.get('residual', True),
        layer_norm=model_config.get('layer_norm', True),
        feature_augmentation=False,
        dtype=torch.float32
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def test_simple_inference():
    """Test just the inference part"""
    
    print("="*70)
    print("Simplified Phase 3 Test - Focus on Predictions")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = load_model(
        "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt",
        "/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/configs/gat_config6.yaml"
    )
    
    # Initialize components
    print("Initializing components...")
    finder = NeighborFinder(
        similarity_path='data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt',
        features_csv_path='data/1M/aiio_sample_1000000_normalized.csv',
        similarity_format='pt'
    )
    
    builder = SubgraphBuilder(
        edge_construction_method='knn',
        max_edges_per_node=5
    )
    
    # Load IOR features
    print("Loading IOR features...")
    ior_df = pd.read_csv('h5_all_ranks_features.csv')
    
    for rank in range(len(ior_df)):
        print(f"\n{'='*50}")
        print(f"Rank {rank}")
        print('='*50)
        
        # Get features
        feature_cols = [col for col in ior_df.columns 
                       if col not in ['performance', 'jobid']]
        job_features = ior_df.iloc[rank][feature_cols].values
        actual_perf = ior_df.iloc[rank]['performance']
        
        # Find neighbors
        similarities, indices = finder.find_neighbors_for_new_job(job_features, k=20)
        
        # Build subgraph
        neighbor_features = finder.training_features[indices]
        subgraph = builder.build_subgraph(
            query_features=job_features,
            neighbor_features=neighbor_features,
            neighbor_indices=indices,
            neighbor_similarities=similarities,
            include_neighbor_edges=True
        )
        
        # Augment features (45 -> 49)
        if subgraph.x.shape[1] == 45:
            augmented = torch.zeros(subgraph.x.shape[0], 4)
            subgraph.x = torch.cat([subgraph.x, augmented], dim=1)
        
        # Run inference
        with torch.no_grad():
            predictions = model(subgraph.x, subgraph.edge_index)
            pred_raw = predictions[0].item()
            
            # Try different scaling approaches
            pred_log10 = np.log10(abs(pred_raw) + 1e-10) if pred_raw > 0 else pred_raw
            pred_direct = pred_raw
            
            print(f"Actual performance: {actual_perf:.4f}")
            print(f"Predicted (raw): {pred_raw:.4f}")
            print(f"Predicted (log10): {pred_log10:.4f}")
            print(f"Predicted (direct): {pred_direct:.4f}")
            
            # Check which scale is closer
            error_log10 = abs(pred_log10 - actual_perf)
            error_direct = abs(pred_direct - actual_perf)
            
            if error_log10 < error_direct:
                print(f"-> Using log10 scale, error: {error_log10:.4f}")
            else:
                print(f"-> Using direct scale, error: {error_direct:.4f}")
        
        # Show neighbor stats for context
        neighbor_perfs = finder.training_performance[indices]
        print(f"Neighbor mean: {np.mean(neighbor_perfs):.4f}")

if __name__ == "__main__":
    test_simple_inference()