#!/usr/bin/env python
"""
Fixed job processor with correct scaling and simple bottleneck detection
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import NeighborFinder, SubgraphBuilder
from src.models.gat import create_gat_model

class SimpleJobProcessor:
    def __init__(self, model_path, config_path, similarity_path, features_path):
        # Load model
        self.model = self.load_model(model_path, config_path)
        
        # Initialize finder
        self.finder = NeighborFinder(
            similarity_path=similarity_path,
            features_csv_path=features_path,
            similarity_format='pt'
        )
        
        # Initialize builder
        self.builder = SubgraphBuilder(
            edge_construction_method='knn',
            max_edges_per_node=10
        )
        
        self.feature_names = self.finder.feature_names
    
    def load_model(self, checkpoint_path, config_path):
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
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
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def process_job(self, job_features, k=50):
        # Find neighbors
        similarities, indices = self.finder.find_neighbors_for_new_job(job_features, k=k)
        
        # Build subgraph (use top 20)
        subgraph_k = min(20, k)
        neighbor_features = self.finder.training_features[indices[:subgraph_k]]
        
        subgraph = self.builder.build_subgraph(
            query_features=job_features,
            neighbor_features=neighbor_features,
            neighbor_indices=indices[:subgraph_k],
            neighbor_similarities=similarities[:subgraph_k],
            include_neighbor_edges=True
        )
        
        # Augment features
        if subgraph.x.shape[1] == 45:
            augmented = torch.zeros(subgraph.x.shape[0], 4)
            subgraph.x = torch.cat([subgraph.x, augmented], dim=1)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(subgraph.x, subgraph.edge_index)
            pred_raw = predictions[0].item()
            pred_log10 = np.log10(abs(pred_raw) + 1e-10)
        
        # Simple bottleneck detection based on feature values
        bottlenecks = self.identify_bottlenecks_simple(job_features, indices)
        
        # Get neighbor stats
        neighbor_perfs = self.finder.training_performance[indices]
        
        return {
            'prediction_raw': pred_raw,
            'prediction_log10': pred_log10,
            'neighbor_indices': indices.tolist(),
            'neighbor_similarities': similarities.tolist(),
            'neighbor_stats': {
                'mean': float(np.mean(neighbor_perfs)),
                'std': float(np.std(neighbor_perfs)),
                'min': float(np.min(neighbor_perfs)),
                'max': float(np.max(neighbor_perfs))
            },
            'bottlenecks': bottlenecks
        }
    
    def identify_bottlenecks_simple(self, job_features, neighbor_indices):
        """Simple bottleneck identification by comparing to neighbors"""
        
        # Get neighbor features
        neighbor_features = self.finder.training_features[neighbor_indices[:10]]
        neighbor_mean = np.mean(neighbor_features, axis=0)
        
        # Find features where job differs most from neighbors
        diff = job_features - neighbor_mean
        
        # Get features with largest negative difference (potential bottlenecks)
        bottleneck_indices = np.argsort(diff)[:10]  # Bottom 10
        
        bottlenecks = []
        for idx in bottleneck_indices:
            if diff[idx] < -0.5:  # Significantly lower than neighbors
                feature_name = self.feature_names[idx]
                bottlenecks.append({
                    'feature': feature_name,
                    'job_value': job_features[idx],
                    'neighbor_mean': neighbor_mean[idx],
                    'difference': diff[idx]
                })
        
        return bottlenecks

def main():
    print("="*70)
    print("Fixed Job Processing with Bottleneck Detection")
    print("="*70)
    
    # Initialize processor
    processor = SimpleJobProcessor(
        model_path="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt",
        config_path="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/configs/gat_config6.yaml",
        similarity_path="data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt",
        features_path="data/1M/aiio_sample_1000000_normalized.csv"
    )
    
    # Load IOR features
    ior_df = pd.read_csv('ior_all_ranks_features.csv')
    
    all_results = []
    
    for rank in range(len(ior_df)):
        print(f"\n{'='*50}")
        print(f"Processing Rank {rank}")
        print('='*50)
        
        # Get features
        feature_cols = [col for col in ior_df.columns 
                       if col not in ['performance', 'jobid']]
        job_features = ior_df.iloc[rank][feature_cols].values
        actual_perf = ior_df.iloc[rank]['performance']
        
        # Process job
        result = processor.process_job(job_features, k=50)
        result['actual_performance'] = actual_perf
        result['rank'] = rank
        
        # Print results
        print(f"Actual performance: {actual_perf:.4f}")
        print(f"Predicted (log10): {result['prediction_log10']:.4f}")
        print(f"Error: {abs(result['prediction_log10'] - actual_perf):.4f}")
        
        print(f"\nNeighbor statistics:")
        stats = result['neighbor_stats']
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        
        # Print bottlenecks
        if result['bottlenecks']:
            print("\nPotential Bottlenecks (features lower than neighbors):")
            for b in result['bottlenecks'][:5]:
                print(f"  {b['feature']:30s}: {b['job_value']:.2f} (neighbors: {b['neighbor_mean']:.2f})")
        
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print('='*70)
    
    errors = [abs(r['prediction_log10'] - r['actual_performance']) for r in all_results]
    print(f"Average prediction error: {np.mean(errors):.4f}")
    
    # Common bottlenecks
    all_bottlenecks = []
    for r in all_results:
        for b in r['bottlenecks']:
            all_bottlenecks.append(b['feature'])
    
    if all_bottlenecks:
        print("\nMost common bottlenecks:")
        for feat, count in Counter(all_bottlenecks).most_common(5):
            print(f"  {feat}: appears {count} times")
    
    print("\nNote: The model predicts ~39 MB/s (1.59 in log10) vs actual 256 MB/s (2.41 in log10)")
    print("This suggests your IOR workload pattern is quite different from training data.")

if __name__ == "__main__":
    main()