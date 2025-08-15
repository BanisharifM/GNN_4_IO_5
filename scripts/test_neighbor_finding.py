#!/usr/bin/env python
"""
Test neighbor finding and subgraph construction for 1M dataset
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import torch

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import NeighborFinder, SubgraphBuilder

def test_neighbor_finding():
    """Test finding neighbors for IOR job using 1M dataset"""
    
    # Load the extracted IOR features (from Darshan parsing)
    ior_features = pd.read_csv('ior_all_ranks_features.csv')
    print(f"Loaded {len(ior_features)} IOR job features")
    
    # Use first rank as query
    query_job = ior_features.iloc[0]
    
    # Get feature columns (exclude performance and jobid, keep only the 45 features)
    feature_cols = [col for col in ior_features.columns 
                   if col not in ['performance', 'jobid']]
    
    # Verify we have exactly 45 features
    print(f"Number of features: {len(feature_cols)}")
    assert len(feature_cols) == 45, f"Expected 45 features, got {len(feature_cols)}"
    
    query_features = query_job[feature_cols].values
    print(f"Query job performance: {query_job['performance']:.4f}")
    
    # Initialize neighbor finder with 1M dataset and precomputed similarity
    print("\nInitializing neighbor finder with 1M dataset...")
    print("Loading precomputed similarity graph and features...")
    
    finder = NeighborFinder(
        similarity_path='data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt',
        features_csv_path='data/1M/aiio_sample_1000000_normalized.csv',
        similarity_format='pt'
    )
    
    print(f"Loaded {finder.num_nodes:,} nodes from similarity graph")
    
    # Find neighbors for the new job (not in the graph)
    k = 50  # Find top 50 neighbors
    print(f"\nFinding {k} nearest neighbors for new IOR job...")
    
    similarities, indices = finder.find_neighbors_for_new_job(query_features, k=k)
    
    # Show top 10 for readability
    print(f"\nTop 10 neighbor indices: {indices[:10]}")
    print(f"Top 10 similarities: {similarities[:10]}")
    
    # Show neighbor performance values
    if finder.training_performance is not None:
        print("\nNeighbor performance values (top 10):")
        print(f"{'Rank':<6} {'Index':<10} {'Similarity':<12} {'Performance':<12} {'Diff from Query':<15}")
        print("-" * 65)
        
        query_perf = query_job['performance']
        for i in range(min(10, len(indices))):
            neighbor_perf = finder.training_performance[indices[i]]
            diff = neighbor_perf - query_perf
            print(f"{i:<6} {indices[i]:<10} {similarities[i]:<12.4f} {neighbor_perf:<12.4f} {diff:+.4f}")
        
        # Statistics about neighbors
        neighbor_perfs = finder.training_performance[indices]
        print(f"\nNeighbor performance statistics (all {k}):")
        print(f"  Mean: {np.mean(neighbor_perfs):.4f}")
        print(f"  Std:  {np.std(neighbor_perfs):.4f}")
        print(f"  Min:  {np.min(neighbor_perfs):.4f}")
        print(f"  Max:  {np.max(neighbor_perfs):.4f}")
        print(f"  Query: {query_perf:.4f}")
    
    # Return top 10 for subgraph construction
    return finder, query_features, indices[:10], similarities[:10]

def prepare_features_for_model(features: np.ndarray, num_features: int = 49) -> np.ndarray:
    """
    Augment features to match model's expected input
    
    Args:
        features: Original features (45 or already augmented)
        num_features: Expected number of features (49 for your model)
    
    Returns:
        Augmented features
    """
    current_features = features.shape[-1] if features.ndim > 1 else len(features)
    
    if current_features == 45 and num_features == 49:
        print(f"\nAugmenting features from {current_features} to {num_features}")
        # Add 4 placeholder features for now
        # In practice, these would be graph-based features computed from the subgraph
        if features.ndim == 1:
            augmented = np.zeros(4)
            return np.concatenate([features, augmented])
        else:
            augmented = np.zeros((features.shape[0], 4))
            return np.concatenate([features, augmented], axis=1)
    
    return features

def test_subgraph_construction(finder, query_features, indices, similarities):
    """Test subgraph construction"""
    
    print("\n" + "="*60)
    print("Testing Subgraph Construction")
    print("="*60)
    
    # Get neighbor features from training data
    neighbor_features = finder.training_features[indices]
    print(f"Neighbor features shape: {neighbor_features.shape}")
    
    # Build subgraph
    builder = SubgraphBuilder(
        edge_construction_method='knn',
        max_edges_per_node=5,
        similarity_threshold=0.75  # Match your training threshold
    )
    
    subgraph = builder.build_subgraph(
        query_features=query_features,
        neighbor_features=neighbor_features,
        neighbor_indices=indices,
        neighbor_similarities=similarities,
        include_neighbor_edges=True
    )
    
    print(f"\nSubgraph created:")
    print(f"  Nodes: {subgraph.num_nodes}")
    print(f"  Edges: {subgraph.edge_index.shape[1]}")
    print(f"  Node features shape: {subgraph.x.shape}")
    print(f"  Query node mask: {subgraph.query_mask}")
    
    # Check if we need to augment features for the model
    if subgraph.x.shape[1] == 45:
        print("\nAugmenting node features for model (45 -> 49)...")
        augmented_features = prepare_features_for_model(subgraph.x.numpy())
        subgraph.x = torch.tensor(augmented_features, dtype=torch.float32)
        print(f"  Augmented features shape: {subgraph.x.shape}")
    
    # Add global context (optional)
    print("\nAdding global context node...")
    subgraph_with_global = builder.add_global_context(subgraph)
    print(f"  Nodes after global: {subgraph_with_global.num_nodes}")
    print(f"  Edges after global: {subgraph_with_global.edge_index.shape[1]}")
    
    return subgraph

def test_with_existing_node():
    """Test finding neighbors for a node already in the graph"""
    
    print("\n" + "="*60)
    print("Testing with Existing Node from Training Data")
    print("="*60)
    
    # Initialize finder
    finder = NeighborFinder(
        similarity_path='data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt',
        features_csv_path='data/1M/aiio_sample_1000000_normalized.csv',
        similarity_format='pt'
    )
    
    # Pick a random node from training data
    test_node_idx = 1000
    print(f"\nFinding neighbors for existing node {test_node_idx}")
    
    # Find neighbors using precomputed graph
    similarities, indices = finder.find_neighbors_for_existing_node(test_node_idx, k=10)
    
    print(f"Neighbors: {indices}")
    print(f"Similarities: {similarities}")
    
    if finder.training_performance is not None:
        node_perf = finder.training_performance[test_node_idx]
        print(f"\nNode {test_node_idx} performance: {node_perf:.4f}")
        print("Neighbor performances:")
        for i, idx in enumerate(indices):
            print(f"  {idx}: {finder.training_performance[idx]:.4f} (sim={similarities[i]:.4f})")

if __name__ == "__main__":
    print("Testing Phase 2: Neighbor Finding & Subgraph Construction")
    print("="*60)
    print("Dataset: 1M samples")
    print("Model expects: 49 features (45 base + 4 augmented)")
    print("="*60)
    
    try:
        # Test 1: Find neighbors for new IOR job
        finder, query_features, indices, similarities = test_neighbor_finding()
        
        # Test 2: Build subgraph
        subgraph = test_subgraph_construction(finder, query_features, indices, similarities)
        
        # Test 3: Test with existing node (optional)
        print("\n" + "="*60)
        print("Optional: Testing with existing training node")
        print("="*60)
        test_with_existing_node()
        
        print("\n" + "="*60)
        print("✓ Phase 2 testing complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. The subgraph is ready for GAT model inference")
        print("2. Features need augmentation from 45 to 49 before model input")
        print("3. Use your trained model checkpoint from experiments/")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print("\nMake sure these files exist:")
        print("  - ior_all_ranks_features.csv")
        print("  - data/1M/similarity_output_0.75/similarity_graph_20250812_043913.pt")
        print("  - data/1M/aiio_sample_1000000_normalized.csv")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()