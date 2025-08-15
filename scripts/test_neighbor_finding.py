#!/usr/bin/env python
"""
Test neighbor finding and subgraph construction
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference import NeighborFinder, SubgraphBuilder

def test_neighbor_finding():
    """Test finding neighbors for IOR job"""
    
    # Load the extracted IOR features
    ior_features = pd.read_csv('ior_all_ranks_features.csv')
    print(f"Loaded {len(ior_features)} IOR job features")
    
    # Use first rank as query
    query_job = ior_features.iloc[0]
    feature_cols = [col for col in ior_features.columns 
                   if col not in ['tag', 'performance', 'jobid']]
    query_features = query_job[feature_cols].values
    
    print(f"Query job performance: {query_job['performance']:.4f}")
    
    # Initialize neighbor finder with small sample for testing
    print("\nInitializing neighbor finder...")
    finder = NeighborFinder(
        training_features_path='data/100/sample_train_100.csv',
        similarity_metric='cosine'
    )
    
    # Find neighbors
    k = 10
    print(f"\nFinding {k} nearest neighbors...")
    distances, indices = finder.find_neighbors(query_features, k=k)
    
    print(f"Neighbor indices: {indices}")
    print(f"Similarities: {distances}")
    
    # Load training data to see neighbor performance
    train_df = pd.read_csv('data/100/sample_train_100.csv')
    neighbor_data = train_df.iloc[indices]
    
    print("\nNeighbor performance values:")
    for i, (idx, sim) in enumerate(zip(indices, distances)):
        perf = neighbor_data.iloc[i]['tag']
        print(f"  Neighbor {i}: idx={idx}, similarity={sim:.4f}, performance={perf:.4f}")
    
    # Test diverse neighbors
    print("\n\nTesting diverse neighbor selection...")
    diverse_distances, diverse_indices = finder.find_diverse_neighbors(
        query_features, k=k, diversity_weight=0.3
    )
    
    print("Diverse neighbor indices:", diverse_indices)
    print("Diverse similarities:", diverse_distances)
    
    return finder, query_features, indices, distances

def test_subgraph_construction(finder, query_features, indices, distances):
    """Test subgraph construction"""
    
    print("\n" + "="*60)
    print("Testing Subgraph Construction")
    print("="*60)
    
    # Get neighbor features
    neighbor_features = finder.training_features[indices]
    
    # Build subgraph
    builder = SubgraphBuilder(
        edge_construction_method='knn',
        max_edges_per_node=5
    )
    
    subgraph = builder.build_subgraph(
        query_features=query_features,
        neighbor_features=neighbor_features,
        neighbor_indices=indices,
        neighbor_similarities=distances,
        include_neighbor_edges=True
    )
    
    print(f"\nSubgraph created:")
    print(f"  Nodes: {subgraph.num_nodes}")
    print(f"  Edges: {subgraph.edge_index.shape[1]}")
    print(f"  Node features shape: {subgraph.x.shape}")
    print(f"  Query node mask: {subgraph.query_mask}")
    
    # Test with global context
    print("\nAdding global context...")
    subgraph_with_global = builder.add_global_context(subgraph)
    print(f"  Nodes after global: {subgraph_with_global.num_nodes}")
    print(f"  Edges after global: {subgraph_with_global.edge_index.shape[1]}")
    
    return subgraph

if __name__ == "__main__":
    print("Testing Phase 2: Neighbor Finding & Subgraph Construction")
    print("="*60)
    
    try:
        # Test neighbor finding
        finder, query_features, indices, distances = test_neighbor_finding()
        
        # Test subgraph construction
        subgraph = test_subgraph_construction(finder, query_features, indices, distances)
        
        print("\n✓ Phase 2 testing complete!")
        print("\nThe subgraph is now ready for GAT model inference!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()