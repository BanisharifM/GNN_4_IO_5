#!/usr/bin/env python3
"""
Predict I/O performance and identify bottlenecks for unseen IOR job
"""

import pandas as pd
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project to path
sys.path.append('/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5')

from src.models.gat import create_gat_model
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class IORPredictor:
    def __init__(self, 
                 model_checkpoint_path,
                 similarity_graph_path=None,
                 features_csv_path=None,
                 similarity_threshold=0.75,
                 use_cpu=True):  # Force CPU usage
        """
        Initialize predictor with trained model and graph data
        """
        # Force CPU for memory efficiency
        if use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.similarity_threshold = similarity_threshold
        
        # Load model
        self.model, self.checkpoint = self._load_model(model_checkpoint_path)
        
        # Load training data if provided
        self.training_features = None
        self.similarity_matrix = None
        
        if features_csv_path:
            self.training_features = self._load_training_features(features_csv_path)
            print(f"Loaded {len(self.training_features)} training samples")
            
        if similarity_graph_path:
            self.similarity_matrix = self._load_similarity_matrix(similarity_graph_path)
    
    def _load_model(self, checkpoint_path):
        """Load trained GAT model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Detect model configuration from checkpoint
        state_dict = checkpoint['model_state_dict']
        hidden_channels = state_dict['input_proj.weight'].shape[0]
        num_layers = max([int(k.split('.')[1]) for k in state_dict.keys() 
                         if k.startswith('gat_layers.')]) + 1
        
        heads = []
        for i in range(num_layers):
            if f'gat_layers.{i}.gat_conv.att_src' in state_dict:
                n_heads = state_dict[f'gat_layers.{i}.gat_conv.att_src'].shape[1]
                heads.append(n_heads)
        
        print(f"Model configuration: hidden={hidden_channels}, layers={num_layers}, heads={heads}")
        
        # Create model
        model = create_gat_model(
            num_features=45,
            model_type='standard',
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=0.1,
            edge_dim=1,
            residual=True,
            layer_norm=True,
            feature_augmentation=True,
            pool_type='mean',
            dtype=torch.float32
        )
        
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        
        return model, checkpoint
    
    def _load_training_features(self, features_path):
        """Load training features"""
        df = pd.read_csv(features_path)
        # Remove tag column if present
        if 'tag' in df.columns:
            df = df.drop('tag', axis=1)
        return df.values.astype(np.float32)
    
    def _load_similarity_matrix(self, graph_path):
        """Load similarity matrix from .npz file"""
        if os.path.exists(graph_path):
            print(f"Loading similarity matrix from {graph_path}")
            similarity_matrix = sp.load_npz(graph_path)
            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(f"Number of edges: {similarity_matrix.nnz}")
            return similarity_matrix
        return None
    
    def create_subgraph_for_prediction(self, new_features, k_neighbors=50, subgraph_size=500):
        """
        Create a smaller subgraph for efficient prediction
        
        Args:
            new_features: Features of new sample [45]
            k_neighbors: Number of nearest neighbors to connect
            subgraph_size: Maximum size of subgraph to create
        """
        if self.training_features is None:
            # No training data, create single-node graph
            features = new_features.reshape(1, -1)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[1.0]], dtype=torch.float32)
            return features, edge_index, edge_attr, 0
        
        # Calculate similarities between new sample and training samples
        new_features = new_features.reshape(1, -1)
        similarities = cosine_similarity(new_features, self.training_features)[0]
        
        # Find top k most similar samples
        top_k = min(subgraph_size, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]  # Sort by similarity
        
        # Filter by threshold
        valid_mask = similarities[top_indices] >= self.similarity_threshold
        selected_indices = top_indices[valid_mask][:k_neighbors]
        
        if len(selected_indices) == 0:
            # Lower threshold if no neighbors found
            print(f"No neighbors above threshold {self.similarity_threshold}, using top {k_neighbors}")
            selected_indices = top_indices[:k_neighbors]
        
        print(f"Selected {len(selected_indices)} neighbors for subgraph")
        
        # Create subgraph with selected samples plus new sample
        subgraph_features = np.vstack([
            self.training_features[selected_indices],
            new_features
        ])
        new_node_idx = len(selected_indices)
        
        # Create edges between new node and selected neighbors
        edges = []
        weights = []
        
        for i, neighbor_idx in enumerate(selected_indices):
            # Connect new node to neighbor
            edges.append([new_node_idx, i])
            edges.append([i, new_node_idx])
            weights.append(similarities[neighbor_idx])
            weights.append(similarities[neighbor_idx])
        
        # If we have the full similarity matrix, add edges between selected neighbors
        if self.similarity_matrix is not None and len(selected_indices) > 1:
            # Extract submatrix for selected indices
            submatrix = self.similarity_matrix[selected_indices][:, selected_indices]
            if sp.issparse(submatrix):
                submatrix_edges = sp.find(submatrix)
                for i, j, w in zip(submatrix_edges[0], submatrix_edges[1], submatrix_edges[2]):
                    if i < j and w >= self.similarity_threshold:  # Avoid duplicates
                        edges.append([i, j])
                        edges.append([j, i])
                        weights.append(w)
                        weights.append(w)
        
        if len(edges) == 0:
            # Create self-loop if no edges
            edges = [[new_node_idx, new_node_idx]]
            weights = [1.0]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        
        print(f"Created subgraph with {len(subgraph_features)} nodes and {edge_index.shape[1]} edges")
        
        return subgraph_features, edge_index, edge_attr, new_node_idx
    
    def predict_with_interpretation(self, features_path):
        """
        Predict performance and identify bottlenecks for new sample
        """
        # Load new sample
        new_data = pd.read_csv(features_path)
        new_features = new_data.iloc[0, :-1].values  # Exclude tag
        actual_tag = new_data.iloc[0, -1]
        
        # Create subgraph for efficient computation
        print("\n=== Creating subgraph for prediction ===")
        subgraph_features, edge_index, edge_attr, new_node_idx = self.create_subgraph_for_prediction(
            new_features, k_neighbors=50, subgraph_size=500
        )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(subgraph_features).to(self.device)
        
        # Apply feature augmentation if needed
        if self.model.feature_augmentation:
            feat_mean = features_tensor.mean(dim=1, keepdim=True)
            feat_std = features_tensor.std(dim=1, keepdim=True)
            feat_min = features_tensor.min(dim=1, keepdim=True)[0]
            feat_max = features_tensor.max(dim=1, keepdim=True)[0]
            features_tensor = torch.cat([
                features_tensor, feat_mean, feat_std, feat_min, feat_max
            ], dim=1)
        
        # Create graph data
        data = Data(
            x=features_tensor,
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device)
        )
        
        # Make prediction
        print("\n=== Making prediction ===")
        with torch.no_grad():
            # Forward pass
            x = self.model.input_proj(data.x)
            
            for i, gat_layer in enumerate(self.model.gat_layers):
                residual = x
                x, _ = gat_layer(x, data.edge_index, data.edge_attr)
                
                if self.model.residual and i < len(self.model.residual_projs):
                    residual = self.model.residual_projs[i](residual)
                    x = x + residual
                
                if self.model.layer_norm and i < len(self.model.layer_norms):
                    x = self.model.layer_norms[i](x)
                
                if i < self.model.num_layers - 1:
                    x = torch.nn.functional.elu(x)
            
            # Get prediction for new node
            node_features = x[new_node_idx].unsqueeze(0)
            prediction = self.model.predictor(node_features).item()
        
        # Convert from log scale
        predicted_bandwidth = 10**prediction - 1
        actual_bandwidth = 10**actual_tag - 1
        
        print(f"Predicted: {predicted_bandwidth:.2f} MB/s")
        print(f"Actual: {actual_bandwidth:.2f} MB/s")
        print(f"Error: {abs(predicted_bandwidth - actual_bandwidth):.2f} MB/s")
        print(f"Relative error: {abs(predicted_bandwidth - actual_bandwidth) / actual_bandwidth * 100:.1f}%")
        
        # Identify bottlenecks
        print("\n=== Identifying I/O bottlenecks ===")
        bottlenecks = self._identify_bottlenecks_simple(new_features)
        
        return predicted_bandwidth, bottlenecks
    
    def _identify_bottlenecks_simple(self, features):
        """
        Simple bottleneck identification based on feature values
        """
        # Feature names for POSIX counters
        feature_names = [
            'nprocs', 'POSIX_OPENS', 'LUSTRE_STRIPE_SIZE', 'LUSTRE_STRIPE_WIDTH',
            'POSIX_FILENOS', 'POSIX_MEM_ALIGNMENT', 'POSIX_FILE_ALIGNMENT',
            'POSIX_READS', 'POSIX_WRITES', 'POSIX_SEEKS', 'POSIX_STATS',
            'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 'POSIX_CONSEC_READS',
            'POSIX_CONSEC_WRITES', 'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
            'POSIX_RW_SWITCHES', 'POSIX_MEM_NOT_ALIGNED', 'POSIX_FILE_NOT_ALIGNED',
            'POSIX_SIZE_READ_0_100', 'POSIX_SIZE_READ_100_1K', 'POSIX_SIZE_READ_1K_10K',
            'POSIX_SIZE_READ_100K_1M', 'POSIX_SIZE_WRITE_0_100', 'POSIX_SIZE_WRITE_100_1K',
            'POSIX_SIZE_WRITE_1K_10K', 'POSIX_SIZE_WRITE_10K_100K', 'POSIX_SIZE_WRITE_100K_1M',
            'POSIX_STRIDE1_STRIDE', 'POSIX_STRIDE2_STRIDE', 'POSIX_STRIDE3_STRIDE',
            'POSIX_STRIDE4_STRIDE', 'POSIX_STRIDE1_COUNT', 'POSIX_STRIDE2_COUNT',
            'POSIX_STRIDE3_COUNT', 'POSIX_STRIDE4_COUNT', 'POSIX_ACCESS1_ACCESS',
            'POSIX_ACCESS2_ACCESS', 'POSIX_ACCESS3_ACCESS', 'POSIX_ACCESS4_ACCESS',
            'POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT',
            'POSIX_ACCESS4_COUNT'
        ]
        
        # Known bottleneck patterns for small I/O
        bottlenecks = []
        
        # Check for small write size (POSIX_SIZE_WRITE_100_1K)
        if features[25] > 2.5:  # log10(316) ≈ 2.5, indicates many small writes
            bottlenecks.append({
                'feature': 'POSIX_SIZE_WRITE_100_1K',
                'importance': 0.9,
                'value': features[25],
                'original_value': 10**features[25] - 1,
                'issue': 'Small write size (100B-1KB) causing overhead'
            })
        
        # Check for high number of seeks (indicates random access)
        if features[9] > 2.5:  # POSIX_SEEKS
            bottlenecks.append({
                'feature': 'POSIX_SEEKS',
                'importance': 0.8,
                'value': features[9],
                'original_value': 10**features[9] - 1,
                'issue': 'High number of seeks (fsync forcing seeks)'
            })
        
        # Check for file alignment issues
        if features[19] > 2.0:  # POSIX_FILE_NOT_ALIGNED
            bottlenecks.append({
                'feature': 'POSIX_FILE_NOT_ALIGNED',
                'importance': 0.7,
                'value': features[19],
                'original_value': 10**features[19] - 1,
                'issue': 'File alignment issues'
            })
        
        # Check total bytes written vs number of writes
        bytes_written = 10**features[12] - 1 if features[12] > 0 else 0
        num_writes = 10**features[8] - 1 if features[8] > 0 else 0
        if num_writes > 0:
            avg_write_size = bytes_written / num_writes
            if avg_write_size < 4096:  # Less than 4KB average
                bottlenecks.append({
                    'feature': 'Average Write Size',
                    'importance': 0.85,
                    'value': np.log10(avg_write_size + 1),
                    'original_value': avg_write_size,
                    'issue': f'Very small average write size: {avg_write_size:.0f} bytes'
                })
        
        # Sort by importance
        bottlenecks.sort(key=lambda x: x['importance'], reverse=True)
        
        print("\nIdentified Bottlenecks:")
        print("-" * 70)
        for i, b in enumerate(bottlenecks[:5], 1):
            print(f"{i}. {b['feature']:30s} | Importance: {b['importance']:.2f}")
            print(f"   Issue: {b['issue']}")
            print(f"   Value: {b['value']:.2f} (Original: {b['original_value']:.0f})")
            print()
        
        return bottlenecks


def main():
    # Paths
    model_checkpoint = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt'
    
    # Look for files
    data_dir = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M'
    
    # Look for .npz similarity matrix
    similarity_matrix = None
    possible_npz_paths = [
        os.path.join(data_dir, 'similarity_output_0.75', 'similarity_matrix.npz'),
        os.path.join(data_dir, 'similarity_matrix.npz'),
        os.path.join(data_dir, 'similarity_graph.npz'),
    ]
    
    for path in possible_npz_paths:
        if os.path.exists(path):
            similarity_matrix = path
            print(f"Found similarity matrix: {path}")
            break
    
    # Look for normalized features
    training_features = os.path.join(data_dir, 'aiio_sample_1000000_normalized.csv')
    if not os.path.exists(training_features):
        training_features = None
        print("Warning: Training features not found")
    
    # New IOR sample
    test_features = '/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/darshan_features_ior_normalized.csv'
    
    # Initialize predictor (force CPU to avoid memory issues)
    predictor = IORPredictor(
        model_checkpoint_path=model_checkpoint,
        similarity_graph_path=similarity_matrix,
        features_csv_path=training_features,
        similarity_threshold=0.75,
        use_cpu=True  # Force CPU usage
    )
    
    # Make prediction and identify bottlenecks
    predicted_bandwidth, bottlenecks = predictor.predict_with_interpretation(test_features)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Predicted Performance: {predicted_bandwidth:.2f} MB/s")
    print("\nMain Performance Bottlenecks:")
    for i, b in enumerate(bottlenecks[:3], 1):
        if 'issue' in b:
            print(f"  {i}. {b['issue']}")
    

if __name__ == "__main__":
    main()