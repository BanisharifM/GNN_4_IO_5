#!/usr/bin/env python
"""
Neighbor finder using precomputed similarity matrices
"""

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class NeighborFinder:
    """Find k-nearest neighbors using precomputed similarity data"""
    
    def __init__(
        self,
        similarity_path: Optional[str] = None,
        features_csv_path: Optional[str] = None,
        similarity_format: str = 'npz',  # 'npz' or 'pt'
        cache_dir: Optional[str] = None
    ):
        """
        Initialize neighbor finder with precomputed similarity
        
        Args:
            similarity_path: Path to similarity file (.npz or .pt)
            features_csv_path: Path to features CSV for getting feature vectors
            similarity_format: Format of similarity file
            cache_dir: Directory to cache data
        """
        self.similarity_format = similarity_format
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        self.similarity_matrix = None
        self.edge_index = None
        self.edge_weight = None
        self.training_features = None
        self.feature_names = None
        self.num_nodes = 0
        
        if similarity_path:
            self.load_similarity_data(similarity_path)
        
        if features_csv_path:
            self.load_features(features_csv_path)
    
    def load_similarity_data(self, similarity_path: str):
        """
        Load precomputed similarity matrix or edge list
        
        Args:
            similarity_path: Path to similarity file
        """
        logger.info(f"Loading similarity data from {similarity_path}")
        
        if self.similarity_format == 'npz':
            # Load sparse matrix format
            data = np.load(similarity_path, allow_pickle=True)
            
            if 'edge_index' in data:
                # Edge list format
                self.edge_index = data['edge_index']
                self.edge_weight = data['edge_weight'] if 'edge_weight' in data else None
                self.num_nodes = data['num_nodes'] if 'num_nodes' in data else self.edge_index.max() + 1
                logger.info(f"Loaded edge list with {self.edge_index.shape[1]} edges, {self.num_nodes} nodes")
            
            elif 'similarity_matrix' in data:
                # Sparse matrix format
                self.similarity_matrix = sp.load_npz(similarity_path)
                self.num_nodes = self.similarity_matrix.shape[0]
                logger.info(f"Loaded similarity matrix: {self.similarity_matrix.shape}")
                
        elif self.similarity_format == 'pt':
            # PyTorch format
            data = torch.load(similarity_path, map_location='cpu')
            
            if 'edge_index' in data:
                self.edge_index = data['edge_index'].numpy()
                self.edge_weight = data['edge_weight'].numpy() if 'edge_weight' in data else None
                self.num_nodes = data.get('num_nodes', self.edge_index.max() + 1)
                logger.info(f"Loaded PyTorch edge list with {self.edge_index.shape[1]} edges")
            
            elif 'similarity_matrix' in data:
                # Dense matrix in PyTorch
                self.similarity_matrix = data['similarity_matrix'].numpy()
                self.num_nodes = self.similarity_matrix.shape[0]
                logger.info(f"Loaded dense similarity matrix: {self.similarity_matrix.shape}")
    
    def load_features(self, features_csv_path: str):
        """Load feature vectors from CSV"""
        logger.info(f"Loading features from {features_csv_path}")
        
        df = pd.read_csv(features_csv_path)
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['tag', 'performance', 'jobid']]
        self.feature_names = feature_cols
        
        # Store features
        self.training_features = df[feature_cols].values.astype(np.float32)
        self.training_performance = df['tag'].values if 'tag' in df else None
        
        logger.info(f"Loaded {len(self.training_features)} feature vectors")
    
    def find_neighbors_for_existing_node(
        self,
        node_idx: int,
        k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors for a node already in the similarity graph
        
        Args:
            node_idx: Index of node in the graph
            k: Number of neighbors
        
        Returns:
            similarities: Similarity scores to neighbors
            indices: Indices of neighbors
        """
        if self.edge_index is not None:
            # Find from edge list
            return self._find_from_edge_list(node_idx, k)
        elif self.similarity_matrix is not None:
            # Find from similarity matrix
            return self._find_from_matrix(node_idx, k)
        else:
            raise ValueError("No similarity data loaded")
    
    def _find_from_edge_list(self, node_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors from edge list"""
        # Find all edges from this node
        mask = self.edge_index[0] == node_idx
        neighbors = self.edge_index[1][mask]
        
        if self.edge_weight is not None:
            weights = self.edge_weight[mask]
        else:
            weights = np.ones(len(neighbors))
        
        # Sort by weight and take top-k
        if len(neighbors) > k:
            top_k_idx = np.argsort(weights)[-k:]
            neighbors = neighbors[top_k_idx]
            weights = weights[top_k_idx]
        
        return weights, neighbors
    
    def _find_from_matrix(self, node_idx: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors from similarity matrix"""
        if sp.issparse(self.similarity_matrix):
            # Get row as dense array
            row = self.similarity_matrix.getrow(node_idx).toarray().squeeze()
        else:
            row = self.similarity_matrix[node_idx]
        
        # Get top-k similar nodes (excluding self)
        row[node_idx] = -np.inf  # Exclude self
        
        if k >= len(row) - 1:
            top_k_idx = np.argsort(row)[1:]  # All except self
        else:
            top_k_idx = np.argsort(row)[-k:]
        
        similarities = row[top_k_idx]
        
        return similarities, top_k_idx
    
    def find_neighbors_for_new_job(
        self,
        new_job_features: np.ndarray,
        k: int = 50,
        method: str = 'exact'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors for a new job not in the graph
        
        Args:
            new_job_features: Feature vector for new job
            k: Number of neighbors
            method: 'exact' or 'approximate'
        
        Returns:
            similarities: Similarity scores to neighbors
            indices: Indices of neighbors in training data
        """
        if self.training_features is None:
            raise ValueError("Training features not loaded. Call load_features() first.")
        
        if method == 'exact':
            # Compute exact cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            new_job_features = new_job_features.reshape(1, -1)
            similarities = cosine_similarity(new_job_features, self.training_features).squeeze()
            
            # Get top-k
            top_k_idx = np.argsort(similarities)[-k:][::-1]
            top_k_sims = similarities[top_k_idx]
            
            return top_k_sims, top_k_idx
            
        else:  # approximate using LSH or random projection
            # Implement approximate method if needed
            raise NotImplementedError("Approximate method not yet implemented")
    
    def get_subgraph_edges(
        self,
        center_nodes: Union[int, List[int]],
        num_hops: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all edges in k-hop neighborhood of center nodes
        
        Args:
            center_nodes: Node index or list of indices
            num_hops: Number of hops
        
        Returns:
            Edge index and weights for subgraph
        """
        if not isinstance(center_nodes, list):
            center_nodes = [center_nodes]
        
        visited = set(center_nodes)
        current_level = set(center_nodes)
        
        subgraph_edges = []
        subgraph_weights = []
        
        for hop in range(num_hops):
            next_level = set()
            
            for node in current_level:
                # Get neighbors
                if self.edge_index is not None:
                    mask = self.edge_index[0] == node
                    neighbors = self.edge_index[1][mask]
                    
                    for i, neighbor in enumerate(neighbors):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            visited.add(neighbor)
                        
                        # Add edge
                        subgraph_edges.append([node, neighbor])
                        if self.edge_weight is not None:
                            subgraph_weights.append(self.edge_weight[mask][i])
                        else:
                            subgraph_weights.append(1.0)
            
            current_level = next_level
            
            if not current_level:
                break
        
        if subgraph_edges:
            edge_index = np.array(subgraph_edges).T
            edge_weight = np.array(subgraph_weights)
        else:
            edge_index = np.array([[0], [0]])
            edge_weight = np.array([0.0])
        
        return edge_index, edge_weight