"""
Graph builder for creating subgraphs for analysis
"""
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds subgraphs for interpretability analysis"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.75,
                 device: str = 'cpu'):
        """
        Initialize graph builder
        
        Args:
            similarity_threshold: Minimum similarity for edges
            device: Device for tensors
        """
        self.similarity_threshold = similarity_threshold
        self.device = torch.device(device)
        logger.info(f"Graph builder initialized with threshold: {similarity_threshold}")
    
    def create_subgraph(self,
                       new_features: np.ndarray,
                       training_features: Optional[np.ndarray] = None,
                       similarity_matrix: Optional[sp.spmatrix] = None,
                       k_neighbors: int = 100,
                       max_subgraph_size: int = 500) -> Tuple[Data, int]:
        """
        Create subgraph for analysis
        
        Args:
            new_features: Features of the new sample
            training_features: Training features array
            similarity_matrix: Pre-computed similarity matrix
            k_neighbors: Number of neighbors to include
            max_subgraph_size: Maximum subgraph size
            
        Returns:
            Tuple of (PyG Data object, new_node_index)
        """
        # Handle case with no training data
        if training_features is None:
            return self._create_single_node_graph(new_features)
        
        # Find similar neighbors
        neighbors_idx, similarities = self._find_neighbors(
            new_features, training_features, k_neighbors, max_subgraph_size
        )
        
        logger.info(f"Selected {len(neighbors_idx)} neighbors for subgraph")
        
        # Build feature matrix
        subgraph_features = np.vstack([
            training_features[neighbors_idx],
            new_features.reshape(1, -1)
        ])
        new_node_idx = len(neighbors_idx)
        
        # Build edges
        edge_index, edge_attr = self._build_edges(
            new_node_idx, neighbors_idx, similarities, 
            similarity_matrix
        )
        
        # Apply feature augmentation if needed
        features_tensor = self._prepare_features(subgraph_features)
        
        # Create PyG Data object
        data = Data(
            x=features_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        logger.info(f"Created subgraph with {len(subgraph_features)} nodes "
                   f"and {edge_index.shape[1]} edges")
        
        return data, new_node_idx
    
    def _find_neighbors(self, 
                       new_features: np.ndarray,
                       training_features: np.ndarray,
                       k_neighbors: int,
                       max_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k most similar neighbors"""
        new_features = new_features.reshape(1, -1)
        similarities = cosine_similarity(new_features, training_features)[0]
        
        # Get top-k
        top_k = min(max_size, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Filter by threshold
        valid_mask = similarities[top_indices] >= self.similarity_threshold
        selected_indices = top_indices[valid_mask][:k_neighbors]
        
        if len(selected_indices) == 0:
            logger.warning(f"No neighbors above threshold {self.similarity_threshold}, "
                         f"using top {k_neighbors}")
            selected_indices = top_indices[:k_neighbors]
        
        return selected_indices, similarities[selected_indices]
    
    def _build_edges(self,
                    new_node_idx: int,
                    neighbors_idx: np.ndarray,
                    similarities: np.ndarray,
                    similarity_matrix: Optional[sp.spmatrix]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge index and attributes"""
        edges = []
        weights = []
        
        # Add edges from new node to neighbors
        for i, sim in enumerate(similarities):
            edges.extend([[new_node_idx, i], [i, new_node_idx]])
            weights.extend([sim, sim])
        
        # Add edges between neighbors if similarity matrix provided
        if similarity_matrix is not None and len(neighbors_idx) > 1:
            submatrix = similarity_matrix[neighbors_idx][:, neighbors_idx]
            if sp.issparse(submatrix):
                i_idx, j_idx, w = sp.find(submatrix)
                for i, j, weight in zip(i_idx, j_idx, w):
                    if i < j and weight >= self.similarity_threshold:
                        edges.extend([[i, j], [j, i]])
                        weights.extend([weight, weight])
        
        # Handle empty edges case
        if len(edges) == 0:
            edges = [[new_node_idx, new_node_idx]]
            weights = [1.0]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        return edge_index, edge_attr
    
    def _prepare_features(self, features: np.ndarray) -> torch.Tensor:
        """Prepare features with augmentation if needed"""
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Add statistical augmentation if we have 45 base features
        if features_tensor.shape[1] == 45:
            feat_mean = features_tensor.mean(dim=1, keepdim=True)
            feat_std = features_tensor.std(dim=1, keepdim=True)
            feat_min = features_tensor.min(dim=1, keepdim=True)[0]
            feat_max = features_tensor.max(dim=1, keepdim=True)[0]
            features_tensor = torch.cat([
                features_tensor, feat_mean, feat_std, feat_min, feat_max
            ], dim=1)
            logger.debug(f"Augmented features from {features.shape[1]} to {features_tensor.shape[1]}")
        
        return features_tensor
    
    def _create_single_node_graph(self, features: np.ndarray) -> Tuple[Data, int]:
        """Create graph with single node for testing"""
        features = features.reshape(1, -1)
        features_tensor = self._prepare_features(features)
        
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
        edge_attr = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
        
        data = Data(
            x=features_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return data, 0